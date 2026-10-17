"""C2 semantic comparator — the missing check, trained.

The downstream arm showed the two available checks fail in opposite directions:
schema validation detects 0% of substitutions; a literal diff detects 86% but
false-alarms on 82% of correct `equivalent` mappings. This script trains a small
cross-encoder to do the semantic comparison and evaluates it on the actual
substituted calls, with item-grouped 5-fold CV.

Input:  results/c2_raw.json (the frozen 420-item base pass)
Output: results/c2_comparator.json, RESULTS-c2-comparator.md

Run:
  .venv/bin/python scripts/20_semantic_comparator.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "items"))
from c2_items import SCHEMAS  # noqa: E402

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEED = 0


def pair_text(item: dict, member: str) -> tuple[str, str]:
    s = SCHEMAS[item["schema_idx"]]
    a = (f"Field: {s['field']}. Description: {s['description']}. "
         f"Instruction value: {item['value']!r}.")
    b = f"Candidate enum member: {member!r}."
    return a, b


def training_rows(items: list[dict]) -> list[dict]:
    """One row per (awkward item, enum member). label=1 iff the mapping is
    acceptable: the item is `equivalent` and the member is its paired target."""
    rows = []
    for it in items:
        if it["arm"] != "awkward":
            continue
        s = SCHEMAS[it["schema_idx"]]
        for m in s["members"]:
            label = int(it["near_miss_kind"] == "equivalent"
                        and m == it["paired_member"])
            a, b = pair_text(it, m)
            rows.append({"item_id": it["id"], "a": a, "b": b,
                         "label": label, "member": m,
                         "kind": it["near_miss_kind"]})
    return rows


def encode(tok, rows: list[dict], max_len: int = 256):
    return tok([r["a"] for r in rows], [r["b"] for r in rows],
               truncation=True, max_length=max_len, padding=True,
               return_tensors="pt")


@torch.no_grad()
def predict(model, tok, rows: list[dict], device, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        enc = encode(tok, rows[i:i + batch])
        enc = {k: v.to(device) for k, v in enc.items()}
        out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).cpu())
    return torch.cat(out).numpy()


def train_fold(train_rows, epochs, lr, pos_weight, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True).to(device)
    enc = encode(tok, train_rows)
    y = torch.tensor([r["label"] for r in train_rows], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(train_rows)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            batch = {k: v[idx].to(device) for k, v in enc.items()}
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, tok


def pick_threshold(y_true, p) -> float:
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        score = balanced_accuracy_score(y_true, (p >= t).astype(int))
        if score > best:
            best, best_t = score, float(t)
    return best_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = json.loads((REPO / "results" / "c2_raw.json").read_text())
    items = raw["items"]
    rows = training_rows(items)
    groups = np.array([r["item_id"] for r in rows])
    y_all = np.array([r["label"] for r in rows])
    pos = int(y_all.sum())
    neg = len(y_all) - pos
    pos_weight = neg / max(1, pos)
    print(f"[comparator] {len(items)} items -> {len(rows)} pairs "
          f"({pos} acceptable / {neg} unacceptable); device={device}", flush=True)

    emitted = [it for it in items
               if it["arm"] == "awkward" and it["status"] == "substituted"]
    by_id = {it["id"]: it for it in items}

    gkf = GroupKFold(n_splits=args.folds)
    fold_metrics, eval_rows = [], []
    for fold, (tr, te) in enumerate(gkf.split(rows, groups=groups), 1):
        train = [rows[i] for i in tr]
        test = [rows[i] for i in te]
        model, tok = train_fold(train, args.epochs, args.lr, pos_weight, device)

        p_train = predict(model, tok, train, device)
        th = pick_threshold(np.array([r["label"] for r in train]), p_train)
        p_test = predict(model, tok, test, device)
        y_test = np.array([r["label"] for r in test])
        auc_pairs = (roc_auc_score(y_test, p_test)
                     if len(set(y_test)) > 1 else float("nan"))

        held_ids = {r["item_id"] for r in test}
        for it in emitted:
            if it["id"] not in held_ids:
                continue
            a, b = pair_text(it, it["emitted"])
            p = float(predict(model, tok, [{"a": a, "b": b}], device)[0])
            eval_rows.append({
                "id": it["id"], "fold": fold, "kind": it["near_miss_kind"],
                "value": it["value"], "emitted": it["emitted"],
                "p_acceptable": p, "flagged_unacceptable": bool(p < th),
                "threshold": th,
            })
        fold_metrics.append({"fold": fold, "n_train": len(train),
                             "n_test": len(test), "auc_pairs": auc_pairs,
                             "threshold": th})
        print(f"[comparator] fold {fold}: pairs AUC={auc_pairs:.3f} "
              f"threshold={th:.2f}", flush=True)

    # Aggregate the actual emitted substitutions (each evaluated by a model that
    # never saw that item).
    shift = [r for r in eval_rows if r["kind"] == "shifted"]
    equiv = [r for r in eval_rows if r["kind"] == "equivalent"]
    detection = (np.mean([r["flagged_unacceptable"] for r in shift])
                 if shift else float("nan"))
    false_alarm = (np.mean([r["flagged_unacceptable"] for r in equiv])
                   if equiv else float("nan"))
    labels = np.array([0 if r["kind"] == "shifted" else 1 for r in eval_rows])
    ps = np.array([r["p_acceptable"] for r in eval_rows])
    auc_emitted = (roc_auc_score(labels, ps)
                   if len(set(labels)) > 1 else float("nan"))

    summary = {
        "model": MODEL_ID,
        "n_items": len(items),
        "n_pairs": len(rows),
        "n_acceptable_pairs": pos,
        "n_unacceptable_pairs": neg,
        "folds": args.folds,
        "epochs": args.epochs,
        "lr": args.lr,
        "fold_metrics": fold_metrics,
        "emitted_evaluated": len(eval_rows),
        "emitted_shifted": len(shift),
        "emitted_equivalent": len(equiv),
        "detection_shifted": detection,
        "false_alarm_equivalent": false_alarm,
        "auc_emitted": auc_emitted,
        "baselines": {
            "schema_validation": {"detection": 0.0, "false_alarm": 0.0},
            "literal_diff": {"detection": 0.861, "false_alarm": 0.822},
        },
        "evaluations": eval_rows,
    }
    out = REPO / "results" / "c2_comparator.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_markdown(summary)
    print(f"[comparator] detection={detection:.3f} false_alarm={false_alarm:.3f} "
          f"AUC_emitted={auc_emitted:.3f} -> {out}", flush=True)
    return 0


def write_markdown(s: dict) -> None:
    f = lambda x: f"{x:.3f}"
    lines = [
        "# RESULTS — C2 semantic comparator (trained)",
        "",
        f"Model `{s['model']}` fine-tuned on {s['n_pairs']} (instruction value, "
        f"candidate member) pairs from the 210 awkward items "
        f"({s['n_acceptable_pairs']} acceptable / "
        f"{s['n_unacceptable_pairs']} unacceptable). "
        f"Item-grouped {s['folds']}-fold CV: every emitted call is scored by a "
        "model that never saw that item.",
        "",
        "## Evaluation on the actual substituted calls",
        "",
        "| check | detection of shifted | false alarms on equivalent |",
        "|---|---|---|",
        f"| schema validation | 0.000 | 0.000 |",
        f"| literal diff vs instruction | 0.861 | 0.822 |",
        f"| **trained semantic comparator** | **{f(s['detection_shifted'])}** "
        f"| **{f(s['false_alarm_equivalent'])}** |",
        "",
        f"AUC on the {s['emitted_evaluated']} emitted calls "
        f"({s['emitted_shifted']} shifted, {s['emitted_equivalent']} equivalent): "
        f"**{f(s['auc_emitted'])}**. Per-fold pair AUC: "
        + ", ".join(f"{m['auc_pairs']:.3f}" for m in s["fold_metrics"]) + ".",
        "",
        "This is a proof of concept on a small, single-annotator item set. The "
        "claim is that the missing semantic check is trainable, not that this "
        "particular model is production-ready.",
        "",
    ]
    (REPO / "RESULTS-c2-comparator.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
