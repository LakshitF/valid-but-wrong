"""C2 semantic comparator, attempt 2: sentence embeddings.

Attempt 1 (`20_semantic_comparator.py`) fine-tuned an MS MARCO cross-encoder and
failed G2 (detection 0.400 at 32.2% false alarms, AUC 0.589). MS MARCO is a
passage-relevance task, which is a poor transfer for deciding whether an enum
member is an acceptable rendering of a prose value. This attempt uses a
sentence embedder that is already cached, with the same items, the same labels
and the same item-grouped folds, so the two attempts are directly comparable.

Two heads are reported:
  (a) raw cosine similarity with a threshold picked on the training folds;
  (b) the same cosine plus lexical features into a logistic regression.

Input:  results/c2_raw.json
Output: results/c2_comparator_bge.json, RESULTS-c2-comparator-bge.md

Run:
  .venv/bin/python code/22_bge_comparator.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "items"))
from c2_items import SCHEMAS  # noqa: E402

MODEL_ID = "BAAI/bge-small-en-v1.5"
SEED = 0
FOLDS = 5


def pair_text(item: dict, member: str) -> tuple[str, str]:
    s = SCHEMAS[item["schema_idx"]]
    return (f"{s['field']}: {s['description']}. Value: {item['value']}.",
            f"{member}")


def training_rows(items: list[dict]) -> list[dict]:
    rows = []
    for it in items:
        if it["arm"] != "awkward":
            continue
        s = SCHEMAS[it["schema_idx"]]
        for m in s["members"]:
            label = int(it["near_miss_kind"] == "equivalent"
                        and m == it["paired_member"])
            rows.append({"item_id": it["id"], "schema_idx": it["schema_idx"],
                         "value": it["value"], "member": m, "label": label})
    return rows


def lexical(a: str, b: str) -> list[float]:
    """Cheap surface features: exact match, token containment, char overlap."""
    ta, tb = a.lower().split(), b.lower().split()
    sa, sb = set(ta), set(tb)
    inter = len(sa & sb)
    jac = inter / max(1, len(sa | sb))
    return [float(a.strip().lower() == b.strip().lower()),
            float(b.strip().lower() in a.lower()),
            inter / max(1, len(sb)),
            jac]


class Encoder:
    """BGE with CLS pooling and L2 normalisation, via transformers directly."""

    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    def encode(self, texts: list[str], batch_size: int = 64,
               **_) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], padding=True,
                               truncation=True, max_length=256,
                               return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state[:, 0]  # CLS
                out.append(self.torch.nn.functional.normalize(h, dim=-1).cpu())
        return self.torch.cat(out).numpy()


def extract_len(items, encoder, batch: int = 64) -> tuple[list[dict], np.ndarray]:
    rows = training_rows(items)
    a_texts = [pair_text(r, r["member"])[0] for r in rows]
    b_texts = [pair_text(r, r["member"])[1] for r in rows]
    ea = encoder.encode(a_texts, batch_size=batch)
    eb = encoder.encode(b_texts, batch_size=batch)
    cos = np.sum(ea * eb, axis=1)
    X = np.array([[c] + lexical(a, b)
                  for c, a, b in zip(cos, a_texts, b_texts)], dtype=np.float64)
    return rows, X


def pick_threshold(y, p) -> float:
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        score = balanced_accuracy_score(y, (p >= t).astype(int))
        if score > best:
            best, best_t = score, float(t)
    return best_t


def main() -> int:
    np.random.seed(SEED)
    raw = json.loads((REPO / "results" / "c2_raw.json").read_text())
    items = raw["items"]
    enc = Encoder(MODEL_ID)

    rows, X_all = extract_len(items, enc)
    groups = np.array([r["item_id"] for r in rows])
    y_all = np.array([r["label"] for r in rows])
    emitted = [it for it in items
               if it["arm"] == "awkward" and it["status"] == "substituted"]

    # Pre-embed every emitted call once.
    emit_text = [(pair_text(it, it["emitted"]), it, it["emitted"])
                 for it in emitted]
    ea = enc.encode([t[0][0] for t in emit_text], normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False)
    eb = enc.encode([t[0][1] for t in emit_text], normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False)
    emit_cos = np.sum(ea * eb, axis=1)
    emit_X = np.array([[c] + lexical(a, b) for c, (a, b), _ in
                       zip(emit_cos, [t[0] for t in emit_text], emit_text)],
                      dtype=np.float64)

    gkf = GroupKFold(n_splits=FOLDS)
    out = {"cosine": [], "lr": []}
    fold_metrics = []
    for fold, (tr, te) in enumerate(gkf.split(X_all, groups=groups), 1):
        held = {rows[i]["item_id"] for i in te}
        idx = [j for j, t in enumerate(emit_text) if t[1]["id"] in held]

        # (a) raw cosine, threshold on train fold
        thr = pick_threshold(y_all[tr], X_all[tr, 0])
        for j in idx:
            out["cosine"].append({
                "id": emit_text[j][1]["id"], "fold": fold,
                "kind": emit_text[j][1]["near_miss_kind"],
                "p_acceptable": float(emit_X[j, 0]),
                "flagged": bool(emit_X[j, 0] < thr), "threshold": thr})

        # (b) logistic regression on cosine + lexical
        sc = StandardScaler().fit(X_all[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                random_state=SEED)
        lr.fit(sc.transform(X_all[tr]), y_all[tr])
        p_tr = lr.predict_proba(sc.transform(X_all[tr]))[:, 1]
        thr_lr = pick_threshold(y_all[tr], p_tr)
        p_te = lr.predict_proba(sc.transform(X_all[te]))[:, 1]
        y_te = y_all[te]
        auc_p = roc_auc_score(y_te, p_te) if len(set(y_te)) > 1 else float("nan")
        for j in idx:
            p = float(lr.predict_proba(sc.transform(emit_X[j:j + 1]))[0, 1])
            out["lr"].append({
                "id": emit_text[j][1]["id"], "fold": fold,
                "kind": emit_text[j][1]["near_miss_kind"],
                "p_acceptable": p, "flagged": bool(p < thr_lr),
                "threshold": thr_lr})
        # Also report the cosine head's pair-level AUC for a like-for-like
        # comparison with attempt 1.
        auc_cos = (roc_auc_score(y_te, X_all[te, 0])
                   if len(set(y_te)) > 1 else float("nan"))
        fold_metrics.append({"fold": fold, "pair_auc_lr": auc_p,
                             "pair_auc_cosine": auc_cos,
                             "threshold_cosine": thr,
                             "threshold_lr": thr_lr})
        print(f"[bge] fold {fold}: pair AUC lr={auc_p:.3f} "
              f"cos={auc_cos:.3f}", flush=True)

    summary = {"model": MODEL_ID, "folds": FOLDS, "fold_metrics": fold_metrics}
    for head in ("cosine", "lr"):
        rs = out[head]
        shift = [r for r in rs if r["kind"] == "shifted"]
        equiv = [r for r in rs if r["kind"] == "equivalent"]
        labels = np.array([0 if r["kind"] == "shifted" else 1 for r in rs])
        ps = np.array([r["p_acceptable"] for r in rs])
        summary[head] = {
            "n_evaluated": len(rs),
            "n_shifted": len(shift),
            "n_equivalent": len(equiv),
            "detection_shifted": float(np.mean([r["flagged"] for r in shift])),
            "false_alarm_equivalent": float(np.mean([r["flagged"] for r in equiv])),
            "auc_emitted": float(roc_auc_score(labels, ps))
            if len(set(labels)) > 1 else float("nan"),
            "evaluations": rs,
        }
        print(f"[bge] {head}: detection="
              f"{summary[head]['detection_shifted']:.3f} FA="
              f"{summary[head]['false_alarm_equivalent']:.3f} "
              f"AUC={summary[head]['auc_emitted']:.3f}", flush=True)

    (REPO / "results" / "c2_comparator_bge.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    write_markdown(summary)
    return 0


def write_markdown(s: dict) -> None:
    f = lambda x: f"{x:.3f}"
    lines = [
        "# RESULTS — C2 semantic comparator, attempt 2 (sentence embeddings)",
        "",
        f"`{s['model']}`, item-grouped {s['folds']}-fold CV, same items, labels "
        "and folds as attempt 1.",
        "",
        "| head | detection of shifted | false alarms on equivalent | AUC (emitted) |",
        "|---|---|---|---|",
        f"| raw cosine | {f(s['cosine']['detection_shifted'])} "
        f"| {f(s['cosine']['false_alarm_equivalent'])} "
        f"| {f(s['cosine']['auc_emitted'])} |",
        f"| cosine + lexical LR | {f(s['lr']['detection_shifted'])} "
        f"| {f(s['lr']['false_alarm_equivalent'])} "
        f"| {f(s['lr']['auc_emitted'])} |",
        "",
        "Per-fold pair AUC (LR / cosine): "
        + ", ".join(f"{m['pair_auc_lr']:.3f}/{m['pair_auc_cosine']:.3f}"
                    for m in s["fold_metrics"]) + ".",
        "",
        "Reference: schema validation 0.000/0.000; literal diff 0.861/0.822; "
        "attempt 1 (MS MARCO cross-encoder) 0.400/0.322, AUC 0.589.",
        "",
    ]
    (REPO / "RESULTS-c2-comparator-bge.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
