"""Schema-clustered bootstrap for the headline rates.

The 210 awkward items are 14 items in each of 15 schemas. Item-level Wilson
intervals and McNemar tests treat those items as independent; they are not.
This script adds a cluster bootstrap that resamples the 15 schemas with
replacement and recomputes the pooled rates and paired differences.

It is a robustness check, not a replacement for the pre-registered item-level
tests. Writes results/c2_cluster_stats.json.

Run:
  .venv/bin/python code/45_cluster_stats.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
B = 10_000
SEED = 20261017


def load(name: str) -> list[dict]:
    return json.loads((RES / name).read_text())["items"]


def awkward(items: list[dict]) -> list[dict]:
    return [x for x in items if x["arm"] == "awkward"]


def by_schema(items: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for x in items:
        out[x["schema_idx"]].append(x)
    return dict(out)


def rate(items: list[dict], pred) -> tuple[int, int]:
    n = len(items)
    k = sum(1 for x in items if pred(x))
    return k, n


def boot_rate(
    groups: dict[int, list[dict]], pred, rng: random.Random, denom=None
) -> list[float]:
    denom = denom or (lambda x: True)
    keys = list(groups)
    out = []
    for _ in range(B):
        sample = [rng.choice(keys) for _ in keys]
        k = n = 0
        for s in sample:
            for x in groups[s]:
                if denom(x):
                    n += 1
                    k += bool(pred(x))
        out.append(k / n if n else 0.0)
    return out


def boot_diff(
    ga: dict[int, list[dict]], gb: dict[int, list[dict]], pred, rng: random.Random
) -> list[float]:
    keys = sorted(set(ga) & set(gb))
    out = []
    for _ in range(B):
        sample = [rng.choice(keys) for _ in keys]
        ka = na = kb = nb = 0
        for s in sample:
            for x in ga[s]:
                na += 1
                ka += bool(pred(x))
            for x in gb[s]:
                nb += 1
                kb += bool(pred(x))
        out.append((ka / na if na else 0.0) - (kb / nb if nb else 0.0))
    return out


def pct(xs: list[float], lo: float = 2.5, hi: float = 97.5) -> list[float]:
    ys = sorted(xs)
    return [ys[int(len(ys) * lo / 100)], ys[int(len(ys) * hi / 100)]]


def p_two_sided(diffs: list[float]) -> float:
    # bootstrap sign-test style p: twice the smaller tail around zero
    neg = sum(1 for d in diffs if d <= 0)
    pos = len(diffs) - neg
    return min(1.0, 2 * min(neg, pos) / len(diffs))


def main() -> int:
    rng = random.Random(SEED)
    files = {
        "qwen7b": "c2_raw.json",
        "qwen3b": "c2_qwen3b.json",
        "mistral7b": "c2_mistral.json",
    }
    groups = {k: by_schema(awkward(load(f))) for k, f in files.items()}
    sub = lambda x: x["status"] == "substituted"
    shifted = lambda x: x["status"] == "substituted" and x["near_miss_kind"] == "shifted"
    equiv = lambda x: x["status"] == "substituted" and x["near_miss_kind"] == "equivalent"

    out: dict = {"n_boot": B, "seed": SEED, "models": {}, "paired": {}, "kind": {}}
    is_shifted = lambda x: x["near_miss_kind"] == "shifted"
    is_equiv = lambda x: x["near_miss_kind"] == "equivalent"
    for k, g in groups.items():
        items = [x for v in g.values() for x in v]
        kk, nn = rate(items, sub)
        sk = sum(1 for x in items if shifted(x))
        sn = sum(1 for x in items if is_shifted(x))
        ek = sum(1 for x in items if equiv(x))
        en = sum(1 for x in items if is_equiv(x))
        out["models"][k] = {
            "substitution": {
                "k": kk,
                "n": nn,
                "rate": kk / nn,
                "cluster_ci": pct(boot_rate(g, sub, rng)),
            },
            "shifted": {
                "k": sk,
                "n": sn,
                "rate": sk / sn,
                "cluster_ci": pct(boot_rate(g, shifted, rng, is_shifted)),
            },
            "equivalent": {
                "k": ek,
                "n": en,
                "rate": ek / en,
                "cluster_ci": pct(boot_rate(g, equiv, rng, is_equiv)),
            },
        }

    for a, b in [("qwen7b", "qwen3b"), ("qwen7b", "mistral7b"), ("qwen3b", "mistral7b")]:
        d = boot_diff(groups[a], groups[b], sub, rng)
        out["paired"][f"{a}_vs_{b}"] = {
            "diff": out["models"][a]["substitution"]["rate"]
            - out["models"][b]["substitution"]["rate"],
            "cluster_ci": pct(d),
            "p_boot": p_two_sided(d),
        }

    # Equivalent minus shifted, within the anchor.
    g = groups["qwen7b"]
    keys = list(g)
    diffs = []
    for _ in range(B):
        sample = [rng.choice(keys) for _ in keys]
        ek = en = sk = sn = 0
        for s in sample:
            for x in g[s]:
                if x["near_miss_kind"] == "equivalent":
                    en += 1
                    ek += x["status"] == "substituted"
                elif x["near_miss_kind"] == "shifted":
                    sn += 1
                    sk += x["status"] == "substituted"
        diffs.append(ek / en - sk / sn)
    out["kind"]["qwen7b_equivalent_minus_shifted"] = {
        "diff": out["models"]["qwen7b"]["equivalent"]["rate"]
        - out["models"]["qwen7b"]["shifted"]["rate"],
        "cluster_ci": pct(diffs),
        "p_boot": p_two_sided(diffs),
    }

    (RES / "c2_cluster_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
