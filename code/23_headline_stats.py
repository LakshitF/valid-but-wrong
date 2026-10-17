"""Paired tests for the numbers quoted in the paper.

Writes results/c2_headline_stats.json. Nothing here is hand-copied into the
draft; the draft reads these values.

Run:
  .venv/bin/python scripts/23_headline_stats.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def load(name: str):
    return json.loads((RES / name).read_text())


def main() -> int:
    out = {}
    runs = {
        "qwen7b": "c2_raw.json",
        "qwen3b": "c2_qwen3b.json",
        "qwen3b_pressure": "c2_qwen3b_pressure.json",
        "qwen7b_pressure": "c2_7b_pressure.json",
        "mistral7b": "c2_mistral.json",
    }
    items = {}
    for key, f in runs.items():
        p = RES / f
        if not p.exists():
            continue
        d = load(f)
        items[key] = {i["id"]: i for i in d["items"]}
        s = d["summary"]
        n = s["n_awkward"]
        sub = s["readout_1_awkward_in_members"]
        lo, hi = wilson(sub, n)
        out[key] = {
            "n_awkward": n,
            "substitution": sub,
            "substitution_rate": sub / n,
            "substitution_ci": [lo, hi],
            "echo": round(s["echo_rate_awkward"] * n),
            "echo_rate": s["echo_rate_awkward"],
            "clean_error": s["readout_2_clean_wrong_member"],
            "clean_n": s["n_clean"],
            "equivalent": s["by_kind"]["equivalent"],
            "shifted": s["by_kind"]["shifted"],
            "abstain": s["readout_4_abstain_rate"],
        }

    # Paired model comparisons on identical items.
    def paired(a: str, b: str) -> dict | None:
        if a not in items or b not in items:
            return None
        common = sorted(set(items[a]) & set(items[b]))
        both = only_a = only_b = neither = 0
        for i in common:
            x = items[a][i]["arm"] == "awkward" and items[a][i]["status"] == "substituted"
            y = items[b][i]["arm"] == "awkward" and items[b][i]["status"] == "substituted"
            if not (items[a][i]["arm"] == "awkward"):
                continue
            if x and y:
                both += 1
            elif x:
                only_a += 1
            elif y:
                only_b += 1
            else:
                neither += 1
        return {"only_a": only_a, "only_b": only_b, "both": both,
                "neither": neither, "mcnemar_p": mcnemar_exact(only_a, only_b),
                "n_items": both + only_a + only_b + neither}

    out["paired_qwen7b_vs_qwen3b"] = paired("qwen7b", "qwen3b")
    out["paired_qwen3b_vs_pressure"] = paired("qwen3b", "qwen3b_pressure")
    out["paired_qwen7b_vs_pressure"] = paired("qwen7b", "qwen7b_pressure")
    out["paired_qwen7b_vs_mistral"] = paired("qwen7b", "mistral7b")

    # Equivalent vs shifted within the 7B run (unpaired items, so Fisher).
    if "qwen7b" in out:
        eq, sh = out["qwen7b"]["equivalent"], out["qwen7b"]["shifted"]
        table = [[eq["substituted"], eq["n"] - eq["substituted"]],
                 [sh["substituted"], sh["n"] - sh["substituted"]]]
        _, p = fisher_exact(table)
        out["qwen7b_equivalent_vs_shifted_fisher_p"] = float(p)
        out["qwen7b_equivalent_vs_shifted_table"] = table

    # Per-schema range for the 7B run.
    if "qwen7b" in items:
        agg = {}
        for i in items["qwen7b"].values():
            if i["arm"] != "awkward":
                continue
            a = agg.setdefault(i["schema_idx"], [0, 0])
            a[1] += 1
            a[0] += int(i["status"] == "substituted")
        rates = [v[0] / v[1] for v in agg.values()]
        out["qwen7b_per_schema"] = {
            "n_schemas": len(agg),
            "min": min(rates), "max": max(rates),
            "counts": {str(k): v for k, v in sorted(agg.items())},
        }

    (RES / "c2_headline_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k.startswith("paired") or "fisher" in k}, indent=2))
    for k in out:
        if k in runs.values() or k in ("qwen7b", "qwen3b"):
            v = out[k]
            print(f"{k}: sub={v['substitution']}/{v['n_awkward']} "
                  f"({v['substitution_rate']:.3f}) echo={v['echo_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
