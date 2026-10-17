"""Statistics for the three added controls.

Reads the new result JSONs and writes `results/c2_extra_stats.json`, with Wilson
intervals and paired McNemar tests where the same items are compared.

Run after 40_comprehension.py, the abstention arm and 42_retry.py:
  .venv/bin/python code/43_extra_stats.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import fisher_exact, binomtest

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
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def load(name: str):
    return json.loads((RES / name).read_text())


def rows_by_id(path: str) -> dict:
    d = load(path)
    return {r["id"]: r for r in d["items"]}


def summarise_run(path: str) -> dict:
    d = load(path)
    s = d["summary"]
    awk = [r for r in d["items"] if r["arm"] == "awkward"]
    n = len(awk)
    sub = sum(r["status"] == "substituted" for r in awk)
    echo = sum(r["status"] == "echoed" for r in awk)
    no_call = sum(r["status"] in ("no_call", "abstain") for r in awk)
    abstain = sum(r["status"] in ("abstain", "abstain_value") for r in awk)
    clean = [r for r in d["items"] if r["arm"] == "clean"]
    clean_err = sum(r["status"] != "exact" for r in clean)
    lo, hi = wilson(sub, n)
    out = {
        "n_awkward": n,
        "substitution": sub,
        "substitution_rate": sub / n,
        "substitution_ci": [lo, hi],
        "echo": echo,
        "echo_rate": echo / n,
        "no_call_or_abstain": no_call,
        "no_call_or_abstain_rate": no_call / n,
        "abstain": abstain,
        "abstain_rate": abstain / n,
        "clean_error": clean_err,
        "clean_n": len(clean),
        "by_kind": s.get("by_kind"),
    }
    if "shifted" in (s.get("by_kind") or {}):
        sh = s["by_kind"]["shifted"]
        slo, shi = wilson(sh["substituted"], sh["n"])
        out["shifted_rate"] = sh["rate"]
        out["shifted_ci"] = [slo, shi]
    return out


def paired_substitution(a_path: str, b_path: str) -> dict:
    a, b = rows_by_id(a_path), rows_by_id(b_path)
    only_a = only_b = both = neither = 0
    for i in sorted(set(a) & set(b)):
        if a[i]["arm"] != "awkward":
            continue
        x = a[i]["status"] == "substituted"
        y = b[i]["status"] == "substituted"
        if x and y:
            both += 1
        elif x:
            only_a += 1
        elif y:
            only_b += 1
        else:
            neither += 1
    return {
        "only_a": only_a,
        "only_b": only_b,
        "both": both,
        "neither": neither,
        "mcnemar_p": mcnemar_exact(only_a, only_b),
        "n": both + only_a + only_b + neither,
    }


def main() -> int:
    out = {"abstention": {}, "retry": {}, "comprehension": {}}

    arms = {
        "qwen7b": ("c2_abstain_7b.json", "c2_raw.json"),
        "qwen3b": ("c2_abstain_3b.json", "c2_qwen3b.json"),
        "mistral7b": ("c2_abstain_mistral.json", "c2_mistral.json"),
    }
    for key, (arm, base) in arms.items():
        p = RES / arm
        if not p.exists():
            continue
        out["abstention"][key] = summarise_run(arm)
        out["abstention"][key]["paired_vs_base"] = paired_substitution(arm, base)

    retries = {
        "qwen7b": "c2_retry_7b.json",
        "qwen3b": "c2_retry_3b.json",
        "mistral7b": "c2_retry_mistral.json",
    }
    for key, f in retries.items():
        p = RES / f
        if not p.exists():
            continue
        d = load(f)
        s = d["summary"]
        n = s["n_retried"]
        k = s["converted_to_silent"]
        lo, hi = wilson(k, n)
        by_kind = {}
        for kind in ("equivalent", "shifted"):
            ks = [r for r in d["items"] if r["near_miss_kind"] == kind]
            kk = sum(r["retry_status"] == "substituted" for r in ks)
            by_kind[kind] = {
                "n": len(ks), "substituted": kk,
                "rate": kk / len(ks) if ks else None,
                "ci": list(wilson(kk, len(ks))) if ks else None,
            }
        out["retry"][key] = {
            "n_retried": n,
            "converted_to_silent": k,
            "converted_to_silent_rate": k / n,
            "converted_to_silent_ci": [lo, hi],
            "status": s["status"],
            "by_kind": by_kind,
        }

    comps = {
        "qwen7b": "c2_comprehension_7b.json",
        "qwen3b": "c2_comprehension_3b.json",
        "mistral7b": "c2_comprehension_mistral.json",
    }
    for key, f in comps.items():
        p = RES / f
        if not p.exists():
            continue
        d = load(f)
        rows = d["items"]
        n = len(rows)
        chose = sum(r["choice"] is not None for r in rows)
        paired = sum(r["chose_paired"] for r in rows)
        by_kind = {}
        for kind in ("equivalent", "shifted"):
            ks = [r for r in rows if r["near_miss_kind"] == kind]
            ck = sum(r["choice"] is not None for r in ks)
            pk = sum(r["chose_paired"] for r in ks)
            by_kind[kind] = {
                "n": len(ks),
                "chose_member": ck,
                "chose_member_rate": ck / len(ks),
                "chose_member_ci": list(wilson(ck, len(ks))),
                "chose_paired": pk,
                "chose_paired_rate": pk / len(ks),
                "chose_paired_ci": list(wilson(pk, len(ks))),
            }
        out["comprehension"][key] = {
            "n": n,
            "chose_member": chose,
            "chose_member_rate": chose / n,
            "chose_member_ci": list(wilson(chose, n)),
            "chose_paired": paired,
            "chose_paired_rate": paired / n,
            "chose_paired_ci": list(wilson(paired, n)),
            "chance_paired_rate": sum(1 / len(r["members"]) for r in rows) / n,
            "chose_paired_vs_chance_p": float(
                binomtest(paired, n, sum(1 / len(r["members"]) for r in rows) / n,
                          alternative="greater").pvalue
            ),
            "by_kind": by_kind,
            "base_join": d["summary"].get("base_join"),
        }

        # Of the items the model ECHOED in tool mode, how many had a member
        # available in the prose probe? This is the quantity that decides
        # whether an echo can be attributed to a missing mapping.
        join = d["summary"].get("base_join") or {}
        echo = join.get("echoed")
        if echo:
            # base_join gives rates only; recompute counts from the base file.
            base_key = {"qwen7b": "c2_raw.json", "qwen3b": "c2_qwen3b.json",
                        "mistral7b": "c2_mistral.json"}[key]
            base = load(base_key)
            echoed_ids = {
                r["id"] for r in base["items"]
                if r["arm"] == "awkward" and r["status"] == "echoed"
            }
            er = [r for r in rows if r["id"] in echoed_ids]
            if er:
                em = sum(r["choice"] is not None for r in er)
                ep = sum(r["chose_paired"] for r in er)
                chance = sum(1 / len(r["members"]) for r in er) / len(er)
                out["comprehension"][key]["echoed_subset"] = {
                    "n": len(er),
                    "chose_member": em,
                    "chose_member_rate": em / len(er),
                    "chose_member_ci": list(wilson(em, len(er))),
                    "chose_paired": ep,
                    "chose_paired_rate": ep / len(er),
                    "chose_paired_ci": list(wilson(ep, len(er))),
                    "chance_paired_rate": chance,
                    "chose_paired_vs_chance_p": float(
                        binomtest(ep, len(er), chance, alternative="greater").pvalue
                    ),
                }

    # Cross-model Fisher on "chose any member" (disjoint item sets are not;
    # same items, so this is a descriptive contrast, not a paired test).
    if "qwen7b" in out["comprehension"] and "qwen3b" in out["comprehension"]:
        a = out["comprehension"]["qwen7b"]
        b = out["comprehension"]["qwen3b"]
        table = [[a["chose_member"], a["n"] - a["chose_member"]],
                 [b["chose_member"], b["n"] - b["chose_member"]]]
        out["comprehension"]["fisher_7b_vs_3b_chose_member"] = float(
            fisher_exact(table)[1]
        )

    (RES / "c2_extra_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
