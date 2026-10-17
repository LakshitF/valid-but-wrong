"""The literal check, computed exactly as the paper defines it.

`33`/`c2_downstream.json` recorded a reviewer's judgements, not a string
comparison. This script restores the definition in §3.4: flag a call when the
emitted value does not appear verbatim in the instruction. The instruction is
the one `c2_pilot.py` rendered, rebuilt here from the item fields.

Reads `results/c2_raw.json`, writes `results/c2_literal_check.json`.

  .venv/bin/python code/25_literal_check.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
INSTRUCTION = 'Set the {field} to "{value}" for record {rid}.'


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def instruction(item: dict) -> str:
    return INSTRUCTION.format(field=item["field"], value=item["value"], rid=item["rid"])


def flagged(item: dict) -> bool:
    """True when the emitted value is not a verbatim substring of the instruction."""
    return str(item["emitted"]) not in instruction(item)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {
        "n": n,
        "flagged": k,
        "rate": k / n if n else None,
        "ci": [lo, hi],
    }


def main() -> int:
    data = json.loads((RES / "c2_raw.json").read_text())
    items = data["items"]
    awkward = [r for r in items if r["arm"] == "awkward"]
    clean = [r for r in items if r["arm"] == "clean"]

    substituted = [r for r in awkward if r["status"] == "substituted"]
    equivalents = [r for r in substituted if r["near_miss_kind"] == "equivalent"]
    shifted = [r for r in substituted if r["near_miss_kind"] == "shifted"]

    out = {
        "definition": (
            "flag when the emitted value is not a verbatim substring of the "
            "rendered instruction"
        ),
        "instruction_template": INSTRUCTION,
        "substituted": rate(sum(flagged(r) for r in substituted), len(substituted)),
        "equivalent": rate(sum(flagged(r) for r in equivalents), len(equivalents)),
        "shifted": rate(sum(flagged(r) for r in shifted), len(shifted)),
        "clean_control": rate(sum(flagged(r) for r in clean), len(clean)),
        "unflagged_substituted": [
            {"id": r["id"], "value": r["value"], "emitted": r["emitted"],
             "near_miss_kind": r["near_miss_kind"]}
            for r in substituted if not flagged(r)
        ],
        "per_item": [
            {
                "id": r["id"],
                "near_miss_kind": r["near_miss_kind"],
                "value": r["value"],
                "emitted": r["emitted"],
                "flagged": flagged(r),
            }
            for r in substituted
        ],
    }
    (RES / "c2_literal_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
