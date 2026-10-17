"""Cross-model summary for C2: 7B anchor vs 3B replication.

Reads results/c2_raw.json (7B base), c2_rot2.json (7B rotated),
c2_qwen3b.json and c2_qwen3b_rot2.json. Writes RESULTS-c2-replication.md.

Run: .venv/bin/python code/21_compare_models.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = [
    ("Qwen2.5-7B-Instruct", "c2_raw.json", "c2_rot2.json"),
    ("Qwen2.5-3B-Instruct", "c2_qwen3b.json", "c2_qwen3b_rot2.json"),
    ("Mistral-7B-Instruct-v0.3", "c2_mistral.json", None),
    ("Qwen2.5-7B + validity only", "c2_7b_pressure_validity.json", None),
    ("Qwen2.5-7B + schema prompt", "c2_7b_pressure.json", None),
    ("Qwen2.5-3B + schema prompt", "c2_qwen3b_pressure.json", None),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def load(name: str | None) -> dict:
    if not name:
        return {}
    p = REPO / "results" / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())["summary"]


def row(label: str, k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"| {label} | {k}/{n} | {k/n if n else float('nan'):.3f} | [{lo:.3f}, {hi:.3f}] |"


def main() -> int:
    lines = [
        "# RESULTS — C2 cross-model replication",
        "",
        "Same 420 frozen items, same renderer, greedy, no constrained decoding.",
        "",
    ]
    for model, base_name, rot_name in MODELS:
        b, r = load(base_name), load(rot_name)
        if not b:
            lines += [f"## {model}", "", "Not run yet.", ""]
            continue
        na, nc = b["n_awkward"], b["n_clean"]
        sub = b["readout_1_awkward_in_members"]
        echo = round(b["echo_rate_awkward"] * na)
        absn = round(b["readout_4_abstain_rate"] * na)
        cln_err = round(b["readout_2_clean_error_rate"] * nc)
        eq = b["by_kind"]["equivalent"]
        sh = b["by_kind"]["shifted"]
        lines += [
            f"## {model}",
            "",
            "| readout | count | rate | Wilson 95% CI |",
            "|---|---|---|---|",
            row("silent substitution (awkward)", sub, na),
            row("clean ordinary error", cln_err, nc),
            row("equivalent substituted", eq["substituted"], eq["n"]),
            row("**shifted substituted**", sh["substituted"], sh["n"]),
            row("echoed (loud, schema-invalid)", echo, na),
            row("abstained", absn, na),
            "",
        ]
        if r:
            lines += [
                f"Rotation pass: substitution "
                f"{r['readout_1_awkward_in_members']}/{r['n_awkward']} "
                f"({r['readout_1_substitution_rate']:.3f}); "
                f"shifted {r['by_kind']['shifted']['substituted']}/"
                f"{r['by_kind']['shifted']['n']} "
                f"({r['by_kind']['shifted']['rate']:.3f}).",
                "",
            ]
    lines += [
        "## Reading",
        "",
        "- Qwen2.5-7B is the silent-failure case: schema-valid substitution is "
        "the modal awkward-arm outcome, and it changes meaning on a third of "
        "the shifted items.",
        "- The other two base models are **not** weaker versions of that. "
        "Qwen2.5-3B and Mistral-7B-Instruct both have clean error 0/210 and both "
        "echo instead: they emit the supplied value, the call fails validation, "
        "and the caller gets a signal. Echo and substitution are opposite "
        "conclusions about whether the failure is silent, so these are "
        "qualitative differences, not rate differences.",
        "- Scale does not predict the outcome. Mistral-7B differs from "
        "Qwen2.5-7B by McNemar p ~ 1e-22 on the same items at the same size, "
        "while Qwen2.5-3B sits close to Mistral. The deciding factor is the "
        "model's tool-call policy, not its parameter count.",
        "- The instruction arms show the route is selectable, and that the "
        "validity requirement alone selects it: telling Qwen2.5-7B only that "
        "the call must validate and must not emit a non-member raises silent "
        "meaning changes from 0.347 to 0.778 (clean error still 0). Adding "
        "\"choose the closest member\" takes it to 0.986. The same prompt "
        "leaves the 3B unmoved, so the instruction reveals a policy rather than "
        "creating a competence.",
        "- The claim therefore narrows to: **when a 7B model does resolve the "
        "conflict silently, schema validation cannot see it** (0 false alarms "
        "by construction, because the output is schema-valid). The rate at which "
        "that happens is a property of the model, and it must be measured per "
        "model rather than assumed.",
        "",
    ]
    out = REPO / "RESULTS-c2-replication.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
