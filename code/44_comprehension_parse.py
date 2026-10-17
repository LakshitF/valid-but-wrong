"""Robustness re-parse of the prose comprehension completions.

The primary parser in `40_comprehension.py` credits only a response whose last
line isolates one member. That penalises verbose models: a Mistral answer like
"...the best choice would be \\"issued\\"." restates the option list on the same
line and is scored ambiguous even though the answer is explicit. This script
re-reads the stored completions (no model calls) with a cue-based rule and
reports the primary and cue-based rates side by side, as a post-hoc robustness
check. The primary readout in the paper is unchanged.

Run:
  .venv/bin/python code/44_comprehension_parse.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"

CUE = re.compile(
    r"(?:closest\s+(?:allowed\s+)?(?:value|match|option)|"
    r"best\s+(?:choice|match|option)|"
    r"(?:the\s+)?answer\s+is|"
    r"should\s+(?:be|use|choose)|"
    r"recommend(?:ing)?|suggest(?:ing)?|"
    r"mapped?\s+to|map\s+(?:it|this)\s+to)",
    re.I,
)


def cue_choice(text: str, members: list[str]) -> str | None:
    """First member named after an answer cue, if every cue agrees.

    The window runs to the end of the sentence or 70 characters, whichever
    comes first, so a cue has to be followed by its answer rather than an
    incidental list further on.
    """
    answers: list[str] = []
    for m in CUE.finditer(text):
        window = text[m.end(): m.end() + 70]
        window = re.split(r"[.\n;]", window, maxsplit=1)[0]
        best = None
        for member in members:
            hit = re.search(rf"\b{re.escape(member)}\b", window, re.I)
            if hit and (best is None or hit.start() < best[0]):
                best = (hit.start(), member)
        if best:
            answers.append(best[1])
    answers = list(dict.fromkeys(answers))
    return answers[0] if len(answers) == 1 else None


def main() -> int:
    out = {}
    for tag in ("7b", "3b", "mistral"):
        path = RES / f"c2_comprehension_{tag}.json"
        d = json.loads(path.read_text())
        rows = d["items"]
        n = len(rows)
        primary_member = sum(r["choice"] is not None for r in rows)
        primary_paired = sum(r["chose_paired"] for r in rows)
        cue_pairs = [(r, cue_choice(r["text"], r["members"])) for r in rows]
        # Combined rule: the primary parser's answer when it has one, else the
        # cue-based answer. Terse answers are unaffected; verbose ones can be
        # recovered.
        combined = [
            (r, r["choice"] if r["choice"] is not None else c)
            for r, c in cue_pairs
        ]
        cue_member = sum(c is not None for _, c in combined)
        cue_paired = sum(
            c is not None and c == r["paired_member"] for r, c in combined
        )
        # only count items where the primary parser found nothing
        rescued_member = sum(
            1 for r, c in cue_pairs if r["choice"] is None and c is not None
        )
        rescued_paired = sum(
            1 for r, c in cue_pairs
            if r["choice"] is None and c == r["paired_member"]
        )
        out[tag] = {
            "n": n,
            "primary_member": primary_member,
            "primary_member_rate": primary_member / n,
            "primary_paired": primary_paired,
            "primary_paired_rate": primary_paired / n,
            "cue_member": cue_member,
            "cue_member_rate": cue_member / n,
            "cue_paired": cue_paired,
            "cue_paired_rate": cue_paired / n,
            "rescued_member": rescued_member,
            "rescued_paired": rescued_paired,
        }
    (RES / "c2_comprehension_reparse.json").write_text(json.dumps(out, indent=2))
    for tag, v in out.items():
        print(
            f'{tag}: member {v["primary_member"]}/{v["n"]} -> {v["cue_member"]}/{v["n"]}; '
            f'paired {v["primary_paired"]}/{v["n"]} -> {v["cue_paired"]}/{v["n"]} '
            f'(rescued {v["rescued_member"]} member, {v["rescued_paired"]} paired)'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
