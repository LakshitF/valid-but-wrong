"""Re-classify stored C2 runs with the corrected tool-call parser.

Why this exists
---------------
The original parser read JSON blocks from the END of the model output. Mistral's
native template emits `[TOOL_CALLS] [{...}]` and sometimes a second, redundant
object after it, so the parser picked the trailing object and scored a correct
call as `missing_field`. The generated `text` is stored per item, so every run
can be re-scored offline with no GPU.

Rules for using this script honestly
------------------------------------
- It only re-parses; it never regenerates and never edits `text`.
- It must be applied to EVERY run, so all comparisons share one parser.
- Diff counts are printed first. If a Qwen run changes, the parser change is not
  the narrow bug fix it claims to be, and the change must be re-examined.

Run:
  .venv/bin/python code/24_reparse.py            # dry run, print diffs
  .venv/bin/python code/24_reparse.py --apply    # rewrite results in place
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from c2_pilot import classify, summarize

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"


def reparse(path: Path) -> tuple[dict, dict, int, list[dict]]:
    d = json.loads(path.read_text())
    old_sum = d["summary"]
    assert d.get("items"), f"{path.name} has no per-item records"
    changed = []
    items = []
    for it in d["items"]:
        members = it["members"]
        cls = classify(it, it["text"], members)
        if it["arm"] == "awkward" and cls["in_members"]:
            cls["picked_paired_member"] = cls["emitted"] == it["paired_member"]
        if cls["status"] != it["status"] or cls.get("emitted") != it.get("emitted"):
            changed.append({
                "id": it["id"], "arm": it["arm"],
                "old": it["status"], "new": cls["status"],
                "old_emitted": it.get("emitted"), "new_emitted": cls.get("emitted"),
                "text": it["text"][:200],
            })
        items.append({**it, **cls})
    new_sum = summarize(
        items,
        rotate=old_sum.get("rotate", 0),
        system_suffix=old_sum.get("system_suffix", ""),
    )
    return {"summary": new_sum, "items": items}, old_sum, len(changed), changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = []
    for p in sorted(RES.glob("c2_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        # Only generation runs: they carry both a summary and per-item records.
        # Downstream and statistics files are derived and are not re-parsed.
        if isinstance(d, dict) and "summary" in d and "items" in d:
            files.append(p)
    if not files:
        print("no c2_*.json files found")
        return 1

    report = ["# RESULTS — C2 parser correction", ""]
    total_changed = 0
    for path in files:
        new, old_sum, n_changed, changed = reparse(path)
        s = new["summary"]
        total_changed += n_changed
        print(f"\n=== {path.name} ===")
        print(f"  changed items: {n_changed}/{len(new['items'])}")
        if n_changed:
            print(f"  transitions: "
                  f"{dict(Counter((c['old'], c['new']) for c in changed))}")
        print(f"  substitution {old_sum['readout_1_substitution_rate']:.3f} -> "
              f"{s['readout_1_substitution_rate']:.3f} "
              f"({s['readout_1_awkward_in_members']}/{s['n_awkward']})")
        print(f"  clean error  {old_sum['readout_2_clean_error_rate']:.3f} -> "
              f"{s['readout_2_clean_error_rate']:.3f} "
              f"({round(s['readout_2_clean_error_rate'] * s['n_clean'])}/{s['n_clean']})")
        print(f"  echo         {old_sum.get('echo_rate_awkward', 0.0):.3f} -> "
              f"{s['echo_rate_awkward']:.3f}")
        print(f"  shifted      {old_sum.get('by_kind', {}).get('shifted', {}).get('rate', 0.0):.3f} -> "
              f"{s['by_kind']['shifted']['rate']:.3f}")

        report += [
            f"## {path.name}",
            "",
            f"- re-classified items: **{n_changed}/{len(new['items'])}**",
            f"- substitution: {old_sum['readout_1_substitution_rate']:.3f} -> "
            f"{s['readout_1_substitution_rate']:.3f}",
            f"- clean error: {old_sum['readout_2_clean_error_rate']:.3f} -> "
            f"{s['readout_2_clean_error_rate']:.3f}",
            f"- echo: {old_sum.get('echo_rate_awkward', 0.0):.3f} -> {s['echo_rate_awkward']:.3f}",
            f"- shifted substitution: "
            f"{old_sum.get('by_kind', {}).get('shifted', {}).get('rate', 0.0):.3f} -> "
            f"{s['by_kind']['shifted']['rate']:.3f}",
            "",
        ]
        if changed:
            report += ["| id | arm | old -> new | text |", "|---|---|---|---|"]
            for c in changed[:12]:
                body = c["text"].replace("|", "\\|").replace("\n", " ")
                report.append(
                    f"| {c['id']} | {c['arm']} | {c['old']} -> {c['new']} | `{body}` |"
                )
            if len(changed) > 12:
                report.append(f"| ... | | {len(changed) - 12} more | |")
            report.append("")

        if args.apply:
            path.write_text(json.dumps(new, indent=2))
            print(f"  wrote {path.name}")

    report += [
        "## Cause",
        "",
        "`extract_json` scanned `{` positions from the end of the output and took "
        "the last balanced block. Mistral's native template emits "
        "`[TOOL_CALLS] [{name, arguments}]` and sometimes a trailing redundant "
        "object, so the trailing object was parsed as the call and its absent "
        "field was scored `missing_field`. The parser now reads candidates in "
        "order and prefers the one that carries the field under test.",
        "",
        f"Total items re-classified: **{total_changed}**.",
        "",
    ]
    (REPO / "RESULTS-c2-reparse.md").write_text("\n".join(report) + "\n")
    print(f"\n[reparse] changed {total_changed} items total -> RESULTS-c2-reparse.md")
    if not args.apply:
        print("[reparse] dry run; pass --apply to rewrite results/c2_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
