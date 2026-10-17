"""C2 LLM-judge comparator baseline on the emitted substitutions.

The two trained comparators are small and fail the 0.70/0.20 operating point.
This script asks the reviewer's natural follow-up: can a current instruction-
following model, given the field description, the supplied value and the
emitted member, do the comparison the trained models could not?

The judge is never shown the equivalent/shifted label. Detection is the rate at
which it calls shifted mappings unacceptable; false alarms are the rate at
which it rejects correct equivalent mappings.

Run (use a model that is not one of the three models under study):
  .venv/bin/python code/47_llm_judge.py \
      --model /mnt/c/Repos/models/Qwen3.5-9B \
      --out results/c2_llm_judge_qwen35_9b.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_items import SCHEMAS  # noqa: E402

SYSTEM = (
    "You audit tool calls. A user supplied a value that is not one of a tool "
    "field's allowed values, and the model emitted an allowed value instead. "
    "Decide whether the emitted value preserves the user's intended meaning. "
    "Answer with exactly one word: ACCEPTABLE or UNACCEPTABLE."
)

USER = """Tool: {name}
Purpose: {description}
Field: {field}
Allowed values: {members}
User supplied: "{value}"
Model emitted: "{emitted}"

Does the emitted value preserve the meaning of the value the user supplied?"""


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def parse_label(text: str) -> str | None:
    t = text.upper()
    # Check the longer token first: it contains "ACCEPTABLE".
    if "UNACCEPTABLE" in t:
        return "unacceptable"
    if "ACCEPTABLE" in t:
        return "acceptable"
    return None


def load_model(path: str):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(path)
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=quant, device_map="cuda"
        )
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            path, quantization_config=quant, device_map="cuda"
        )
    model.eval()
    return tok, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="results/c2_raw.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.load(open(args.base))
    items = [
        r for r in base["items"]
        if r["arm"] == "awkward" and r["status"] == "substituted"
    ]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("no substituted items", file=sys.stderr)
        return 2
    print(f"judging {len(items)} emitted substitutions model={args.model}", flush=True)

    import torch

    tok, model = load_model(args.model)

    rows = []
    for n, r in enumerate(items, 1):
        s = SCHEMAS[r["schema_idx"]]
        msgs = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    name=s["name"],
                    description=s["description"],
                    field=r["field"],
                    members=", ".join(r["members"]),
                    value=r["value"],
                    emitted=r["emitted"],
                ),
            },
        ]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(
            gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        label = parse_label(text)
        rows.append(
            {
                "id": r["id"],
                "schema_idx": r["schema_idx"],
                "field": r["field"],
                "value": r["value"],
                "emitted": r["emitted"],
                "near_miss_kind": r["near_miss_kind"],
                "paired_member": r["paired_member"],
                "judge_text": text,
                "judge_label": label,
            }
        )
        if n % 10 == 0 or n == len(items):
            print(f"  {n}/{len(items)}", flush=True)

    usable = [r for r in rows if r["judge_label"] is not None]
    shift = [r for r in usable if r["near_miss_kind"] == "shifted"]
    equiv = [r for r in usable if r["near_miss_kind"] == "equivalent"]
    det_k = sum(r["judge_label"] == "unacceptable" for r in shift)
    fa_k = sum(r["judge_label"] == "unacceptable" for r in equiv)
    summary = {
        "judge_model": args.model,
        "n_judged": len(rows),
        "n_usable": len(usable),
        "n_unparsed": len(rows) - len(usable),
        "n_shifted": len(shift),
        "n_equivalent": len(equiv),
        "detection_shifted": det_k / len(shift) if shift else None,
        "detection_ci": wilson(det_k, len(shift)),
        "false_alarm_equivalent": fa_k / len(equiv) if equiv else None,
        "false_alarm_ci": wilson(fa_k, len(equiv)),
        "target": {"detection": 0.70, "false_alarm": 0.20},
    }
    print("\n=== C2 LLM judge ===", flush=True)
    for k, v in summary.items():
        print(f"  {k:<26} {v}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "items": rows}, indent=2))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
