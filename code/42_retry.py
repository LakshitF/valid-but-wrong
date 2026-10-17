"""C2 retry control — does a validation retry convert loud failure into silence?

The paper calls the echoing models "loud" because their call is schema-invalid,
so a real pipeline raises. Real pipelines also retry. This pass replays every
echoed awkward call once through a realistic validation-retry turn and asks what
the model does next. If echoed calls become schema-valid substitutions on retry,
the loud/silent distinction between models is an artifact of the no-retry
harness, and the deployment risk is uniform.

The retry message states the validation error and asks for a response to the
original instruction. It does not tell the model to pick a member, so abstention
and a second echo remain available.

Run:
  .venv/bin/python code/42_retry.py --model Qwen/Qwen2.5-3B-Instruct \
      --base results/c2_qwen3b.json --out results/c2_retry_qwen3b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_items import INSTRUCTION_TEMPLATE, SCHEMAS, tool_schema  # noqa: E402
from c2_pilot import SYSTEM, classify  # noqa: E402

RETRY_TEMPLATE = (
    'Your previous tool call was rejected by schema validation: the value '
    '"{value}" is not one of the allowed values ({members}). '
    "Respond to the original instruction."
)

# The no-list arm omits the allowed values. The base retry lists them, as a
# validator's error commonly does, which makes the option set salient at the
# moment the model revises; this arm separates schema pressure from salience.
RETRY_TEMPLATE_NO_LIST = (
    'Your previous tool call was rejected by schema validation: the value '
    '"{value}" is not one of the allowed values for this field. '
    "Respond to the original instruction."
)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    rates = {}
    for status in ("substituted", "echoed", "invalid", "abstain", "no_call",
                   "missing_field", "non_string", "parse_error"):
        k = sum(r["retry_status"] == status for r in rows)
        rates[status] = k
        rates[f"{status}_rate"] = k / n if n else None
    return {
        "n_retried": n,
        "converted_to_silent": rates["substituted"],
        "converted_to_silent_rate": rates["substituted_rate"],
        "status": {k: v for k, v in rates.items() if not k.endswith("_rate")},
        "by_kind": {
            kind: {
                "n": sum(r["near_miss_kind"] == kind for r in rows),
                "substituted": sum(
                    r["near_miss_kind"] == kind
                    and r["retry_status"] == "substituted"
                    for r in rows
                ),
            }
            for kind in ("equivalent", "shifted")
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True, help="base result JSON with echoes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument(
        "--no-list",
        action="store_true",
        help="omit the allowed values from the retry message",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.load(open(args.base))
    items = [
        r for r in base["items"]
        if r["arm"] == "awkward" and r["status"] == "echoed"
    ]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("no echoed items to retry", file=sys.stderr)
        return 2
    print(f"base={args.base} echoed={len(items)} model={args.model}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="cuda",
    )
    model.eval()

    rows = []
    for n, r in enumerate(items, 1):
        s = SCHEMAS[r["schema_idx"]]
        members = s["members"]
        schema = tool_schema(s, 0)
        msgs = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": INSTRUCTION_TEMPLATE.format(
                    field=r["field"], value=r["value"], rid=r["rid"]
                ),
            },
            {"role": "assistant", "content": r["text"]},
            {
                "role": "user",
                "content": (
                    RETRY_TEMPLATE_NO_LIST.format(value=r["emitted"])
                    if args.no_list
                    else RETRY_TEMPLATE.format(
                        value=r["emitted"], members=", ".join(members)
                    )
                ),
            },
        ]
        prompt = tok.apply_chat_template(
            msgs, tools=[schema], tokenize=False, add_generation_prompt=True
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
        )
        cls = classify(r, text, members)
        # classify() sets status from the item arm; for retry we care only about
        # the outcome of the corrected call.
        rows.append(
            {
                "id": r["id"],
                "schema_idx": r["schema_idx"],
                "field": r["field"],
                "value": r["value"],
                "near_miss_kind": r["near_miss_kind"],
                "paired_member": r["paired_member"],
                "original_status": r["status"],
                "original_emitted": r["emitted"],
                "original_text": r["text"],
                "retry_status": cls["status"],
                "retry_emitted": cls.get("emitted"),
                "retry_in_members": cls.get("in_members"),
                "retry_text": text,
            }
        )
        if n % 10 == 0 or n == len(items):
            print(f"  {n}/{len(items)}", flush=True)

    summary = summarize(rows)
    print("\n=== C2 validation retry ===", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<34} {v:.4f}", flush=True)
        else:
            print(f"  {k:<34} {v}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "items": rows}, indent=2))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
