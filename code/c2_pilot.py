"""C2 pilot — near-miss enum substitution in required tool-call arguments.

Pre-registration (see PREREGISTRATION-c2.md: readout, control class, and
comparison band are all named there).

CLAIM. When a required enum field cannot be filled with the exact gold member,
the model substitutes a near-miss member rather than abstaining or erroring. The
output is schema-valid, so the framework executes the call, and the caller sees
no signal.

ARMS. Matched twins sharing schema, field, instruction template and record id.
  awkward  value is NOT a member. Gold behaviour: ask or flag.
  clean    value IS the exact member. Gold behaviour: emit it.

READOUTS (all reported; 1 and 2 decide the verdict).
  1. substitution_rate  = P(emitted value in members | awkward arm)
  2. clean_error_rate   = P(emitted value != gold | clean arm)
  3. schema_validity    = P(tool call parses and value in members), both arms
  4. abstain_rate       = P(no tool call, or call with no value), awkward arm
  5. rotation pass      = under enum-order rotation, is the SAME MEMBER chosen
                          (prototypicality) or the SAME INDEX (position
                          artefact)?

FALSIFICATION CONDITION, one sentence, as pre-registered:
  If the near-miss substitution rate on awkward items equals the ordinary error
  rate on matched clean items, there is no substitution effect — only ordinary
  argument error — and the claim is dead.

NEAR-MISS TAGS. "equivalent" paraphrases could be normalised losslessly;
"shifted" ones mean something different from the member they sit near. A
substitution rate carried only by "equivalent" is weak, so the two are split.

Run:
  .venv/bin/python code/c2_pilot.py --limit 4 --out results/c2_smoke.json
  .venv/bin/python code/c2_pilot.py --out results/c2_raw.json
  .venv/bin/python code/c2_pilot.py --rotate 2 --out results/c2_rot2.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "items"))
from c2_items import INSTRUCTION_TEMPLATE, SCHEMAS, build_items, tool_schema  # noqa: E402

REFUSAL_RE = re.compile(
    r"\b(cannot|can't|can not|unable|not a valid|invalid|no valid|"
    r"is not one of|not in the enum|which one|clarif|please (specify|confirm)|"
    r"did you mean|not sure|ambiguous|no matching|does not match)\b",
    re.I,
)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

SYSTEM = (
    "You are a tool-calling assistant. Call the tool with the arguments you are "
    "given. Emit a single tool call."
)


def json_blocks(text: str) -> list[str]:
    """Every outermost balanced {...} block in the text, in order of appearance."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[i : j + 1])
                    i = j + 1
                    break
        else:
            break
    return out


def extract_json(text: str) -> tuple[list[str], str]:
    """Return (candidate json texts, mode). mode is 'tool_call' / 'bare' / 'none'."""
    m = TOOL_CALL_RE.search(text)
    if m:
        return [m.group(1)], "tool_call"
    blocks = json_blocks(text)
    return blocks, "bare" if blocks else "none"


def parse_call(text: str, field: str | None = None) -> dict:
    """Parse the emitted tool call. Returns {status, value, raw_args, mode}.

    Mistral's native template emits `[TOOL_CALLS] [{name, arguments}]` and
    sometimes a second, redundant JSON object after it. Reading candidate blocks
    from the end of the text (the original rule) then picks the wrong object and
    reports a missing field. We instead take every candidate and prefer the one
    that actually carries the field under test, so a stray trailing object cannot
    masquerade as the call.
    """
    raws, mode = extract_json(text)
    if not raws:
        return {"status": "no_call", "value": None, "args": None, "mode": mode}

    cands = []
    for raw in raws:
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        args = obj.get("arguments", obj.get("parameters", obj))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                continue
        if not isinstance(args, dict):
            continue
        cands.append((obj, args))

    if not cands:
        return {"status": "parse_error", "value": None, "args": None, "mode": mode}

    if field:
        for _obj, args in cands:
            if field in args:
                return {"status": "ok", "value": None, "args": args, "mode": mode}
    for obj, args in cands:
        if any(k in obj for k in ("name", "arguments", "parameters")):
            return {"status": "ok", "value": None, "args": args, "mode": mode}
    return {"status": "ok", "value": None, "args": cands[0][1], "mode": mode}


def classify(item: dict, text: str, members: list[str]) -> dict:
    p = parse_call(text, item["field"])
    field = item["field"]
    out = {
        "status": p["status"],
        "emitted": None,
        "emitted_index": None,
        "mode": p["mode"],
    }
    if p["status"] != "ok":
        # Distinguish a deliberate refusal from a format failure.
        if p["status"] == "no_call" and REFUSAL_RE.search(text):
            out["status"] = "abstain"
        out["in_members"] = False
        return out

    args = p["args"]
    if field not in args:
        out["status"] = "missing_field"
        out["in_members"] = False
        return out
    val = args[field]
    if not isinstance(val, str):
        out["status"] = "non_string"
        out["in_members"] = False
        return out
    out["emitted"] = val
    if val in members:
        out["in_members"] = True
        out["emitted_index"] = members.index(val)
    else:
        out["in_members"] = False
        # A refusal written into the value slot.
        if REFUSAL_RE.search(val):
            out["status"] = "abstain_value"
        elif val.strip().lower() == str(item["value"]).strip().lower():
            # The model ignored the enum and copied the supplied value through.
            # This is NOT substitution: the call is schema-invalid, so the
            # framework raises and the caller does get a signal. Kept separate
            # from "invalid" because the two are opposite conclusions about
            # whether the failure is silent.
            out["status"] = "echoed"
        else:
            out["status"] = "invalid"

    if item["arm"] == "clean":
        if val == item["gold_member"]:
            out["status"] = "exact"
        elif out["in_members"]:
            out["status"] = "wrong_member"
    else:  # awkward
        if out["in_members"]:
            out["status"] = "substituted"
    return out


def render(tok, s: dict, item: dict, rotate: int) -> str:
    schema = tool_schema(s, rotate)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": INSTRUCTION_TEMPLATE.format(
                field=item["field"], value=item["value"], rid=item["rid"]
            ),
        },
    ]
    return tok.apply_chat_template(
        msgs, tools=[schema], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="cap items (smoke test)")
    ap.add_argument("--schemas", type=int, default=0, help="cap schemas")
    ap.add_argument("--rotate", type=int, default=0, help="rotate enum order by k")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out", default="results/c2_raw.json")
    ap.add_argument("--system-suffix", default="",
                    help="extra system-prompt text (schema-pressure arm)")
    ap.add_argument("--system", default="",
                    help="replace the system prompt entirely (abstention arm)")
    args = ap.parse_args()

    global SYSTEM
    if args.system:
        SYSTEM = args.system
    if args.system_suffix:
        SYSTEM = f"{SYSTEM} {args.system_suffix}"

    items = build_items()
    if args.schemas:
        items = [i for i in items if i["schema_idx"] < args.schemas]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("no items", file=sys.stderr)
        return 2

    print(
        f"items={len(items)} rotate={args.rotate} model={args.model}", flush=True
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, device_map="cuda"
        )
    except (ValueError, KeyError):
        # Current multimodal checkpoints (e.g. Qwen3.5) are conditional-
        # generation models rather than causal LMs. Text-only prompts still
        # work through the same generate() path.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model, quantization_config=quant, device_map="cuda"
        )
    model.eval()

    # Smoke the template once so a template failure is not 300 failures.
    probe = render(tok, SCHEMAS[0], items[0], args.rotate)
    assert "<tool_call>" in probe or "tools" in probe or "enum" in probe, probe[:400]
    print("--- rendered prompt (first 600 chars) ---", flush=True)
    print(probe[:600], flush=True)
    print("---------------------------------------", flush=True)

    results = []
    for n, item in enumerate(items, 1):
        s = SCHEMAS[item["schema_idx"]]
        members = s["members"]
        if args.rotate:
            k = args.rotate % len(members)
            members = members[k:] + members[:k]
        prompt = render(tok, s, item, args.rotate)
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
        text = tok.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        cls = classify(item, text, members)
        rec = {**item, **cls, "members": members, "text": text}
        if item["arm"] == "awkward" and cls["in_members"]:
            rec["picked_paired_member"] = cls["emitted"] == item["paired_member"]
        results.append(rec)
        if n % 10 == 0 or n == len(items):
            print(f"  {n}/{len(items)}", flush=True)

    summary = summarize(results, args.rotate, args.system_suffix)

    print("\n=== C2 pilot ===", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<34} {v:.4f}", flush=True)
        elif k not in ("by_kind",):
            print(f"  {k:<34} {v}", flush=True)
    print(f"  by_kind {summary['by_kind']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "items": results}, indent=2))
    print(f"\nwrote {out}", flush=True)
    return 0


def summarize(results: list[dict], rotate: int = 0, system_suffix: str = "") -> dict:
    """All readouts, computed from per-item records only.

    Kept separate from generation so stored runs can be re-classified offline
    with the same arithmetic (see code/24_reparse.py).
    """
    awk = [r for r in results if r["arm"] == "awkward"]
    cln = [r for r in results if r["arm"] == "clean"]

    sub = sum(r["status"] == "substituted" for r in awk)
    awk_in = sum(r["in_members"] for r in awk)
    cln_err = sum(r["status"] != "exact" for r in cln)
    cln_wrong = sum(r["status"] == "wrong_member" for r in cln)
    abstain = sum(r["status"] in ("abstain", "abstain_value") for r in awk)
    no_call = sum(r["status"] == "no_call" for r in awk)

    kind_stats = {}
    for kind in ("equivalent", "shifted"):
        ks = [r for r in awk if r["near_miss_kind"] == kind]
        kind_stats[kind] = {
            "n": len(ks),
            "substituted": sum(r["status"] == "substituted" for r in ks),
            "rate": rate(sum(r["status"] == "substituted" for r in ks), len(ks)),
        }

    paired = [r for r in awk if r["in_members"]]
    paired_hit = sum(r.get("picked_paired_member", False) for r in paired)
    echoed = sum(r["status"] == "echoed" for r in awk)

    schema_valid = rate(sum(r["in_members"] for r in results), len(results))
    # Per-arm, because the claim's second half is specifically that the AWKWARD
    # arm's output is schema-valid: the framework executes the call and the
    # caller sees no signal. A schema-validity figure pooled over both arms hides
    # the number that carries the argument.
    schema_valid_awkward = rate(awk_in, len(awk))
    schema_valid_clean = rate(sum(r["in_members"] for r in cln), len(cln))

    return {
        "n_items": len(results),
        "n_awkward": len(awk),
        "n_clean": len(cln),
        "rotate": rotate,
        "system_suffix": system_suffix,
        "readout_1_substitution_rate": rate(sub, len(awk)),
        "readout_1_awkward_in_members": awk_in,
        "readout_2_clean_error_rate": rate(cln_err, len(cln)),
        "readout_2_clean_wrong_member": cln_wrong,
        "readout_2_clean_exact": rate(len(cln) - cln_err, len(cln)),
        "readout_3_schema_validity": schema_valid,
        "readout_3_schema_validity_awkward": schema_valid_awkward,
        "readout_3_schema_validity_clean": schema_valid_clean,
        "readout_4_abstain_rate": rate(abstain, len(awk)),
        "readout_4_no_call_rate": rate(no_call, len(awk)),
        # Not one of the pre-registered readouts. Added during the smoke pass,
        # before any full run, because the smoke showed the two failure modes are
        # opposite: substitution is silent (schema-valid), echo is loud
        # (schema-invalid). The claim's "no signal" half depends on the split.
        "echo_rate_awkward": rate(echoed, len(awk)),
        "by_kind": kind_stats,
        "near_target_hit_rate": rate(paired_hit, len(paired)),
        "awkward_status": dict(Counter(r["status"] for r in awk)),
        "clean_status": dict(Counter(r["status"] for r in cln)),
        "ratio_sub_over_cleanerr": (
            rate(sub, len(awk)) / rate(cln_err, len(cln))
            if cln_err
            else float("inf")
        ),
    }

if __name__ == "__main__":
    raise SystemExit(main())
