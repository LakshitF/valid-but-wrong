"""C2 base arm through the OpenRouter API.

Same items, prompts, tool schema, decoding settings and taxonomy as
`c2_pilot.py`, but inference runs on a hosted provider instead of the local
4-bit model. The local Qwen3.5-9B run was abandoned because the hybrid-
attention kernels fall back to PyTorch on this box and the 420-item pass
needed about six hours; the API finishes it in minutes for roughly two cents.

This is a separate arm, not a drop-in replacement for the local runs: the
provider's quantisation, hardware and sampler are not the local NF4/greedy
setup, and the run is marked as such in its output. Keep it out of any table
that claims a single local protocol unless the difference is stated.

One more difference belongs in any report: the hosted tool-call path appears
to constrain arguments to a declared `enum`. An explicit request to emit a
non-member value still returns a member, and dropping `enum` from the schema
restores the echo. Run `--no-enum` for the open-field control; the two
hosted arms together are a constrained-decoding experiment, not a free-choice
replication.

Run:
  OPENROUTER_API_KEY=... .venv/bin/python code/48_openrouter.py \
      --model qwen/qwen3.5-9b --out results/c2_qwen35_9b_api.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_items import INSTRUCTION_TEMPLATE, SCHEMAS, build_items, tool_schema  # noqa: E402
from c2_pilot import SYSTEM, classify, summarize  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"


def call(payload: dict, key: str, timeout: int) -> dict:
    """POST with retries on 429/5xx/network errors. Returns the JSON body."""
    delay = 2.0
    last = None
    for attempt in range(6):
        try:
            r = requests.post(
                API,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/local/valid-but-wrong",
                    "X-Title": "valid-but-wrong",
                },
                json=payload,
                timeout=timeout,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"{r.status_code} {r.text[:200]}"
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 400 and (
                "chat_template_kwargs" in r.text or "reasoning" in r.text
            ):
                # Some providers reject the optional body fields. Retry without
                # them rather than losing the item.
                payload = {
                    k: v
                    for k, v in payload.items()
                    if k not in ("chat_template_kwargs", "reasoning")
                }
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network, timeout, JSON
            last = repr(e)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"request failed after retries: {last}")


def message_text(msg: dict) -> str:
    """The model's call as text, in a shape `c2_pilot.parse_call` understands."""
    calls = msg.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return json.dumps({"name": fn.get("name"), "arguments": args})
    return msg.get("content") or ""


def schema_for(item: dict, s: dict, enum_on: bool) -> dict:
    """The tool definition, optionally with the closed enum opened up.

    `--no-enum` drops the `enum` key but keeps the declared type and the tool
    description. The hosted provider appears to constrain tool-call arguments
    to a declared enum, which makes the closed-enum arm unable to echo; the
    open-schema arm is the control that shows what the model does when it can.
    """
    tool = tool_schema(s, 0)
    if not enum_on:
        tool["function"]["parameters"]["properties"][item["field"]].pop("enum", None)
    return tool


PROBE_TEMPLATE = (
    'Set the {field} to "{value}" for record {rid}. '
    'The value must be exactly "{value}"; do not use any other value.'
)


def one(item: dict, s: dict, model: str, key: str, args) -> dict:
    members = s["members"]
    if args.probe:
        # The probe demands a value the enum cannot express. If the provider
        # constrains arguments, the returned argument is still a member.
        item = {**item, "value": args.probe_value}
        user = PROBE_TEMPLATE.format(
            field=item["field"], value=args.probe_value, rid=item["rid"]
        )
    else:
        user = INSTRUCTION_TEMPLATE.format(
            field=item["field"], value=item["value"], rid=item["rid"]
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "tools": [schema_for(item, s, args.enum)],
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "seed": args.seed,
        # Without this the provider spends the whole token budget before the
        # call and returns empty content (finish_reason "length").
        "reasoning": {"enabled": False},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.provider:
        payload["provider"] = {
            "order": [args.provider],
            "allow_fallbacks": False,
        }
    body = call(payload, key, args.timeout)
    msg = body["choices"][0]["message"]
    text = message_text(msg)
    cls = classify(item, text, members)
    usage = body.get("usage") or {}
    return {
        **item,
        **cls,
        "members": members,
        "text": text,
        "provider": body.get("provider"),
        "api_model": body.get("model"),
        "finish_reason": body["choices"][0].get("finish_reason"),
        "usage": {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "cost": usage.get("cost", 0.0),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3.5-9b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--schemas", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--provider", default="",
                    help="pin one OpenRouter provider (no fallbacks)")
    ap.add_argument("--no-enum", dest="enum", action="store_false",
                    help="drop the enum from the tool schema (open-field control)")
    ap.add_argument("--probe", action="store_true",
                    help="demand a made-up non-member value (enum-enforcement probe)")
    ap.add_argument("--probe-value", default="definitely not a member")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--price-in", type=float, default=0.10,
                    help="USD per million prompt tokens")
    ap.add_argument("--price-out", type=float, default=0.15,
                    help="USD per million completion tokens")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        kf = Path(".openrouter_key")
        if kf.exists():
            key = kf.read_text().strip()
    if not key and Path(".env").exists():
        lines = [
            ln.strip()[len("export "):] if ln.strip().startswith("export ") else ln.strip()
            for ln in Path(".env").read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        for line in lines:
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                break
        if not key and len(lines) == 1 and lines[0].startswith("sk-or-"):
            # `.env` holding the bare key, with no assignment.
            key = lines[0]
    if not key:
        print(
            "OPENROUTER_API_KEY not found in environment, .openrouter_key or .env",
            file=sys.stderr,
        )
        return 2

    items = build_items()
    if args.probe:
        items = [i for i in items if i["arm"] == "awkward"]
        args.limit = args.limit or 6
    if args.schemas:
        items = [i for i in items if i["schema_idx"] < args.schemas]
    if args.limit:
        items = items[: args.limit]
    print(f"items={len(items)} model={args.model} workers={args.workers}", flush=True)

    results: list[dict | None] = [None] * len(items)
    errors: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(one, it, SCHEMAS[it["schema_idx"]], args.model, key, args): n
            for n, it in enumerate(items)
        }
        done = 0
        for f in as_completed(futs):
            n = futs[f]
            try:
                results[n] = f.result()
            except Exception as e:
                errors.append(f"{items[n]['id']}: {e}")
            done += 1
            if done % 10 == 0 or done == len(items):
                el = time.time() - t0
                print(f"  {done}/{len(items)}  {el:.0f}s", flush=True)

    ok = [r for r in results if r is not None]
    if errors:
        print(f"{len(errors)} item(s) failed:", file=sys.stderr)
        for e in errors[:10]:
            print("  " + e, file=sys.stderr)

    summary = summarize(ok, 0, "")
    usage = {
        "prompt_tokens": sum(r["usage"]["prompt"] for r in ok),
        "completion_tokens": sum(r["usage"]["completion"] for r in ok),
    }
    cost = (
        usage["prompt_tokens"] / 1e6 * args.price_in
        + usage["completion_tokens"] / 1e6 * args.price_out
    )
    reported_cost = sum(r["usage"].get("cost") or 0.0 for r in ok)
    meta = {
        "api": "openrouter",
        "model": args.model,
        "api_models": dict(Counter(r.get("api_model") for r in ok)),
        "providers": dict(Counter(r.get("provider") for r in ok)),
        "temperature": args.temperature,
        "seed": args.seed,
        "enum_in_schema": bool(args.enum),
        "probe": bool(args.probe),
        "probe_value": args.probe_value if args.probe else None,
        "max_new_tokens": args.max_new_tokens,
        "usage": usage,
        "cost_usd": round(cost, 6),
        "reported_cost_usd": round(reported_cost, 6),
        "wall_seconds": round(time.time() - t0, 1),
        "errors": errors,
        "note": (
            "Hosted provider, not the local 4-bit protocol. Report separately "
            "from local runs."
        ),
    }

    print("\n=== C2 pilot (OpenRouter) ===", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<34} {v:.4f}", flush=True)
        elif k != "by_kind":
            print(f"  {k:<34} {v}", flush=True)
    print(f"  by_kind {summary['by_kind']}", flush=True)
    print(f"  tokens {usage}  cost ${cost:.4f}  wall {meta['wall_seconds']}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "meta": meta, "items": ok}, indent=2))
    print(f"\nwrote {out}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
