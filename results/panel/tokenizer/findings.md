# Tokenizer / family axis

Base arm: 420 items (210 awkward / 210 clean), closed-enum tool schema,
`temperature=0`, `seed=20261017`, 96 max new tokens (DeepSeek primary run uses
256; see caveat), 4 workers, no `--no-enum`. All rates are shares of the 210
awkward items except Clean error, which is a share of the 210 clean items.
Total spend for this axis: **$0.038**.

## Results

| Model | Family | Tokenizer | Params | Provider | Quant | Enforcement | Substitution | Equivalent | Shifted | Echo | Clean error | Cost | Wall |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.1 8B Instruct | Llama 3 | Llama 3 BPE (128k) | 8B | DeepInfra | FP8 | n/a | — | — | — | — | — | $0.0000 | n/a |
| Gemma 3 12B IT | Gemma 3 | SentencePiece (262k) | 12B | DeepInfra | BF16 | n/a | invalid | invalid | invalid | invalid | invalid | $0.0073 | 215.1 s |
| Mistral Small 3.2 24B Instruct | Mistral Small 3 | Tekken (131k) | 24B dense | DeepInfra | FP8 | no | 75.7% | 80.4% | 66.7% | 5.2% | 0.0% | $0.0058 | 219.8 s |
| DeepSeek V4 Flash 0731 | DeepSeek V4 | DeepSeek BPE | 13B active / 284B total MoE | DeepInfra | FP8 | no | 65.7% | 73.2% | 51.4% | 33.8% | 0.0% | $0.0123 | 110.7 s |
| Qwen3.5-9B (enforcement reference) | Qwen | Qwen BPE | 9B | SiliconFlow | not reported | yes | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | $0.0161 | 170.3 s |

### Exclusions

- **Llama 3.1 8B:** DeepInfra's pinned FP8 endpoint does not advertise tool
  support. With `allow_fallbacks: false`, OpenRouter removed every candidate
  under "Filter by Tool Compatibility"; all six probe items returned HTTP 404.
  The row is missing, not a model result. A different provider or precision
  would add provider/precision confounds to this axis.
- **Gemma 3 12B:** DeepInfra serialized tool arguments as `{}` or truncated them
  at a space/underscore and appended `{}` (`in_progress` → `in{}`,
  `awaiting triage` → `awaiti{}`, `open` → `{}`). The defect reproduces with
  the enum removed and affects clean items too (100% clean error). Its rates are
  not interpretable as mapping behaviour and must not be used.

### Qwen reference

The SiliconFlow Qwen3.5-9B row is an **enforcement ceiling**, not a free-choice
arm: its `--no-enum` control echoed 210/210 awkward items, while the closed-enum
arm substituted 210/210. It must not be compared to the free-choice rows as a
tokenizer contrast.

The correct free-choice Qwen reference is the precision-axis DeepInfra BF16 run:
**31.9% substitution, 13.9% shifted, 68.1% echo, 0/210 clean error**. That row
is directly comparable to Mistral and DeepSeek on the free-choice route.

## Findings

- **No tokenizer/family clustering is supported by this panel.** Only two rows
  are usable, and they order as Mistral (Tekken, 24B dense) 75.7% > DeepSeek
  (DeepSeek BPE, 13B active MoE) 65.7%. That difference is equally a
  size/architecture difference, and it runs opposite to a simple size
  prediction: the larger dense model substitutes more.
- **The Qwen comparison is an enforcement contrast, not a tokenizer contrast.**
  Qwen on SiliconFlow substituted 210/210 in the closed-enum arm and echoed
  210/210 with the enum removed, the signature of provider constrained decoding.
  The free-choice Qwen3.5-9B result is the DeepInfra BF16 arm (31.9%
  substitution, 68.1% echo). Against that reference, Mistral and DeepSeek both
  substitute more and echo less: the route varies by model/training, not by a
  single tokenizer mechanism.
- **Substitution survives shifted near-misses.** Mistral 66.7% and DeepSeek
  51.4% on items whose near-miss means something different from any member,
  versus 80.4% and 73.2% on equivalent paraphrases. The effect is not merely
  normalizer-friendly rephrasing, but it is far from universal.
- **The two working models split mainly on echo, not on tokenizer.** DeepSeek
  copied the non-member through the schema on 33.8% of awkward items; Mistral
  did so on 5.2% and declined 40/210 outright. This is a model/training
  difference in what to do when no member fits (verbatim copy vs nearest-member
  choice vs refusal); no tokenizer mechanism is needed to explain it.
- **Working clean arms are at ceiling.** Mistral and DeepSeek were both 210/210
  exact on clean items, so their substitutions are selective rather than a
  general inability to fill arguments. Gemma's clean arm failed 210/210 for the
  provider-serialization reason above; treating that as a model or tokenizer
  result would be wrong.
- **DeepSeek needed a token-budget caveat.** At 96 tokens, 3/420 items sat at
  the ceiling (one `finish_reason: length` containing prose), so the primary run
  uses 256 tokens. The rerun resolved all three: substitution 64.3% → 65.7%,
  echo 34.3% → 33.8%. The 96-token run is retained as
  `deepseek_v4_flash_96.json`. Reasoning was requested off throughout, and no
  item showed a reasoning field left on.
- **The base arm alone cannot distinguish policy from a mapping deficit.** An
  in-enum substitution is equally consistent with a policy (prefer a
  schema-valid call over asking) and with a mapping deficit (the near-miss is
  represented as, or defaulted to, the nearest member). A prose probe with no
  tool schema should ask the model to (a) state whether the value is a legal
  member, (b) name the closest member and say whether the two are equivalent,
  and (c) say whether it would confirm before writing. If it articulates the
  mismatch yet still substitutes under the schema, that supports a policy
  account; if it cannot articulate it, that supports a mapping deficit. A
  no-enum control on the same provider should accompany it to rule out provider
  enforcement.

## Files

- `summary.json` — compact metrics for every run.
- `probe_llama31_8b.json`, `probe_gemma3_12b.json`,
  `probe_mistral_small_32_24b.json`, `probe_deepseek_v4_flash.json`
- `gemma3_12b.json`, `mistral_small_32_24b.json`,
  `deepseek_v4_flash.json` (primary, 256 tokens),
  `deepseek_v4_flash_96.json` (companion)
