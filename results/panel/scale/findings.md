# Scale axis

Base arm: 420 items (210 awkward / 210 clean), closed-enum tool schema, `temperature=0`, `seed=20261017`, 96 max new tokens, 4 workers, no `--no-enum`. The Qwen3.5-9B reference was not rerun; it is the precision-axis run at `results/panel/precision/deepinfra.json` (DeepInfra, BF16). All rates below are shares of the 210 awkward items except Clean error, which is a share of the 210 clean items.

| Model | Family | Tokenizer | Params | Provider | Quant | Enforcement | Substitution | Equivalent | Shifted | Echo | Clean error | Cost | Wall |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B (precision-axis ref) | Qwen3.5 | Qwen3 | 9B | DeepInfra | BF16 | no | 31.9% | 41.3% | 13.9% | 68.1% | 0.0% | $0.0160 | 554 s |
| Qwen3.5-27B | Qwen3.5 | Qwen3 | 27B | Novita | BF16 | no | 61.9% | 75.4% | 36.1% | 38.1% | 0.0% | $0.0710 | 229 s |
| Qwen3-32B | Qwen3 | Qwen3 | 32B | DeepInfra | FP8 | no | 44.3% | 55.8% | 22.2% | 15.7% | 0.0% | $0.0103 | 193 s |
| Qwen2.5-72B-Instruct | Qwen2.5 | Qwen | 72B | DeepInfra | FP8 | no | 78.1% | 84.8% | 65.3% | 6.7% | 0.0% | $0.0610 | 150 s |
| Llama-3.3-70B-Instruct | Llama3.3 | Llama3 | 70B | DeepInfra | FP8 | no | 65.7% | 73.9% | 50.0% | 25.2% | 0.0% | $0.0139 | 352 s |

All five runs completed 420/420 with zero request errors. Total spend for the four new full arms plus probes was $0.1603; including the 9B precision-axis run, $0.1763.

## Interpretation

- **The route is not monotonic in scale.** Substitution across 9B / 27B / 32B / 72B is 31.9% -> 61.9% -> 44.3% -> 78.1%. The 32B sits below the nominally smaller 27B, so the ordering does not support a monotonic "more parameters, more substitution" reading. Echo falls in the same list (68.1% -> 38.1% -> 15.7% -> 6.7%), but that decline is descriptive of this mixed panel, not a demonstrated scale effect.
- **Only the 9B -> 27B step is same-generation.** Both are Qwen3.5 at BF16, and the step moves toward substitution (31.9% -> 61.9%) and away from echo (68.1% -> 38.1%). It is consistent with a scale effect but is confounded by provider (DeepInfra vs Novita), so it cannot establish one.
- **The nominal ladder mixes generations.** Qwen3.5-27B, Qwen3-32B and Qwen2.5-72B are three different generations, and the 27B also uses a different provider. The 27B > 32B break in substitution, equivalent and shifted rates is therefore better read as a generation/provider difference than as a size effect; this design cannot attribute it to parameter count.
- **The cross-family control argues for model policy.** Llama-3.3-70B and Qwen2.5-72B are matched on approximate size, provider, quantization and protocol. Their routes differ sharply: 65.7% substitution / 25.2% echo for Llama versus 78.1% / 6.7% for Qwen. A family change at matched scale moves the route, which is consistent with the existing claim that the route is a model policy rather than a scale-determined outcome.
- **Clean-task competence does not predict the route.** Every model was exact on 210/210 clean items (0.0% clean error), while substitution ranged from 31.9% to 78.1% and echo from 6.7% to 68.1%. Perfect clean behaviour coexists with very different awkward-arm behaviour.
- **Meaning-changing substitutions persist at large scale.** Shifted rates are 13.9%, 36.1%, 22.2%, 65.3% and 50.0%; equivalent rates exceed shifted rates in every model. The silent, meaning-changing share does not disappear at 70B/72B, though it is not monotonic either (the 32B is again below the 27B).
- **These are free-choice arms.** No provider enforced the declared enum. The 9B (precision-axis), 27B and 72B probes echoed a non-member via tool calls; the 32B and Llama probes refused, but their full arms returned 33/210 and 53/210 non-member tool-call arguments, which a strict constraint would have prevented. Substitution and echo here are therefore model choices, not constrained decoding. This differs from the earlier SiliconFlow Qwen3.5-9B arm (enforcement yes, 210/210 substituted), which must not be pooled with this axis.
- **Scope.** The data support: larger models still take the substitution route, route mix varies across family/generation at matched provider and precision, and clean accuracy does not predict it. They do not support: a monotonic scaling law, a causal effect of parameter count, or generalisation beyond these providers, BF16/FP8 endpoints, one greedy sample per item, and 420 items. The 32B and Llama enforcement probes were inconclusive on their own; full-run tool-call echoes were used to resolve them.
