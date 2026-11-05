# Precision / provider axis

Base arm: 420 items (210 awkward / 210 clean), closed-enum tool schema,
`temperature=0`, `seed=20261017`, 96 max new tokens, 4 workers, no
`--no-enum`. Model: `qwen/qwen3.5-9b`. Total spend including probes: **$0.0561**.

## Results

| Endpoint | Provider | Quant | Probe enforcement | Substitution | Equivalent | Shifted | Echo | Clean error | Cost | Wall |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| deepinfra | DeepInfra | BF16 | no — 6/6 probe value | 31.9% (67/210) | 41.3% (57/138) | 13.9% (10/72) | 68.1% (143/210) | 0.0% (0/210) | $0.0160 | 554 s |
| parasail | Parasail | BF16 | unavailable — 6/6 HTTP 404 | — | — | — | — | — | $0.0000 | — |
| siliconflow | SiliconFlow | FP8 | yes — 6/6 emitted a member | 98.6% (207/210) | 97.8% (135/138) | 100% (72/72) | 0.0% (0/210) | 0.5% (1/207) | $0.0159 | 907 s |
| venice | Venice | FP8 | no — 6/6 probe value | 31.9% (67/210) | 41.3% (57/138) | 13.9% (10/72) | 68.1% (143/210) | 0.0% (0/210) | $0.0160 | 125 s |
| darkbloom | Darkbloom | FP4 | unclear — 6/6 refusal, no call | 90.5% (38/42)* | 94.6% (35/37)* | 60.0% (3/5)* | 0.0% | 0.0% (0/210) | $0.0073 | 7,865 s |

\* Darkbloom partial: 168/420 items failed with upstream HTTP 429, mostly
awkward. All 210 clean items completed but only 42/210 awkward items did, so
its awkward metrics are a selected subset and are not comparable.

## Interpretation

- **No BF16/FP8 difference in the one clean pair.** DeepInfra BF16 and Venice
  FP8, both non-enforcing, are identical down to the item counts: 67
  substitutions, 143 echoes, 10/72 shifted, 0 clean errors. This is a null
  result on these two serving stacks, not a general equivalence; provider,
  sampler and serialization still differ.
- **Enforcement confounds any naive FP8 comparison.** SiliconFlow constrains
  the enum: its probe emitted a member on 6/6 items, and the closed-enum arm
  substituted on 207/210 awkward items (three no-calls) with zero echoes. The
  three malformed clean responses are excluded from the clean denominator. Its
  100% shifted rate is constrained decoding, not a free-choice tendency.
- **FP4 cannot support a trend.** Darkbloom's awkward arm is 20% complete and
  plausibly biased toward shorter generations; its clean arm is complete.
- **Probe behaviour differs by provider, not by quantization.** DeepInfra and
  Venice echo the fake value, SiliconFlow emits members, Darkbloom refuses with
  no call, Parasail 404s. Enforcement must be established per endpoint.
- **In the unconstrained arms, silent substitution is real but not dominant:**
  31.9% silent schema-valid substitution versus 68.1% loud echo; shifted
  substitution is 13.9%, equivalent substitution 41.3%.
- **No stability claim across BF16/FP8/FP4 is supportable.** The only clean
  precision pair is identical, one FP8 endpoint enforces, FP4 is partial, and
  the second BF16 endpoint is unavailable.

## Exclusions

- **Parasail BF16:** persistent HTTP 404 for this provider/model pair. The probe
  was retried once with the same result; the full arm was skipped.
- **SiliconFlow:** three malformed responses (missing `choices`) were excluded;
  metrics use 417 completed items.
- **Darkbloom:** partial awkward arm from upstream HTTP 429; not used for a
  precision comparison.

## Files

- `summary.json` — compact metrics for every run.
- `probe_deepinfra.json`, `probe_venice.json`, `probe_siliconflow.json`,
  `probe_darkbloom.json`, `probe_parasail.json`
- `deepinfra.json`, `venice.json`, `siliconflow.json`, `darkbloom.json`
  (partial; no Parasail full arm)
