# C2 — Pre-registration

Written 2026-10-02, **before the pilot ran**. Item set frozen at
`items/c2_items.py`; harness frozen at `scripts/c2_pilot.py`. Neither is edited
after the first full run. Any later change is a new pass, recorded as such.

This file exists because of `dismissed.md` rule 7. G1 pre-registered a ratio
threshold and a deletion count, passed, and then died when a control class it had
never named reversed the result. **The control class is named here, before items
were written.**

---

## Claim

When a required enum field cannot be filled with the exact gold member, a
tool-calling model substitutes a **near-miss member** rather than abstaining or
erroring. The emitted call is schema-valid, so the framework executes it, and the
caller receives no signal that the argument differs from the instruction.

## Falsification condition — one sentence, fixed now

> **If the near-miss substitution rate on awkward items is not higher than the
> ordinary argument-error rate on matched clean items, there is no substitution
> effect — only ordinary argument error — and the claim is dead.**

Per `003` and the G1 precedent: a null here is a death, not a discovery. No
re-framing.

---

## Arms and the control class

The control class is **matched clean twins**, and this is the whole design. Each
awkward item is paired with a clean item that shares its schema, its field, its
instruction template and its record id, and differs in exactly one respect: the
supplied value is an exact member rather than a near-miss.

| | value supplied | gold behaviour |
|---|---|---|
| **awkward** | not a member | ask or flag |
| **clean** | the exact member | emit it |

Matching the clean twin on schema, field, template and id means the comparison
isolates **the mutation from ordinary argument error**. Without the twin arm,
a substitution rate of 80% is uninterpretable — it could be the model's base
error rate on this task.

The scoring annotation is the *supplied value*, not the schema, so ground truth
is exact string equality and needs no judge.

## Item set

15 tool schemas, each with exactly one required field and a closed enum of 4–5
members. 420 items: **210 awkward, 210 clean**.

Every awkward value is tagged:

- **equivalent** (138) — a paraphrase a normaliser could map losslessly
  ("underway" for `in_progress`).
- **shifted** (72) — means something different from the member it sits near
  ("in review" for `in_progress`, "cancelled" for `closed`, "same day" for
  `overnight`).

Substitution on **shifted** items changes the action the tool takes. Substitution
on **equivalent** items does not. A rate carried only by `equivalent` is weak, so
the two are reported separately and **the shifted rate is the one that carries
the claim.**

Hard constraints, asserted in `items/c2_items.py`:
no near-miss equals a member; no member appears as a whole word inside a
near-miss (so substitution cannot be explained by string matching rather than
semantic proximity); every member has 3 near-misses.

## Readouts

| # | readout | decides |
|---|---|---|
| 1 | **substitution rate** — P(emitted value is a member \| awkward) | verdict |
| 2 | **clean error rate** — P(emitted value ≠ gold \| clean) | verdict |
| 3 | **schema validity**, reported per arm | the "no signal" half |
| 4 | abstain rate and no-call rate, awkward arm | the gold behaviour |
| 5 | **rotation pass** (below) | prototypicality vs position |

**Decision rule, fixed now.** The claim survives iff `readout_1 > readout_2` by a
margin that is not explainable by sampling noise, **and** the shifted-item
substitution rate is itself above the clean error rate. Both conditions. A pass on
the pooled rate with a dead shifted rate is a fail.

Also reported, not pre-registered as decisive: **near-target hit rate** — of the
awkward items that produced a member, what fraction produced the member the value
was written against. High hit rate means the substitution tracks the intended
meaning; low hit rate means the model is picking a member for unrelated reasons.

## Readout 5 — enum-order rotation

The one measurement `candidates.md` records as unoccupied. Same items, same
prompts, with the enum's member order rotated by k = 2. Two statistics against
chance 1/k:

- `same_member_rate` — the emitted member is unchanged. **Prototypicality**: the
  substitution is semantic and survives re-serialising the schema.
- `same_index_rate` — the emitted *index* is unchanged. **Position artefact**: the
  model fixed on a slot in the list, not on a category.

This matters because a framework re-serialises its tool list on every call, and
because Constraint Tax (2605.26128) reports the semantic error without controlling
for where the enum members sat in the prompt.

## Models

Local `Qwen/Qwen2.5-7B-Instruct`, 4-bit nf4, greedy decode, `transformers` chat
template with the native tool-call format. Deterministic, so a re-run is a
replication, not a resample.

## What would kill it, ranked by likelihood

1. **Clean error rate ≈ substitution rate.** Then there is no effect, only a
   model that is bad at argument copying, and the claim dies by its own
   pre-registered sentence.
2. **Shifted rate ≈ clean error rate while equivalent rate is high.** The model
   normalises legitimate paraphrases and fails on genuine shifts. That is
   defensible behaviour, not a silent-corruption finding, and it is not a paper.
3. **`same_index_rate` high, `same_member_rate` near chance.** The substitution is
   a position artefact. Kills the framing, and would be a result in the other
   direction.
4. **Abstain rate high.** The model asks when the value is not a member. That is
   the correct behaviour and the premise of the claim is false on this model.

## Amendments made before the full run

Recorded so the record shows what was fixed when. Neither touches the
hypothesis, the control class, the comparison band or the decision rule.

1. **`echoed` added as an outcome class, separate from `invalid`.** The 8-item
   smoke pass showed the two failure modes are opposite: substitution is
   schema-**valid** (silent), echoing the value through is schema-**invalid**
   (loud, the framework raises). Pooling them would have hidden the distinction
   the "no signal" half of the claim rests on.
2. **Per-arm schema validity reported** rather than pooled, for the same reason.

## Not in this pilot

No API models (a later replication arm), no multi-turn, no optional enum fields,
no free-text bridging. The typed-vs-free-text arm is dropped — occupied twice
(PhantomFill 2607.20492, Constraint Tax 2605.26128) — and the range-snapping
headline is dropped to FinMathBench (AAAI 2026). See `candidates.md`.
