# C2 — Pre-registration addendum: follow-up controls

Written 2026-10-17, after the primary runs and before the follow-up analyses
were written up. The primary pre-registration (`PREREGISTRATION-c2.md`), the
frozen item set and the primary harness are unchanged. This addendum records
three controls added after the primary result to test interpretations of it.

They are **descriptive follow-ups, not confirmatory tests**. They do not
replace the primary readouts and cannot rescue or kill the primary claim by
themselves. All three are reported in full, with intervals, whatever they show.

## A1. Prose comprehension control

Question: can a model perform the awkward-value-to-member mapping at all, when
no tool schema is present? The clean-arm result does not answer this, because
copying a valid literal requires no mapping.

Administration: every awkward item, in a fresh conversation with no tools. The
prompt lists the members, quotes the supplied value, and asks the model to name
the single allowed value that best matches the user's meaning. The budget is 64
new tokens, enough for a model that explains before answering to reach its
answer; an earlier 16-token pass was discarded because it cut off 54 of 210
Mistral responses and 11 of 210 3B responses mid-sentence. The same probe is also run
on clean items as a prompt-compliance baseline.

Readouts, per model:

1. fraction of items on which any allowed member is named;
2. fraction on which the member the item was written against (`paired_member`)
   is named, with the chance rate being the mean reciprocal enum size;
3. the same two rates restricted to the items the model echoed in the primary
   tool run, which is the subset that decides whether an echo is attributable
   to a missing mapping;
4. clean-item exact-match rate.

Interpretation rule fixed here: a model that names a member on an item it
echoed is not failing to produce a member on that item. This does not show the
mapping is *correct*; that is what readout 2 measures.

Post-hoc robustness check (added after seeing the completions): the primary
parser credits only a response whose last line isolates one member, which
penalises verbose answers. `code/44_comprehension_parse.py` re-reads the stored
completions with a cue-based rule and reports both rates; the primary readout
above is unchanged and is what the paper reports, with the re-parse given as a
sensitivity check.

## A2. Abstention arm

Question: is the anchor's silent route selected by the base system prompt,
which tells the model to call the tool and emit a single call?

Manipulation: replace the system prompt with one that explicitly permits
asking — *"If the requested value cannot be expressed with the allowed values,
do not call the tool; instead ask the user which allowed value they want."*
Items, decoding and classification are unchanged.

Readouts: substitution rate (all and shifted), echo rate, no-call/abstention
rate, clean error rate, and a paired McNemar against the base run on the same
items.

## A3. Validation retry

Question: does one realistic validation retry convert a loud failure into a
silent one?

Administration: replay every awkward item the model echoed in the primary run.
The retry turn states that the previous call was rejected because the value is
not one of the allowed values, lists those values (as a validator's error
commonly does), and asks the model to respond to the original instruction. It does not say to choose a member, so a second echo or a
clarification remain available.

Readouts: fraction of retried calls that become schema-valid substitutions
(overall and by near-miss kind), second-echo rate, abstention/no-call rate,
and the outcome distribution.

## Reporting rule

All readouts above are reported whether or not they favour the paper's
interpretation. No threshold gate is attached to any of them. The primary
claim continues to stand or fall on the readouts and falsification condition in
`PREREGISTRATION-c2.md`.
