# arXiv submission details

Everything the arXiv form asks for, in the form it expects it.

| Field | Value |
|---|---|
| Title | Valid but Wrong: Silent Enum Substitution in Tool Calls, and Why Schema Validation Cannot See It |
| Primary category | cs.CL (Computation and Language) |
| Cross-list | cs.SE (Software Engineering) |
| Authors | Lakshit Farswan |
| Affiliation (in the paper, not the metadata) | Department of Computer Science, The University of Texas at Austin |
| Comments | Preprint. Code, frozen items and results will be released with the paper. |
| License | arXiv non-exclusive distribution licence (default); the work itself is MIT / CC BY 4.0 |
| Files | `arxiv.tar.gz` from `build_arxiv.sh` (LaTeX source + figures) |

## Abstract for the metadata field

arXiv stores the abstract separately from the PDF: plain text, no LaTeX, no
Unicode, and **at most 1,920 characters**. The paper's abstract is 3,162
characters, so the field gets the version below, which is the same abstract
with the comparator sentences, the hosted clause and clause-level detail
compressed.

```
Tool-calling pipelines use JSON-schema validation as their safety gate: if the call validates, it executes. The gate does not measure what it is assumed to measure. We construct 420 frozen items across 15 tools in which an ordinary instruction supplies a value a closed enum cannot express, paired with matched clean items. Qwen2.5-7B-Instruct silently substitutes a schema-valid near-miss member on 54.8% of awkward items and makes zero errors on the clean ones; where the nearest member changes the meaning, it substitutes on 34.7%, so 25 meaning-changing calls are among the 115 substitutions. Every one validates and would execute; schema validation detects none. A literal comparison detects 98.3% but false-alarms on 97.8% of correct paraphrases. The anchor's behaviour is policy, not a capability threshold: asked in prose with a forced choice, it names an allowed member for all 210 awkward items, including every one it echoed. Two other models, equally perfect on clean items, name one for 76.2% and 65.7% and echo the invalid value 86.2% and 78.1%, but their loudness mixes policy with a mapping deficit. Merely permitting a clarification cuts the anchor's silent substitutions from 54.8% to 34.8%, while the loud models ignore the permission. The contrast is not a stable safety order: one validation retry converts 78.7%, 72.4% and 94.5% of their echoed calls into schema-valid substitutions, including 69.2%, 69.1% and 94.2% of the meaning-changing ones. Behind a hosted provider that enforces the enum, Qwen3.5-9B substitutes on all 210 awkward items and all 72 shifted ones and never echoes; with the enum removed it echoes 210/210, so enforcement does not close the gap. In our three models, neither scale nor clean-task accuracy explains the route, and two trained semantic comparators fail to supply the missing check. We will release the items, prompts and generations.
```

Note: arXiv does not accept TeX in this field and will reject Unicode minus
signs. Any exponent or p-value restored to this text must stay ASCII.

## Rules that shaped the paper

- No degree suffix in the metadata or the author block ("MSc", "Master's
  student" are not allowed in the arXiv metadata; UT Austin papers do not put
  student status in the author block either).
- Affiliation belongs in the PDF author block, not in arXiv's metadata form.
- Cross-listing to cs.SE is one category, within arXiv's "one or two
  cross-lists" guidance; bad cross-lists can be removed by moderators.
- The Comments field is frozen at announcement and cannot be edited later. Do
  not put a "to appear" venue there until it is true.
- Review/survey and position papers in cs now need documented peer review; this
  is an empirical study, so that rule does not apply.
