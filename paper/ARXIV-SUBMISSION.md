# arXiv submission details

Everything the arXiv form asks for, in the form it expects it.

| Field | Value |
|---|---|
| Title | Valid but Wrong: Silent Enum Substitution in Tool Calls, and Why Schema Validation Cannot See It |
| Primary category | cs.CL (Computation and Language) |
| Cross-list | cs.SE (Software Engineering) |
| Authors | Lakshit Farswan |
| Affiliation (in the paper, not the metadata) | Department of Computer Science, The University of Texas at Austin |
| Comments | Preprint. Code, frozen items and results: https://github.com/LakshitF/valid-but-wrong |
| License | arXiv non-exclusive distribution licence (default); the work itself is MIT / CC BY 4.0 |
| Files | `arxiv.tar.gz` from `build_arxiv.sh` (LaTeX source + figures) |

## Abstract for the metadata field

arXiv stores the abstract separately from the PDF: plain text, no LaTeX, no
Unicode, and **at most 1,920 characters**. The paper's abstract is now short
enough (1,386 characters) to paste unchanged.

```
Tool-calling pipelines treat JSON-schema validation as a gate: if the call validates, it executes. We test what that gate misses when an instruction supplies a value that a closed enum cannot express. On 420 frozen items across 15 tools (210 awkward items, 210 matched clean twins), Qwen2.5-7B-Instruct silently substitutes a schema-valid near-miss member on 54.8% of awkward items and makes no errors on the clean ones. On the 72 items whose nearest member changes the meaning, it substitutes on 34.7%. Schema validation detects none of these calls; a literal comparison against the instruction detects 98.3% but rejects 97.8% of correct paraphrases. In our models the behaviour is not explained by scale or clean-task accuracy: a same-size Mistral-7B and a Qwen2.5-3B, equally perfect on clean items, instead echo the invalid value, but one validation retry turns most of those echoes (72% to 95%) into silent substitutions too. Permitting a clarification cuts the 7B's silent substitutions to 34.8%, and a hosted API that enforces the enum leaves substitution as the only route (Qwen3.5-9B: 210/210). Two trained comparators and a zero-shot judge fail to supply the missing semantic check. The study is small (three local models, one hosted model, a designed item set, one annotator); we release the items, prompts and every generation at https://github.com/LakshitF/valid-but-wrong.
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
