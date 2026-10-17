# Valid but Wrong — paper, code and data

Release accompanying the paper *"Valid but Wrong: Silent Enum Substitution in
Tool Calls, and Why Schema Validation Cannot See It"* (Lakshit Farswan,
Department of Computer Science, The University of Texas at Austin, 2026).

**Paper:** `paper/main.pdf` (source `paper/main.md`), 26 pages.

## What is here

| Path | Contents |
|---|---|
| `paper/main.pdf` | The paper |
| `paper/main.md` | Source, with figures referenced from `figures/` |
| `paper/ARXIV-SUBMISSION.md` | arXiv categories, comments field and the plain-text metadata abstract |
| `code/c2_items.py` | The 15 tool schemas and the 420 frozen items |
| `code/c2_pilot.py` | Model runner: renders the native tool template, decodes greedily, classifies outcomes |
| `code/20_semantic_comparator.py` | Comparator attempt 1 (fine-tuned MS MARCO cross-encoder) |
| `code/22_bge_comparator.py` | Comparator attempt 2 (BGE embeddings, cosine and logistic head) |
| `code/23_headline_stats.py` | Wilson intervals, exact McNemar, Fisher tests |
| `code/21_compare_models.py` | Cross-model table |
| `code/24_reparse.py` | Re-scores stored generations from their saved text (see note below) |
| `code/25_literal_check.py` | Deterministic literal check: flags a call whose value is not verbatim in the instruction |
| `code/40_comprehension.py` | Prose comprehension control: can a model map the value to a member at all? |
| `code/42_retry.py` | Validation-retry control: what an echoed call becomes on one retry |
| `code/43_extra_stats.py` | Wilson intervals and paired tests for the three added controls |
| `code/44_comprehension_parse.py` | Post-hoc cue-based re-parse of the prose-probe completions |
| `code/45_cluster_stats.py` | Schema-level bootstrap intervals for the headline rates |
| `code/47_llm_judge.py` | Zero-shot LLM judge on the emitted substitutions |
| `code/48_openrouter.py` | Hosted tool-call arm with a closed enum and an enum-removed control |
| `code/30_figures.py` | All four figures |
| `PREREGISTRATION-c2.md` | Readouts, gates and decision rules, frozen before the primary runs |
| `PREREGISTRATION-c2-addendum.md` | The three post-primary controls, declared descriptive rather than confirmatory |
| `results/` | Every JSON artifact the paper's numbers come from, including per-item generations |
| `figures/` | Figures as PDF and PNG |
| `LICENSE` / `LICENSE-DATA` | MIT for code; CC BY 4.0 for data, figures and paper text |

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# base models, 420 items each
.venv/bin/python code/c2_pilot.py --model Qwen/Qwen2.5-7B-Instruct \
    --out results/c2_raw.json
.venv/bin/python code/c2_pilot.py --model Qwen/Qwen2.5-3B-Instruct \
    --out results/c2_qwen3b.json
.venv/bin/python code/c2_pilot.py --model mistralai/Mistral-7B-Instruct-v0.3 \
    --out results/c2_mistral.json

# rotation control (enum order shifted by two)
.venv/bin/python code/c2_pilot.py --model Qwen/Qwen2.5-7B-Instruct \
    --rotate 2 --out results/c2_rot2.json

# instruction arms (system-prompt suffix)
.venv/bin/python code/c2_pilot.py --model Qwen/Qwen2.5-7B-Instruct \
    --system-suffix "The tool call must validate against the schema. Never emit \
an argument value that is not one of the allowed enum members." \
    --out results/c2_7b_pressure_validity.json
.venv/bin/python code/c2_pilot.py --model Qwen/Qwen2.5-7B-Instruct \
    --system-suffix "The tool call must validate against the schema. Never emit \
an argument value that is not one of the allowed enum members. If the value you \
were given is not allowed, choose the closest allowed member." \
    --out results/c2_7b_pressure.json

# abstention arm (system prompt explicitly permits asking)
.venv/bin/python code/c2_pilot.py --model Qwen/Qwen2.5-7B-Instruct \
    --system "You are a tool-calling assistant. Call the tool with the arguments \
you are given. Emit a single tool call. If the requested value cannot be \
expressed with the allowed values, do not call the tool; instead ask the user \
which allowed value they want." --out results/c2_abstain_7b.json

# comprehension control (prose, no tool schema; 64 tokens so long answers are not cut off)
.venv/bin/python code/40_comprehension.py --model Qwen/Qwen2.5-7B-Instruct \
    --max-new-tokens 64 --base results/c2_raw.json \
    --out results/c2_comprehension_7b.json
.venv/bin/python code/40_comprehension.py --model Qwen/Qwen2.5-7B-Instruct \
    --arm clean --max-new-tokens 64 --out results/c2_comprehension_clean_7b.json

# validation retry (replays every echoed call once)
.venv/bin/python code/42_retry.py --model Qwen/Qwen2.5-7B-Instruct \
    --base results/c2_raw.json --out results/c2_retry_7b.json

# same retry, error message withholding the allowed-value list
.venv/bin/python code/42_retry.py --model Qwen/Qwen2.5-7B-Instruct --no-list \
    --base results/c2_raw.json --out results/c2_retry_nolist_7b.json

# zero-shot LLM judge on the emitted substitutions
.venv/bin/python code/47_llm_judge.py --model Qwen/Qwen3.5-9B \
    --base results/c2_raw.json --out results/c2_llm_judge_qwen35_9b.json

# hosted arm (needs OPENROUTER_API_KEY); closed enum, then enum removed
.venv/bin/python code/48_openrouter.py --model qwen/qwen3.5-9b \
    --provider SiliconFlow --out results/c2_qwen35_9b_api.json
.venv/bin/python code/48_openrouter.py --model qwen/qwen3.5-9b \
    --provider SiliconFlow --no-enum --out results/c2_qwen35_9b_api_noenum.json

# enum-enforcement probe: all 210 awkward items, explicit demand for a made-up non-member
.venv/bin/python code/48_openrouter.py --model qwen/qwen3.5-9b \
    --provider SiliconFlow --probe --limit 210 --out results/c2_enum_probe_closed.json
.venv/bin/python code/48_openrouter.py --model qwen/qwen3.5-9b \
    --provider SiliconFlow --probe --no-enum --limit 210 --out results/c2_enum_probe_open.json

# literal check on the stored calls (no model run)
.venv/bin/python code/25_literal_check.py

# comparators
.venv/bin/python code/20_semantic_comparator.py
.venv/bin/python code/22_bge_comparator.py

# statistics and figures
.venv/bin/python code/23_headline_stats.py
.venv/bin/python code/43_extra_stats.py
.venv/bin/python code/44_comprehension_parse.py
.venv/bin/python code/21_compare_models.py
.venv/bin/python code/30_figures.py
```

Model weights are not included. Nothing needs to be
trained before the runner can be used; the comparators train inside their own
scripts.

The pre-registration was written before the release layout was fixed. It refers
to `items/c2_items.py` and `scripts/c2_pilot.py`; in this repository those files
are `code/c2_items.py` and `code/c2_pilot.py`. The code is otherwise unchanged.
`code/45_cluster_stats.py` is the schema-clustered bootstrap added in revision;
it reads the same released JSONs and writes `results/c2_cluster_stats.json`.
`results/c2_downstream.json` is retained from an earlier reviewer-based audit;
it is not the literal check. The paper's §5.8 numbers are reproduced by
`code/25_literal_check.py` from `results/c2_raw.json`.

## Building the PDF

The paper source is `paper/main.md` (pandoc Markdown). It is rendered to a real
LaTeX document and compiled by tectonic — there is no hand-written `.tex`:

```bash
cd paper
pandoc main.md -o main.pdf --pdf-engine=tectonic \
    -V geometry:margin=1in -V fontsize=11pt
```

`tectonic` is used because it is a self-contained LaTeX engine that fetches its
own packages; `pdflatex` and `xelatex` are not installed on this machine. To
inspect or hand in the LaTeX, dump the intermediate file:

```bash
pandoc main.md -o main.tex -s --resource-path=.
```

References are written directly in the Markdown in APA style (alphabetical,
`Author, A. B., & Author, C. (Year). Title. *Venue*.`), not generated by a
citation manager.

## Submitting to arXiv

arXiv does not accept Markdown as source. `paper/build_arxiv.sh` writes
`paper/arxiv/` (`main.tex` plus `figures/`) and the upload tarball
`paper/arxiv.tar.gz`; both are generated and git-ignored. The tarball was
compiled with tectonic at 26 pages before this README was written.

```bash
cd paper && ./build_arxiv.sh
```

Categories, the Comments field, the author-block conventions used by UT Austin
papers, and the plain-text abstract for arXiv's metadata field (3,182
characters in the paper, 1,889 in the field's 1,920-character limit) are in
`paper/ARXIV-SUBMISSION.md`.

## Note on `24_reparse.py`

The first parser read JSON blocks from the end of each model output. Mistral's
native template emits `[TOOL_CALLS] [{...}]` and sometimes a trailing redundant
object, so correct calls were scored as missing-field errors. The fix reads
candidates in order and prefers the one carrying the field under test; it was
applied uniformly to every stored run and changed **zero** Qwen items and 74
Mistral items, after which Mistral's clean-item error rate is 0.000. Because
every generation is stored verbatim in the result JSONs, the correction is
reproducible without a GPU:

```bash
.venv/bin/python code/24_reparse.py          # dry run, prints the diff
```

## Traceability

Every number in the paper is written by one of the scripts above into a JSON
file in `results/`. None was transcribed by hand.

## License

The code is released under the MIT License (`LICENSE`). The item set, result
JSONs, figures and manuscript text are released under the Creative Commons
Attribution 4.0 International License (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. The paper is a 2026 preprint; a venue and DOI will be added
if it is published.
