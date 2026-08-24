# DiaLLM — Linguistic / Corpus Analysis of Model Outputs

Independent, reward-free linguistic analysis of DiaLLM generation outputs, built to
support the paper's findings (robustness–generation dissociation; reward–quality gap)
with evidence that does **not** reuse the eWAVE feature classifier that defines the
training reward. Runs on a laptop in a couple of minutes (spaCy small model, CPU).

## What it measures

Three measure families (the point is to capture *how* features are used, not just
how many — the eWAVE reward already sums density):

| Measure | Meaning | Reward-independent? |
|---|---|---|
| **density** (features /1k tokens) | how dialectally marked | overlaps reward concept, different instrument |
| **diversity** (distinct feature types) | breadth vs narrow over-use | yes |
| **stacking** (features per sentence) | cramming / caricature | yes |
| **orthography** (British vs American spelling) | objective en-UK/en-AU signal | yes (outside eWAVE morphosyntax) |
| **lexis** (curated per-variety markers) | vocab shift | yes |
| **lexical diversity** (TTR, distinct-2, MTLD) | degeneration under RL | yes |
| **register** (contractions, politeness) | naturalness / pragmatics | yes |

Plus:
- **JS divergence vs the standard pole** (each family's base/instruct outputs) — how far
  each stage shifts the feature distribution from standard English.
- **Human-preference bridge** — links the measures to the human pairwise judgements
  collected on Llama (the six `annotations_*.csv`), so the measures that track human
  preference can be read off Qwen/Gemma too.

## Detectors (rule-based, high-precision)

- **Lexical** — curated en-AU / en-IN / en-UK(Northern) lexicons (`lexicons.py`).
- **Orthographic** — explicit British↔American spelling pairs (no suffix heuristics).
- **Morphosyntactic** (spaCy POS + dependency): progressive-with-stative (`I am understanding`),
  was/were generalisation (`she were`), possessive *me* (`me brother`), pluralised mass
  nouns (`informations`), bare adverbs (`drive slow`), invariant tags (`…, no?`).

> ⚠️ spaCy is trained on standard English and mis-tags some dialectal forms. Every
> morphosyntactic detector is unit-tested (`tests/test_detectors.py`) and was spot-checked
> on real outputs (the progressive detector was tightened to exclude standard catenatives
> like "been wanting to" and predicate adjectives like "be understanding"). Treat per-detector
> rates as a **lower bound** and hand-validate precision before quoting exact numbers.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data

Place inputs under `data/` (git-ignored):
- `data/annotation_responses.jsonl` — one JSON object per output with fields:
  `model_id, stage, family, variant, prompt_id, domain, prompt, response`.
  `stage ∈ {base,cpt,instruct,sft,dpo,grpo,gspo}`; `variant ∈ {base,all,aus,ind,brit}`
  (`all` = implicit/broad thread; `aus/ind/brit` = explicit variety-targeted; `brit`→en-UK).
- `data/annotations/annotations_*.csv` — human pairwise judgements (for the bridge).

## Run

```bash
python run_analysis.py \
  --jsonl data/annotation_responses.jsonl \
  --annot-dir data/annotations \
  --out results
```

## Outputs (`results/`)

| File | Contents |
|---|---|
| `per_output_measures.csv` | every measure for every output |
| `agg_by_condition.csv` | means + bootstrap 95% CIs per (family, stage, variant) |
| `reward_quality_gap.csv` | explicit thread: density/diversity/stacking by method |
| `js_vs_standard.csv` | JS divergence of each stage vs the standard pole |
| `bridge_summary.csv` | human winner−loser measure deltas (per task, pooled) |
| `tables.tex` | LaTeX for the headline linguistic table |
| `fig_*.png` | density-vs-diversity scatter; stacking-by-method bar |

## Tests

```bash
python -m pytest tests/          # or: python -c "import tests.test_detectors as T; ..."
```

## Repo layout

```
diallm_analysis/   lexicons.py  features.py  analyze.py  bridge.py
run_analysis.py    end-to-end orchestrator
tests/             detector precision sanity tests
docs/REWORKED_PLAN.md   laptop-feasible plan (matched to the real schema)
RESULTS_SUMMARY.md      verified findings + how they map to the paper
```

See `RESULTS_SUMMARY.md` for the headline results and `docs/REWORKED_PLAN.md` for the
analysis design and caveats.
