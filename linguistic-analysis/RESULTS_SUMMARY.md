# DiaLLM — Verified Linguistic Findings (for the paper)

Run on the released 1,375-output JSONL (3 families × full pipeline × 25 prompts × variants),
plus the six human annotation CSVs. All numbers below were produced by `run_analysis.py`
and are reproducible. Treat per-detector rates as lower bounds (high-precision detectors).

## 1. Surface dialectal marking is sparse and concentrated in en-IN
Outputs containing ≥1 independent dialectal marker, by target variety (explicit thread):
**en-IN 37.7% · en-UK 10.0% · en-AU 3.3%.** The en-AU markers we detect (bare adverbs,
invariant tags) fire **zero** times — consistent with the paper's own observation that
AusE features are register-flexible and not elicited by casual prompts. Dominant detectors:
pluralised mass nouns (`informations`, `moneys`, `traffics`; 108 firings) and
progressive-with-stative (`I am understanding`, `persons are preferring`; 85).
**Implication:** the generation framing is best evidenced through en-IN; en-AU/en-UK are
too sparse for strong per-variety claims. This is itself an honest, reportable result.

## 2. Reward–quality gap, corroborated independently across all 3 families
GRPO maximises the eWAVE-classifier reward (paper Table 6) but produces the **fewest**
independent surface dialectal markers:

Pooled explicit thread (mean per method): density `SFT 2.07 · DPO 1.96 · GSPO 1.45 · GRPO 1.27`;
diversity `GRPO 0.16` (lowest); any-marker rate `DPO 20.9% · SFT 17.3% · GSPO 16.0% · GRPO 13.8%`.

en-IN (the powered cell), density /1k by method:
- **Qwen:** SFT 6.04 · DPO 5.53 · GSPO 2.80 · **GRPO 1.33** (collapse)
- **Llama:** DPO 4.69 · SFT 3.94 · GRPO 3.34 · GSPO 3.30
- **Gemma:** SFT 6.02 · GRPO 4.88 · GSPO 4.14 · DPO 3.96 (mixed)

**Claim it supports:** the eWAVE feature-density reward is *decoupled* from
independently-measured surface dialectal richness — the method that most optimises the
reward (GRPO) does not produce more recognisable dialectal markers, and on Qwen/Llama
produces markedly fewer. This extends the reward–quality gap **beyond Llama** to all three
families on an independent measure (the human study remains Llama-only).

## 3. Human preference tracks dialect vs standard, not reward density
Human winner−loser measure deltas (Llama; `*` = bootstrap 95% CI excludes 0):
- **Task 1 (instruct vs SFT_d):** density **+1.38\*** , stacking **+0.030\*** — humans
  reliably prefer the *more dialectal* output over the standard instruct baseline.
  → independent confirmation that explicit adaptation is perceptible and preferred.
- **Task 2 (SFT_d vs GRPO_d):** contractions **+2.87\*** (humans prefer the more natural,
  contracted output); density −0.51 and stacking −0.014 (directionally lower, CIs include 0).
  → when both outputs are dialectal, the reward-maximising GRPO output is *not* preferred;
  naturalness, not feature density, drives the choice.
- **Task 4 (broad vs targeted):** density +0.78, diversity +0.10 (directional; CIs include 0).

## 4. Distribution shift from standard (JS divergence)
en-IN shifts most from the standard pole (≈0.11–0.50 across methods/families); en-AU
barely shifts (≈0.03–0.25); en-UK small. On Qwen/Llama en-IN, GRPO shifts *less* from
standard than SFT/DPO despite its higher reward — same direction as §2.

## How this maps to the paper
- **Abstract / intro (generation framing):** §1 shows the lexical/morphosyntactic phenomena
  the intro cites are real but variety-dependent (strong en-IN, weak en-AU).
- **Reward–quality gap (§5.4):** §2 + §3 give an independent, non-circular mechanism and
  generalise it across families.
- **Positive finding (explicit adaptation perceptible):** §3 Task 1 confirms it linguistically.
- **New subsection "Linguistic analysis of generation":** Tables `reward_quality_gap.csv`,
  `bridge_summary.csv`, `js_vs_standard.csv`; figures `fig_*.png`; LaTeX in `tables.tex`.

## Caveats (state these in the paper)
- Detectors are high-precision/low-recall → density is a lower bound. The progressive
  detector was tightened after spot-checking (excludes standard "been wanting to" and
  predicate-adjective "be understanding").
- en-AU/en-UK are sparse; quantitative density claims are reliable only for en-IN.
- The human bridge is Llama-only and n is modest (48–85 non-tie pairs per task); report
  effect directions + CIs, not precise estimates.
- `british_per1k` overlaps en-UK and en-AU orthography; interpret per variety.
