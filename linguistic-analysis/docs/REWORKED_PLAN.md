# Reworked Analysis Plan (laptop-feasible, matched to the real data)

Supersedes the design doc. Pinned to the actual JSONL schema and to what runs quickly on a
laptop (spaCy `en_core_web_sm`, CPU, ~2 min for 1,375 outputs).

## Real schema (confirmed)
`annotation_responses.jsonl`, 1,375 rows. Fields: `model_id, stage, family, variant,
prompt_id, domain, prompt, response`.
- `family ∈ {llama, qwen, gemma}` (llama has both `base` and `instruct`).
- `stage ∈ {base, cpt, instruct, sft, dpo, grpo, gspo}`.
- `variant ∈ {base, all, aus, ind, brit}`: `all` = implicit/broad thread; `aus/ind/brit` =
  explicit variety-targeted (`brit`→en-UK Northern). 25 prompts × 5 domains.
- Outputs carry chat-template/tokenizer artifacts (`xford`, role tags) → cleaned before parsing;
  `base`/`cpt` are partly degenerate.

## What we run (all from the JSONL alone — no external corpus)
1. **A. Independent feature inventory** — lexical (curated lexicons), orthographic
   (British↔American pairs), morphosyntactic (spaCy rule detectors). High precision; validated.
2. **B. Density / diversity / stacking** — the core trio; the reward-independent diagnostic.
3. **C1. JS divergence vs the standard pole** — each family's `base`/`instruct` outputs are the
   matched standard reference (clean, same prompts).
4. **D/E. Lexical diversity + register** — TTR, distinct-2, MTLD; contractions, politeness.
5. **F. Human-preference bridge** — join the six annotation CSVs to Llama outputs via
   `comparison` + `prompt_id`; winner−loser measure deltas with bootstrap CIs.

## Deferred
- **C2 (authenticity vs naturalistic ICE):** prompts are handmade (not ICE), so there is no
  register-matched dialect reference. C2 needs ICE conversational text per variety; add later
  as a strengthening result. Do **not** use Multi-VALUE-perturbed text as the reference
  (circular with training).

## Stats & rigour
- All counts normalised per 1,000 tokens; bootstrap 95% CIs on the core trio and bridge deltas.
- Detector precision: unit tests + real-output spot-check (the progressive detector was tightened
  after spot-checking). Quote rates as lower bounds; hand-validate before final numbers.
- Honest about power: en-IN is the only well-powered variety; en-AU/en-UK are sparse.

## Verified outcome
See `../RESULTS_SUMMARY.md`. Headline: GRPO maximises the eWAVE reward but yields the fewest
independent surface markers (decoupling = reward–quality gap, across all 3 families); humans
prefer the more dialectal output vs standard (T1) and the more natural output vs GRPO (T2).
