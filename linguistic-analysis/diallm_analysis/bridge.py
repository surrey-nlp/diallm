"""Human-preference bridge (analysis F).

Links the linguistic measures to the *human* pairwise judgements collected on
Llama outputs (the six annotation CSVs). Establishes which linguistic measures
track human preference; the same per-condition measure profiles can then be read
off Qwen/Gemma (in the main table) to argue the pattern generalises beyond the
Llama-only human study.

Key outputs, per task and pooled (winner - loser deltas, bootstrap CI):
  Task 1 (instruct vs SFT_d): do humans prefer the MORE dialectal output?
  Task 2 (SFT_d vs GRPO_d):   does HIGHER density/stacking get DISpreferred?  <- the reward-quality gap
  Task 4 (broad vs targeted): is variety-targeting preferred?
"""
from __future__ import annotations
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .analyze import measure_doc, bootstrap_ci

DIALECT_TO_VARIETY = {"en-AU": "aus", "en-IN": "ind", "en-UK": "uk"}
# en-UK detectors live under variety key 'uk'; the JSONL variant is 'brit'.
VARIANT_TO_VARIETY = {"aus": "aus", "ind": "ind", "brit": "uk"}

BRIDGE_MEASURES = [
    "density_per1k", "diversity_types", "stacking_per_sent", "british_per1k",
    "feat_lexical", "ttr", "distinct2", "contraction_per1k", "politeness_per1k",
]


def load_annotations(csv_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(csv_dir, "annotations_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _parse_option(tok: str) -> Tuple[str, str]:
    """'instruct_base' -> ('instruct','base'); 'sft_aus' -> ('sft','aus')."""
    stage, variant = tok.split("_", 1)
    return stage, variant


def _winner_index(row, options: List[Tuple[str, str]]) -> int:
    """Return index of the human-preferred option, or -1 for tie/undecidable."""
    ws, wv = row["winner_stage"], row["winner_variant"]
    if ws == "tie" or wv == "tie":
        return -1
    # Task 4: both options are GRPO; the variant distinguishes broad vs targeted.
    stages = {o[0] for o in options}
    if stages == {"grpo"}:
        target = "all" if wv == "all" else "variety"
        for i, (_, var) in enumerate(options):
            if (var == "all") == (target == "all"):
                return i
        return -1
    # Tasks 1/2: the winning stage identifies the option.
    for i, (stage, _) in enumerate(options):
        if stage == ws:
            return i
    return -1


def build_bridge(annot: pd.DataFrame, responses: Dict[Tuple, str],
                 family: str = "llama") -> pd.DataFrame:
    """Build a winner/loser measure table for the pairwise tasks (1,2,4)."""
    recs = []
    cache: Dict[Tuple, Dict] = {}

    def measured(stage, variant, prompt_id, variety):
        key = (stage, variant, prompt_id, variety)
        if key not in cache:
            raw = responses.get((family, stage, variant, prompt_id))
            cache[key] = measure_doc(raw, variety) if raw is not None else None
        return cache[key]

    for _, row in annot.iterrows():
        if row["task_id"] not in (1, 2, 4):
            continue
        try:
            toks = row["comparison"].split("_vs_")
            options = [_parse_option(t) for t in toks]
        except Exception:
            continue
        if len(options) != 2:
            continue
        widx = _winner_index(row, options)
        if widx < 0:
            continue
        variety = VARIANT_TO_VARIETY.get(
            DIALECT_TO_VARIETY.get(row["dialect"], ""), None)
        if variety is None:
            continue
        win = options[widx]
        lose = options[1 - widx]
        mw = measured(win[0], win[1], row["prompt_id"], variety)
        ml = measured(lose[0], lose[1], row["prompt_id"], variety)
        if mw is None or ml is None:
            continue
        rec = {"task": int(row["task_id"]), "dialect": row["dialect"],
               "annotator": row["annotator"]}
        for m in BRIDGE_MEASURES:
            wv = mw.get(m, np.nan)
            lv = ml.get(m, np.nan)
            rec[f"delta_{m}"] = wv - lv if (wv == wv and lv == lv) else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def summarise_bridge(bridge: pd.DataFrame) -> pd.DataFrame:
    """Mean winner-minus-loser delta per measure, per task and pooled, with CI.
    A positive delta means humans preferred the output with MORE of that measure.
    """
    out = []
    groups = [("pooled", bridge)] + [(f"task{t}", bridge[bridge.task == t])
                                     for t in sorted(bridge.task.unique())]
    for label, g in groups:
        if g.empty:
            continue
        for m in BRIDGE_MEASURES:
            col = f"delta_{m}"
            mean, lo, hi = bootstrap_ci(g[col].values)
            sig = "" if (lo <= 0 <= hi) else "*"  # CI excludes 0
            out.append({"group": label, "measure": m, "n": int(g[col].notna().sum()),
                        "mean_delta": mean, "ci_lo": lo, "ci_hi": hi, "sig": sig})
    return pd.DataFrame(out)
