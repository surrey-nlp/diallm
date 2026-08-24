"""Per-output measures, corpus-level distribution distances, and bootstrap stats.

Three measure families support the paper's findings:
  * density   -> how marked the output is        (what the eWAVE reward maximises)
  * diversity -> how many feature TYPES are used  (reward-independent)
  * stacking  -> features crammed per sentence    (reward-independent; "mock dialect")
Plus lexical-diversity / degeneration and register measures.
"""
from __future__ import annotations
import re
import math
from typing import Dict, List, Callable

import numpy as np

from .features import (
    get_nlp, clean_text, extract_features, FEATURE_CATEGORIES,
    american_spellings,
)
from .lexicons import POLITENESS_MARKERS

try:
    from lexical_diversity import lex_div as ld
    _HAVE_LD = True
except Exception:
    _HAVE_LD = False

_CONTRACTION = re.compile(r"\b\w+(?:n't|'re|'ll|'ve|'d|'m|'s)\b", re.IGNORECASE)


def _ttr(tokens: List[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else float("nan")


def _distinct_n(tokens: List[str], n: int) -> float:
    if len(tokens) < n:
        return float("nan")
    grams = list(zip(*[tokens[i:] for i in range(n)]))
    return len(set(grams)) / len(grams) if grams else float("nan")


def _max_token_share(tokens: List[str]) -> float:
    """Share of the single most frequent token: a crude degeneration signal."""
    if not tokens:
        return float("nan")
    from collections import Counter
    return Counter(tokens).most_common(1)[0][1] / len(tokens)


def measure_doc(raw_text: str, variety: str) -> Dict[str, float]:
    """Compute every per-output measure for one raw response string."""
    cleaned = clean_text(raw_text)
    artifact = int(cleaned != re.sub(r"\s+", " ", (raw_text or "")).strip())
    nlp = get_nlp()
    doc = nlp(cleaned)

    alpha = [t.text.lower() for t in doc if t.is_alpha]
    n_tok = len(alpha)
    n_sent = max(1, len(list(doc.sents)))

    feats = extract_features(doc, variety)
    total = sum(feats.values())
    types_present = sum(1 for v in feats.values() if v > 0)
    brit = feats["orthographic_british"]
    amer = american_spellings(doc)

    row: Dict[str, float] = {}
    row.update({f"feat_{k}": v for k, v in feats.items()})
    row["n_tokens"] = n_tok
    row["n_sentences"] = n_sent
    row["artifact_stripped"] = artifact
    row["max_token_share"] = round(_max_token_share(alpha), 4)

    # --- core trio ---
    row["density_per1k"] = round(total / n_tok * 1000, 3) if n_tok else float("nan")
    row["diversity_types"] = types_present
    row["stacking_per_sent"] = round(total / n_sent, 3)

    # --- orthography ---
    row["british_per1k"] = round(brit / n_tok * 1000, 3) if n_tok else float("nan")
    row["british_ratio"] = round(brit / (brit + amer), 3) if (brit + amer) else float("nan")

    # --- lexical diversity / degeneration ---
    row["ttr"] = round(_ttr(alpha), 4)
    row["distinct2"] = round(_distinct_n(alpha, 2), 4)
    if _HAVE_LD and n_tok >= 50:
        try:
            row["mtld"] = round(ld.mtld(alpha), 3)
        except Exception:
            row["mtld"] = float("nan")
    else:
        row["mtld"] = float("nan")

    # --- register ---
    contractions = len(_CONTRACTION.findall(cleaned))
    polite = sum(1 for w in alpha if w in POLITENESS_MARKERS)
    row["contraction_per1k"] = round(contractions / n_tok * 1000, 3) if n_tok else float("nan")
    row["politeness_per1k"] = round(polite / n_tok * 1000, 3) if n_tok else float("nan")
    return row


# ---------------------------------------------------------------------------
# Corpus-level feature-category distribution + Jensen-Shannon divergence (C1).
# ---------------------------------------------------------------------------
def category_distribution(rows: List[Dict[str, float]], smoothing: float = 0.5) -> np.ndarray:
    """Smoothed probability vector over FEATURE_CATEGORIES for a group of rows."""
    counts = np.array([
        sum(r.get(f"feat_{c}", 0) for r in rows) for c in FEATURE_CATEGORIES
    ], dtype=float)
    counts += smoothing
    return counts / counts.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2), in [0, 1]."""
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals.
# ---------------------------------------------------------------------------
def bootstrap_ci(values, stat: Callable = np.mean, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 1234):
    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(arr, arr.size, replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(stat(arr)), 3), round(float(lo), 3), round(float(hi), 3))
