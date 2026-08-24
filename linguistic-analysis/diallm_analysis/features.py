"""Text cleaning, spaCy pipeline, and dialectal feature detectors.

Each detector returns an integer count of firings for one output. Detectors are
deliberately high-precision (rule-based over spaCy POS/dependency parses or tight
regexes). spaCy is trained on standard English and mis-tags some dialectal forms,
so every morphosyntactic detector MUST be precision-validated (see
tests/test_detectors.py and `validate` mode in run_analysis.py) before its rate
is reported in the paper.
"""
from __future__ import annotations
import re
import functools
from typing import Dict, List

import spacy

from .lexicons import (
    LEXICON, BRITISH_SPELLINGS, AMERICAN_SPELLINGS, STATIVE_VERBS,
    POLITENESS_MARKERS,
)

# ---------------------------------------------------------------------------
# Cleaning: strip chat-template / tokenizer artifacts seen in the raw outputs
# ("xfordassistantxford", bare role tags, special-token residue). Conservative:
# we only remove obvious template/special-token junk, never normal content.
# ---------------------------------------------------------------------------
_ARTIFACT_PATTERNS = [
    re.compile(r"x*ford", re.IGNORECASE),                 # 'xford' residue
    re.compile(r"<\|.*?\|>"),                              # <|...|> special tokens
    re.compile(r"\[/?INST\]|\[/?SYS\]", re.IGNORECASE),    # llama inst tags
    re.compile(r"<\/?s>"),                                  # bos/eos
    re.compile(r"\b(?:assistant|user|system|model)\b(?=\s*(?:assistant|user|system|model|$))",
               re.IGNORECASE),                              # repeated role tags
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text
    for pat in _ARTIFACT_PATTERNS:
        t = pat.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@functools.lru_cache(maxsize=1)
def get_nlp():
    # NER not needed; disabling it speeds up the laptop run.
    return spacy.load("en_core_web_sm", disable=["ner"])


# ---------------------------------------------------------------------------
# Lexical + orthographic detectors (operate on the token surface).
# ---------------------------------------------------------------------------
def lexical_markers(doc, variety: str) -> int:
    """Count variety-specific lexical markers (tokens + multiword phrases).

    variety='all' (the broad/implicit thread, or base/instruct references) counts
    the union across all three variety lexicons.
    """
    if variety == "all":
        varieties = list(LEXICON)
    elif variety in LEXICON:
        varieties = [variety]
    else:
        return 0
    lowered = [t.text.lower() for t in doc]
    text = " ".join(lowered)
    n = 0
    for v in varieties:
        toks = LEXICON[v]["tokens"]
        n += sum(1 for w in lowered if w in toks)
        for ph in LEXICON[v]["phrases"]:
            n += len(re.findall(r"\b" + re.escape(ph) + r"\b", text))
    return n


def british_spellings(doc) -> int:
    return sum(1 for t in doc if t.text.lower() in BRITISH_SPELLINGS)


def american_spellings(doc) -> int:
    return sum(1 for t in doc if t.text.lower() in AMERICAN_SPELLINGS)


# ---------------------------------------------------------------------------
# Morphosyntactic detectors (spaCy parse + tight regex).
# ---------------------------------------------------------------------------
# Conservative stative set for the progressive detector: verbs whose progressive
# is clearly non-standard in en-IN. Excludes want/like/love/need/wish/have/see
# etc. because their progressives ("been wanting to see") are standard catenatives
# and inflate false positives.
_STATIVE_PROG = {
    "understand", "know", "believe", "prefer", "contain", "depend", "consist",
    "possess", "own", "belong", "doubt", "realise", "realize", "comprise",
}


def prog_stative(doc) -> int:
    """Progressive with stative verbs: 'I am understanding', 'persons are preferring'.
    High-precision: requires a 'be' auxiliary, excludes catenative '...ing to' and
    predicate-adjective uses ('be understanding')."""
    n = 0
    for t in doc:
        if t.tag_ == "VBG" and t.lemma_.lower() in _STATIVE_PROG:
            # skip catenative "<verb>ing to <inf>" (standard English)
            if t.i + 1 < len(doc) and doc[t.i + 1].text.lower() == "to":
                continue
            # skip predicate-adjective use ('be understanding', 'be appealing')
            if t.dep_ in ("acomp", "amod", "attr"):
                continue
            has_be = any(c.dep_ in ("aux", "auxpass") and c.lemma_ == "be"
                         for c in t.children)
            if not has_be and t.head.lemma_ == "be":
                has_be = True
            if has_be:
                n += 1
    return n


_WERE_GEN = re.compile(r"\b(i|he|she|it)\s+were\b", re.IGNORECASE)
_WAS_GEN = re.compile(r"\b(we|they|you)\s+was\b", re.IGNORECASE)


def be_agreement(doc) -> int:
    """was/were generalisation: 'she were happy', 'they was'."""
    text = doc.text
    return len(_WERE_GEN.findall(text)) + len(_WAS_GEN.findall(text))


_DITRANSITIVE = {"give", "tell", "show", "send", "buy", "get", "make", "offer",
                 "teach", "lend", "bring", "pass", "hand", "owe", "promise",
                 "ask", "call", "find", "leave"}


def possessive_me(doc) -> int:
    """Object pronoun as possessive: 'me brother', 'me order'.
    Excludes ditransitive contexts ('give me X')."""
    n = 0
    for i, t in enumerate(doc):
        if t.text.lower() == "me" and i + 1 < len(doc):
            nxt = doc[i + 1]
            if nxt.pos_ in ("NOUN", "PROPN"):
                prev = doc[i - 1] if i > 0 else None
                if prev is not None and prev.lemma_.lower() in _DITRANSITIVE:
                    continue
                n += 1
    return n


_MASS_NOUNS = {"money", "information", "advice", "furniture", "equipment",
               "luggage", "homework", "research", "software", "hardware",
               "feedback", "knowledge", "traffic", "scenery", "machinery"}


def mass_noun_plural(doc) -> int:
    """Pluralised mass nouns: 'moneys', 'informations', 'advices' (en-IN)."""
    n = 0
    for t in doc:
        if t.tag_ == "NNS" and t.lemma_.lower() in _MASS_NOUNS:
            n += 1
    return n


_BARE_ADVERB = re.compile(
    r"\b(drive|drives|driving|drove|run|runs|ran|walk|walks|talk|talks|go|goes|"
    r"come|comes|work|works|speak|speaks|move|moves|play|plays)\s+"
    r"(slow|quick|real|good|bad|easy|loud|different|nice|safe|careful|quiet)\b",
    re.IGNORECASE,
)


def bare_adverb(doc) -> int:
    """Adjective used adverbially: 'drive slow', 'talk loud' (en-AU)."""
    return len(_BARE_ADVERB.findall(doc.text))


_INVARIANT_TAG = re.compile(r",\s*(isn't it|no|na|naa)\s*\?", re.IGNORECASE)


def invariant_tag(doc) -> int:
    """Invariant tag question: '..., isn't it?', '..., no?' (en-IN)."""
    return len(_INVARIANT_TAG.findall(doc.text))


# Map of detectors and which variety each is primarily diagnostic for.
MORPHO_DETECTORS = {
    "prog_stative": prog_stative,       # en-IN (also generic non-standard)
    "be_agreement": be_agreement,       # en-UK Northern
    "possessive_me": possessive_me,     # en-UK Northern
    "mass_noun_plural": mass_noun_plural,  # en-IN
    "bare_adverb": bare_adverb,         # en-AU
    "invariant_tag": invariant_tag,     # en-IN
}

# Feature categories aggregated into the density / diversity / stacking measures.
FEATURE_CATEGORIES = ["lexical", "orthographic_british"] + list(MORPHO_DETECTORS)


def extract_features(doc, variety: str) -> Dict[str, int]:
    """Return per-category firing counts for one cleaned spaCy Doc."""
    feats = {
        "lexical": lexical_markers(doc, variety),
        "orthographic_british": british_spellings(doc),
    }
    for name, fn in MORPHO_DETECTORS.items():
        feats[name] = fn(doc)
    return feats
