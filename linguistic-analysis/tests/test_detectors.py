"""Precision sanity tests for the dialectal detectors.

These assert that each detector fires on a positive example and stays silent on
a near-miss negative. They are NOT a substitute for hand-validating detector
precision on real model outputs (see README), but they guard against regressions.
"""
import pytest
from diallm_analysis.features import (
    get_nlp, clean_text, prog_stative, be_agreement, possessive_me,
    mass_noun_plural, bare_adverb, invariant_tag, lexical_markers,
    british_spellings,
)

nlp = get_nlp()


def d(text):
    return nlp(clean_text(text))


def test_prog_stative():
    assert prog_stative(d("I am understanding the problem now.")) == 1
    assert prog_stative(d("She is having a doubt about it.")) == 1
    assert prog_stative(d("I am running to the shop.")) == 0  # eventive, not stative


def test_be_agreement():
    assert be_agreement(d("She were dead chuffed.")) == 1
    assert be_agreement(d("They was late again.")) == 1
    assert be_agreement(d("We were ready on time.")) == 0


def test_possessive_me():
    assert possessive_me(d("Me brother is coming round.")) == 1
    assert possessive_me(d("Please give me the book.")) == 0  # ditransitive


def test_mass_noun_plural():
    assert mass_noun_plural(d("He gave me many advices and informations.")) == 2
    assert mass_noun_plural(d("He gave me good advice.")) == 0


def test_bare_adverb():
    assert bare_adverb(d("You should drive slow on that road.")) == 1
    assert bare_adverb(d("That was a slow car.")) == 0


def test_invariant_tag():
    assert invariant_tag(d("You are coming, no?")) == 1
    assert invariant_tag(d("It is fine, isn't it?")) == 1


def test_lexical_markers():
    assert lexical_markers(d("Let's grab a feed this arvo at the servo."), "aus") >= 2
    assert lexical_markers(d("Kindly do the needful at the earliest."), "ind") >= 2
    assert lexical_markers(d("There were nowt and summat in the ginnel."), "uk") >= 2


def test_british_spellings():
    assert british_spellings(d("The colour of the theatre, I realise, is grey.")) == 4
    assert british_spellings(d("The color of the theater is gray.")) == 0
