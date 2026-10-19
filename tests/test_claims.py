"""Tests for the sentence splitting and claim cleaning."""
from probity.claims import claim_of, facts_of, is_checkable, split_claims


def test_split_removes_markdown_and_splits_sentences():
    assert split_claims("Sure! **Paris** is in France. It is big?") == ["Sure!", "Paris is in France.", "It is big?"]


def test_split_handles_lists():
    assert split_claims("- Rome is in Italy.\n2) Oslo is in Norway.") == ["Rome is in Italy.", "Oslo is in Norway."]


def test_claim_of_drops_polite_openers():
    assert claim_of("I'm sorry, but the city of Paris is in France.") == "The city of Paris is in France."
    assert claim_of("You're right, I apologize. Anyang-si is in Iraq.") != ""


def test_non_claims_are_skipped():
    assert not is_checkable("Hello there!")
    assert not is_checkable("Is it Paris?")
    assert not is_checkable("Paris.")
    assert is_checkable("Paris is the capital of France.")


def test_facts_of_keeps_only_facts():
    assert facts_of("Great question! Anyang-si is located in South Korea. Let me know if you need more.") == \
        "Anyang-si is located in South Korea."
