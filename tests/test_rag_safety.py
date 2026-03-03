import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.rag_explainer import generate_trait_explanation


def test_rag_safety_empty_evidence_returns_safe_message():
    trait = {"trait_name": "Focus tendency", "category": "Neurobehavior", "gene": "COMT", "rsid": "rs4680", "user_genotype": "AG"}
    out = generate_trait_explanation(trait, [], mode="patient")
    assert out["status"] == "missing"
    assert "could not retrieve enough research" in out["explanation"].lower()
    assert "medical advice" in out["explanation"].lower()


def test_doctor_mode_includes_citation_when_snippets_present():
    trait = {"trait_name": "Focus tendency", "category": "Neurobehavior", "gene": "COMT", "rsid": "rs4680", "user_genotype": "AG", "effect_label": "higher variability"}
    snippets = [
        {"text": "COMT variation is associated with dopamine signaling differences.", "citation": {"pmid": "12345", "title": "COMT and cognition", "year": "2019", "url": "https://europepmc.org/article/MED/12345"}},
        {"text": "Associations are probabilistic and not diagnostic.", "citation": {"pmid": "67890", "title": "Genetics and behavior", "year": "2021", "url": "https://europepmc.org/article/MED/67890"}},
    ]
    out = generate_trait_explanation(trait, snippets, mode="doctor")
    assert out["status"] == "found"
    assert "[PMID:" in out["explanation"]
