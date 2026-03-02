from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explain import build_explanation


def test_insufficient_evidence_returns_pending():
    trait = {"trait_name": "Sleep tendency", "category": "Neurobehavior"}
    evidence = {"citations": [], "snippets": [], "queries": ["ADORA2A rs5751876 sleep"]}
    out = build_explanation(trait, evidence, "Doctor (technical)")
    assert "Evidence pending" in out.get("summary", "")
    assert out.get("life_impact", "") == ""


def test_doctor_mode_has_citation_when_supported():
    trait = {"trait_name": "Sleep tendency", "category": "Neurobehavior", "effect_label": "Elevated sensitivity"}
    evidence = {
        "citations": [
            {"pmid": "12345", "title": "Paper A", "year": "2020", "url": "https://europepmc.org/article/MED/12345"},
            {"pmid": "67890", "title": "Paper B", "year": "2021", "url": "https://europepmc.org/article/MED/67890"},
        ],
        "snippets": [
            {"text": "Variant associated with sleep latency differences.", "citation": {"pmid": "12345"}},
            {"text": "Association replicated in population cohorts.", "citation": {"pmid": "67890"}},
        ],
        "queries": ["ADORA2A sleep"],
    }
    out = build_explanation(trait, evidence, "Doctor (technical)")
    assert "Evidence pending" not in out.get("summary", "")
    assert "[PMID:" in out.get("summary", "")
