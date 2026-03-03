from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genomics_interpreter import build_prs_report_from_upload
from utils.rag_explainer import generate_trait_explanation


def test_generate_trait_explanation_non_empty_when_evidence_missing():
    trait = {
        "trait_name": "Sleep duration tendency",
        "category": "Neurobehavior",
        "gene": "ABCC9",
        "rsid": "rs11046205",
        "user_genotype": "AA",
        "effect_label": "Higher tendency",
        "coverage": 0.8,
    }
    patient = generate_trait_explanation(trait, [], mode="patient")
    doctor = generate_trait_explanation(trait, [], mode="doctor")
    assert patient.get("explanation", "").strip() != ""
    assert doctor.get("explanation", "").strip() != ""
    assert "educational" in patient.get("explanation", "").lower()


def test_all_traits_have_explanations_with_fallback():
    report = build_prs_report_from_upload(
        genotype_path="patient_A.txt",
        categories_selected=["Neurobehavior", "Nutrition", "Fitness"],
        include_optional_liver=False,
        red_flag_only=False,
    )
    traits = [t for t in report.get("traits", []) if isinstance(t, dict)]
    assert traits, "Expected at least one trait in report"
    for trait in traits:
        out = generate_trait_explanation(trait, [], mode="patient")
        assert out.get("explanation", "").strip() != ""


def test_patient_a_neurobehavior_coverage():
    patient_path = Path("patient_A.txt")
    assert patient_path.exists(), "patient_A.txt missing"
    report = build_prs_report_from_upload(
        genotype_path=str(patient_path),
        categories_selected=["Neurobehavior", "Nutrition", "Fitness"],
        include_optional_liver=False,
        red_flag_only=False,
    )
    traits = [t for t in report.get("traits", []) if isinstance(t, dict)]
    neuro = [t for t in traits if t.get("category") == "Neurobehavior" and str(t.get("user_genotype", "")).strip()]
    assert len(neuro) >= 5
