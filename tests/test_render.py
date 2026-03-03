import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genomics_interpreter import build_prs_report_from_upload
from utils.rag_explainer import generate_trait_explanation


def test_report_traits_get_renderable_explanation_strings():
    report = build_prs_report_from_upload(
        genotype_path="patient_A.txt",
        categories_selected=["Neurobehavior", "Nutrition", "Fitness"],
        include_optional_liver=False,
        red_flag_only=False,
    )
    traits = [t for t in report.get("traits", []) if isinstance(t, dict)]
    assert traits
    for trait in traits[:20]:
        out = generate_trait_explanation(trait, [], mode="patient")
        assert isinstance(out.get("explanation", ""), str)
        assert out.get("explanation", "").strip() != ""
