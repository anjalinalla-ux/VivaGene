from pathlib import Path

from scripts.build_trait_packs import build_packs


def test_build_packs_from_fixture_csv(tmp_path: Path):
    csv_path = tmp_path / "fixture.csv"
    csv_path.write_text(
        "trait_id,trait_name,category,rsid,gene,genotype,effect_label,effect_level,explanation,evidence_strength\n"
        "NUTR_LACTOSE,Lactose tolerance,Nutrition,rs4988235,LCT,CC,Low tolerance,LOW,Example explanation,strong\n",
        encoding="utf-8",
    )

    out_root = tmp_path / "packs"
    summary = build_packs(csv_path, out_root)

    assert summary["Neurobehavior"]["total"] == 25
    assert summary["Nutrition"]["total"] == 25
    assert summary["Fitness"]["total"] == 20
    assert summary["Liver"]["total"] == 20

    files = list(out_root.rglob("*.json"))
    assert len(files) == 90
