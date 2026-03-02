import json
from pathlib import Path

from scripts.validate_trait_packs import validate_dir


def test_validate_pack_dir(tmp_path: Path):
    packs = tmp_path / "trait_study_packs"
    packs.mkdir(parents=True)

    good = {
        "trait_id": "X",
        "category": "Nutrition",
        "subcategory": "Digestion",
        "trait_name": "Example",
        "description_public": "desc",
        "is_optional": False,
        "variants": [
            {
                "rsid": "rs1",
                "gene": "GENE",
                "effect_allele": "A",
                "other_allele": "G",
                "weight": 0.1,
                "unit": "beta",
                "genotype_map": {"AA": {"effect_level": "H", "effect_label": "L", "notes": "n"}},
                "evidence_strength": "Preliminary",
                "citations": ["PMID:TBD"],
            }
        ],
        "evidence": [
            {
                "citation_id": "PMID:TBD",
                "citation_type": "PMID",
                "title": "Placeholder",
                "year": 2020,
                "journal": "TBD",
                "url": "https://example.org",
                "quote": "q",
                "claim_tags": ["x"],
            }
        ],
        "limitations": ["l"],
        "safety": ["Educational only"],
        "metadata": {"curation_status": "Placeholder", "last_updated": "2026-02-15", "source_notes": "n"},
    }
    (packs / "ok.json").write_text(json.dumps(good), encoding="utf-8")

    code, errs = validate_dir(packs)
    assert code == 0
    assert errs == []
