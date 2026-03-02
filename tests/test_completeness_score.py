from trait_completeness import compute_trait_completeness


def test_completeness_complete_status():
    pack = {
        "variants": [
            {
                "rsid": "rs1",
                "gene": "GENE1",
                "effect_allele": "A",
                "other_allele": "G",
                "genotype_map": {
                    "AA": {"effect_level": "Higher", "effect_label": "x", "notes": "n"},
                    "AG": {"effect_level": "Typical", "effect_label": "x", "notes": "n"},
                    "GG": {"effect_level": "Lower", "effect_label": "x", "notes": "n"},
                },
            }
        ],
        "evidence": [
            {
                "citation_id": "PMID:1",
                "title": "t",
                "year": 2020,
                "quote": "q",
            }
        ],
        "limitations": ["Not deterministic"],
        "metadata": {"curation_status": "Complete", "last_updated": "2026-02-15"},
    }
    out = compute_trait_completeness(pack)
    assert out["status"] == "Complete"
    assert out["score"] >= 85


def test_completeness_comingsoon_status():
    pack = {
        "variants": [],
        "evidence": [],
        "limitations": [],
        "metadata": {"curation_status": "Placeholder", "last_updated": ""},
    }
    out = compute_trait_completeness(pack)
    assert out["status"] == "ComingSoon"
    assert out["score"] < 60
