from genomics_polygenic import compute_prs_for_trait


def test_compute_prs_for_trait_basic():
    pack = {
        "trait_id": "T1",
        "trait_name": "Trait One",
        "category": "Nutrition",
        "subcategory": "Digestion",
        "variants": [
            {
                "rsid": "rs1",
                "gene": "GENE1",
                "effect_allele": "A",
                "other_allele": "G",
                "weight": 0.2,
                "unit": "beta",
                "genotype_map": {
                    "AA": {"effect_level": "Higher", "effect_label": "High", "notes": "n"},
                    "AG": {"effect_level": "Typical", "effect_label": "Mid", "notes": "n"},
                    "GG": {"effect_level": "Lower", "effect_label": "Low", "notes": "n"},
                },
                "evidence_strength": "Moderate",
                "citations": ["PMID:1"],
            },
            {
                "rsid": "rs2",
                "gene": "GENE2",
                "effect_allele": "G",
                "other_allele": "A",
                "weight": 0.3,
                "unit": "beta",
                "genotype_map": {
                    "AA": {"effect_level": "Lower", "effect_label": "Low", "notes": "n"},
                    "AG": {"effect_level": "Typical", "effect_label": "Mid", "notes": "n"},
                    "GG": {"effect_level": "Higher", "effect_label": "High", "notes": "n"},
                },
                "evidence_strength": "High",
                "citations": ["PMID:2"],
            },
        ],
        "evidence": [{"citation_id": "PMID:1", "title": "t", "year": 2020, "quote": "q"}],
        "limitations": ["not deterministic"],
        "safety": ["Educational only"],
        "metadata": {"curation_status": "Complete", "last_updated": "2026-02-15", "source_notes": "x"},
    }
    user_map = {"rs1": "AA", "rs2": "AG"}
    out = compute_prs_for_trait(pack, user_map)
    assert out["num_variants_total"] == 2
    assert out["num_variants_found"] == 2
    assert out["coverage"] == 1.0
    # rs1 AA => 2*0.2 = 0.4, rs2 AG => 1*0.3 = 0.3
    assert abs(out["prs_raw"] - 0.7) < 1e-9
    assert out["bucket"] in {"Low", "Typical", "High"}
