from polygenic import compute_prs


def test_compute_prs_coverage_and_score():
    trait_pack = {
        "trait_id": "TEST",
        "variants": [
            {"rsid": "rs1", "effect_allele": "A", "weight": 0.1},
            {"rsid": "rs2", "effect_allele": "G", "weight": 0.2},
            {"rsid": "rs3", "effect_allele": "T", "weight": -0.05},
        ],
    }
    user = [
        {"rsid": "rs1", "genotype": "AA"},
        {"rsid": "rs2", "genotype": "AG"},
    ]

    out = compute_prs(trait_pack, user)
    assert out["found"] == 2
    assert out["total"] == 3
    assert abs(out["coverage"] - (2 / 3)) < 1e-3
    # rs1 AA dosage=2 => 0.2, rs2 AG dosage=1 => 0.2
    assert abs(out["prs_score"] - 0.4) < 1e-9


def test_compute_prs_low_coverage_warning():
    trait_pack = {
        "trait_id": "TEST2",
        "variants": [
            {"rsid": "rs1", "effect_allele": "A", "weight": 0.1},
            {"rsid": "rs2", "effect_allele": "G", "weight": 0.2},
            {"rsid": "rs3", "effect_allele": "T", "weight": 0.3},
            {"rsid": "rs4", "effect_allele": "C", "weight": 0.4},
            {"rsid": "rs5", "effect_allele": "C", "weight": 0.5},
            {"rsid": "rs6", "effect_allele": "A", "weight": 0.6},
        ],
    }
    user = [{"rsid": "rs1", "genotype": "AA"}]
    out = compute_prs(trait_pack, user)
    assert out["coverage"] < 0.2
    assert out["confidence"] == "Low"
    assert out["warning"]
