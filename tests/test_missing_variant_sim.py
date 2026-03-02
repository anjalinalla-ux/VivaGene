from validation.scoring_sim import simulate_missing_variants


def test_simulate_missing_variants_fraction():
    variants = [{"rsid": f"rs{i}", "genotype": "AG"} for i in range(100)]
    out = simulate_missing_variants(variants, missing_rate=0.3, seed=7)
    dropped = 100 - len(out)
    assert 20 <= dropped <= 40
