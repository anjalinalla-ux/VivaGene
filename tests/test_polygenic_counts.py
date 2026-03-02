from genomics_polygenic import normalize_genotype, count_effect_alleles


def test_normalize_genotype_and_counts():
    assert normalize_genotype("GA") == "AG"
    assert normalize_genotype("--") == ""
    assert normalize_genotype("00") == ""
    assert normalize_genotype("") == ""
    assert count_effect_alleles("AG", "A") == 1
    assert count_effect_alleles("AA", "A") == 2
    assert count_effect_alleles("GG", "A") == 0
