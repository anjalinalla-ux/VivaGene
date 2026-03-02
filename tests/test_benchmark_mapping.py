from validation.baselines import prs_bucket_direction, benchmark_agreement


def test_benchmark_mapping_comparable_and_not():
    assert prs_bucket_direction("High") == "high"
    assert prs_bucket_direction("Insufficient data") == "unknown"

    prs_obj_ok = {
        "trait_id": "T1",
        "bucket": "High",
        "variant_hits": [
            {
                "effect_label": "Slightly higher tendency",
                "evidence_strength": "High",
                "weight": 0.2,
            }
        ],
    }
    out1 = benchmark_agreement(prs_obj_ok)
    assert out1["comparable"] == 1

    prs_obj_bad = {"trait_id": "T2", "bucket": "Insufficient data", "variant_hits": []}
    out2 = benchmark_agreement(prs_obj_bad)
    assert out2["comparable"] == 0
