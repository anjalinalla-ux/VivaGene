from genomics_interpreter import filter_prs_results_for_display


def test_red_flag_filter_subset_and_fallback():
    prs = [
        {
            "trait_id": "A",
            "bucket": "High",
            "confidence": "Medium",
            "coverage": 0.7,
            "completeness": {"status": "Complete"},
            "warnings": [],
        },
        {
            "trait_id": "B",
            "bucket": "Typical",
            "confidence": "High",
            "coverage": 0.9,
            "completeness": {"status": "Complete"},
            "warnings": [],
        },
    ]
    out, note = filter_prs_results_for_display(prs, red_flag_only=True)
    assert len(out) == 1
    assert out[0]["trait_id"] == "A"
    assert note == ""

    prs_none = [
        {
            "trait_id": "C",
            "bucket": "Typical",
            "confidence": "Medium",
            "coverage": 0.8,
            "completeness": {"status": "Complete"},
            "warnings": [],
        },
        {
            "trait_id": "D",
            "bucket": "Low",
            "confidence": "High",
            "coverage": 0.6,
            "completeness": {"status": "Complete"},
            "warnings": [],
        },
    ]
    out2, note2 = filter_prs_results_for_display(prs_none, red_flag_only=True)
    assert len(out2) == 2
    assert "No high flags detected" in note2
