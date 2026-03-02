from rag_generator import enforce_citations


def test_enforce_citations_failures_and_success():
    snippets = [
        {"citation_id": "CID1", "text": "a"},
        {"citation_id": "CID2", "text": "b"},
    ]

    missing = {
        "trait_id": "T",
        "explanation_1": "This has no citation.",
        "explanation_2": "",
        "unsupported": False,
        "refusal_reason": "",
    }
    out1 = enforce_citations(missing, snippets)
    assert out1["unsupported"] is True

    bad_idx = {
        "trait_id": "T",
        "explanation_1": "Claim [3]",
        "explanation_2": "",
        "unsupported": False,
        "refusal_reason": "",
    }
    out2 = enforce_citations(bad_idx, snippets)
    assert out2["unsupported"] is True

    good = {
        "trait_id": "T",
        "explanation_1": "Claim [1][2]",
        "explanation_2": "Second [2]",
        "unsupported": False,
        "refusal_reason": "",
    }
    out3 = enforce_citations(good, snippets)
    assert out3["unsupported"] is False
    assert out3["citations_used"] == ["CID1", "CID2"]
