from rag_retrieval import load_study_packs, retrieve_evidence


def test_retrieve_evidence_known_trait():
    packs = load_study_packs("trait_study_packs")
    assert packs
    known_trait_id = packs[0].get("trait_id", "")
    out = retrieve_evidence(known_trait_id, packs, k=2)

    assert out["trait_id"] == known_trait_id
    assert isinstance(out["passages"], list)
    assert len(out["passages"]) >= 1
    p = out["passages"][0]
    assert "quote" in p and "citation_id" in p
