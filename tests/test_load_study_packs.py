from rag_retrieval import load_study_packs


def test_load_study_packs_required_keys():
    packs = load_study_packs("trait_study_packs")
    assert isinstance(packs, list)
    assert len(packs) >= 1

    required = {
        "trait_id",
        "category",
        "trait_name",
        "description_public",
        "variants",
        "evidence",
        "limitations",
        "safety",
    }
    for p in packs:
        assert required.issubset(set(p.keys()))
