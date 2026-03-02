from rag_retriever import retrieve_evidence


def test_retrieve_gene_overlap_boost(monkeypatch):
    corpus = [
        {
            "citation_id": "A",
            "title": "General",
            "year": 2020,
            "source": "Local",
            "url": "",
            "genes": ["GENEX"],
            "trait_ids": ["T2"],
            "text": "general nutrition text",
        },
        {
            "citation_id": "B",
            "title": "Gene hit",
            "year": 2021,
            "source": "Local",
            "url": "",
            "genes": ["ADORA2A"],
            "trait_ids": ["T1"],
            "text": "sleep and adora2a association text",
        },
        {
            "citation_id": "C",
            "title": "Other",
            "year": 2022,
            "source": "Local",
            "url": "",
            "genes": ["GENEY"],
            "trait_ids": ["T3"],
            "text": "other text",
        },
    ]

    monkeypatch.setattr("rag_retriever.load_corpus", lambda base_dir="evidence_corpus": corpus)

    trait_pack = {
        "trait_id": "T1",
        "variants": [{"gene": "ADORA2A"}],
    }

    out = retrieve_evidence("sleep ADORA2A", trait_pack, k=3)
    assert out[0]["citation_id"] == "B"
