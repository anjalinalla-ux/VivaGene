from rag_generator import generate_trait_explanation_rag


def test_refusal_when_low_evidence_without_model_call(monkeypatch):
    called = {"v": False}

    def fake_call(system, user):
        called["v"] = True
        raise RuntimeError("should not be called")

    monkeypatch.setattr("rag_generator._call_openrouter_json", fake_call)

    trait = {"trait_id": "T1", "trait_name": "Trait"}
    snippets = [{"citation_id": "CURATED:T1:1", "text": "placeholder text"}]
    quality = {"quality": "Low"}

    out = generate_trait_explanation_rag(trait, snippets, quality)
    assert out["unsupported"] is True
    assert "Insufficient evidence" in out["refusal_reason"]
    assert called["v"] is False
