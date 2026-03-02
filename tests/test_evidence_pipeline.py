from pathlib import Path
import importlib.util
import sys


def test_corpus_builder_and_retrieval(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "build_evidence_corpus",
        root / "scripts" / "build_evidence_corpus.py",
    )
    bec = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(bec)
    import genomics_interpreter as gi

    evidence_dir = tmp_path / "evidence_corpus"
    index_path = evidence_dir / "index.jsonl"
    search_log_path = evidence_dir / "search_log.jsonl"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bec, "EVIDENCE_DIR", evidence_dir)
    monkeypatch.setattr(bec, "INDEX_PATH", index_path)
    monkeypatch.setattr(bec, "SEARCH_LOG_PATH", search_log_path)

    def fake_packs(base_dir="trait_study_packs"):
        return [
            {
                "trait_id": "NB_SLEEP_001",
                "trait_name": "Sleep tendency",
                "category": "Neurobehavior",
                "keywords": ["sleep", "duration"],
                "variants": [{"gene": "ADORA2A", "rsid": "rs5751876"}],
            },
            {
                "trait_id": "NUT_CAFF_001",
                "trait_name": "Caffeine response",
                "category": "Nutrition",
                "keywords": ["caffeine"],
                "variants": [{"gene": "CYP1A2", "rsid": "rs762551"}],
            },
        ]

    monkeypatch.setattr(bec, "load_all_trait_packs", fake_packs)

    def fake_fetch(query, page_size=10):
        return (
            [
                {
                    "pmid": "123456",
                    "pmcid": "",
                    "doi": "10.1000/test",
                    "title": f"Paper for {query}",
                    "journalTitle": "Demo Journal",
                    "pubYear": "2022",
                    "abstractText": "A short evidence snippet for testing retrieval.",
                    "source": "MED",
                }
            ],
            "https://example.org/request",
        )

    monkeypatch.setattr(bec, "_fetch_europe_pmc", fake_fetch)
    monkeypatch.setattr(bec.time, "sleep", lambda *_: None)

    summary = bec.build_evidence_corpus(max_traits=2)
    assert summary["processed_traits"] == 2
    assert index_path.exists()
    assert search_log_path.exists()
    assert search_log_path.read_text(encoding="utf-8").strip()

    monkeypatch.setattr(gi, "EVIDENCE_INDEX_PATH", index_path)
    monkeypatch.setattr(gi, "EVIDENCE_SEARCH_LOG_PATH", search_log_path)
    gi._EVIDENCE_CACHE["sig"] = ""
    gi._EVIDENCE_CACHE["rows"] = []

    rows = gi.load_local_corpus(index_path)
    assert len(rows) >= 1
    hits = gi.retrieve_evidence(
        {"trait_id": "NB_SLEEP_001", "gene": "ADORA2A", "rsid": "rs5751876", "trait_name": "Sleep tendency"},
        k=3,
    )
    assert hits

    report = gi.normalize_report(
        {
            "summary": {},
            "traits": [
                {"trait_id": "NB_SLEEP_001", "evidence_status": "found", "explanation": "Evidence-backed."},
                {"trait_id": "NUT_CAFF_001", "evidence_status": "missing", "explanation": ""},
            ],
        }
    )
    assert len(report["traits"]) == 2
    assert any(t.get("evidence_status") == "missing" for t in report["traits"])
