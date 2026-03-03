from __future__ import annotations

import time
from pathlib import Path

from rag_evidence import ensure_trait_evidence, get_cached_evidence


BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence_corpus"


def retrieve_trait_evidence(
    trait: dict,
    min_citations: int = 2,
    min_snippets: int = 2,
    max_results: int = 8,
    retries: int = 1,
) -> dict:
    t = trait if isinstance(trait, dict) else {}
    payload = {
        "trait_id": str(t.get("trait_id", "")).strip(),
        "trait_name": str(t.get("trait_name", "")).strip(),
        "category": str(t.get("category", "")).strip(),
        "gene": str(t.get("gene", "")).strip(),
        "rsid": str(t.get("rsid", "")).strip(),
    }

    last_error = ""
    cached = get_cached_evidence(payload["trait_id"])
    cached_citations = cached.get("citations", []) if isinstance(cached, dict) and isinstance(cached.get("citations", []), list) else []
    cached_snippets = cached.get("snippets", []) if isinstance(cached, dict) and isinstance(cached.get("snippets", []), list) else []

    for attempt in range(max(1, int(retries) + 1)):
        try:
            ev = ensure_trait_evidence(
                payload,
                min_citations=min_citations,
                min_snippets=min_snippets,
                max_results=max_results,
            )
            citations = ev.get("citations", []) if isinstance(ev.get("citations", []), list) else []
            snippets = ev.get("snippets", []) if isinstance(ev.get("snippets", []), list) else []
            out = {
                "status": "found" if snippets else "missing",
                "trait_id": payload["trait_id"],
                "queries": ev.get("queries", []) if isinstance(ev.get("queries", []), list) else [],
                "citations": citations,
                "snippets": snippets,
                "retrieval_error": "",
            }
            return out
        except Exception as exc:
            last_error = str(exc)
            if cached_snippets:
                return {
                    "status": "found",
                    "trait_id": payload["trait_id"],
                    "queries": cached.get("queries", []) if isinstance(cached, dict) and isinstance(cached.get("queries", []), list) else [],
                    "citations": cached_citations,
                    "snippets": cached_snippets,
                    "retrieval_error": last_error,
                }
            if attempt < int(retries):
                time.sleep(0.5)

    return {
        "status": "missing",
        "trait_id": payload["trait_id"],
        "queries": [],
        "citations": [],
        "snippets": [],
        "retrieval_error": last_error or "Evidence retrieval failed",
    }
