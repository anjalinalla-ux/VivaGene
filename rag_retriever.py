from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


_CORPUS_CACHE: dict[str, Any] = {"path": None, "corpus": None}
_INDEX_CACHE: dict[str, Any] = {"sig": None, "vectorizer": None, "matrix": None}


def _normalize_snippet(obj: dict) -> dict:
    text = str(obj.get("snippet_text", "") or obj.get("text", "")).strip()
    rsid_raw = obj.get("rsid", obj.get("rsids", []))
    if isinstance(rsid_raw, str):
        rsids = [x.strip().lower() for x in rsid_raw.replace(";", ",").split(",") if x.strip()]
    elif isinstance(rsid_raw, list):
        rsids = [str(x).strip().lower() for x in rsid_raw if str(x).strip()]
    else:
        rsids = []
    return {
        "citation_id": str(obj.get("citation_id", "")).strip(),
        "title": str(obj.get("title", "")).strip(),
        "year": obj.get("year", ""),
        "source": str(obj.get("source", "")).strip() or "LocalCorpus",
        "url": str(obj.get("url", "")).strip(),
        "authors": str(obj.get("authors", obj.get("authorString", ""))).strip(),
        "doi": str(obj.get("doi", "")).strip(),
        "pmid": str(obj.get("pmid", "")).strip(),
        "rsids": rsids,
        "genes": [str(g).strip().upper() for g in (obj.get("genes", []) if isinstance(obj.get("genes", []), list) else []) if str(g).strip()],
        "trait_ids": [str(t).strip() for t in (obj.get("trait_ids", []) if isinstance(obj.get("trait_ids", []), list) else []) if str(t).strip()],
        "text": text,
        "snippet_text": text,
    }


def load_corpus(base_dir: str = "evidence_corpus") -> list[dict]:
    base = Path(base_dir)
    sig = str(base.resolve()) if base.exists() else str(base)
    if _CORPUS_CACHE.get("path") == sig and isinstance(_CORPUS_CACHE.get("corpus"), list):
        return _CORPUS_CACHE["corpus"]

    corpus: list[dict] = []
    snippets_path = base / "snippets.jsonl"
    manual_path = base / "manual_trait_notes.json"

    if snippets_path.exists():
        for line in snippets_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                corpus.append(_normalize_snippet(obj))

    if manual_path.exists():
        try:
            manual = json.loads(manual_path.read_text(encoding="utf-8"))
        except Exception:
            manual = {}
        if isinstance(manual, dict):
            for trait_id, entries in manual.items():
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    obj = dict(e)
                    tids = obj.get("trait_ids", [])
                    if not isinstance(tids, list):
                        tids = []
                    if str(trait_id).strip() and str(trait_id).strip() not in tids:
                        tids.append(str(trait_id).strip())
                    obj["trait_ids"] = tids
                    corpus.append(_normalize_snippet(obj))

    _CORPUS_CACHE["path"] = sig
    _CORPUS_CACHE["corpus"] = corpus
    return corpus


def _index_text(sn: dict) -> str:
    return " ".join(
        [
            sn.get("title", ""),
            sn.get("text", ""),
            " ".join(sn.get("rsids", [])),
            " ".join(sn.get("genes", [])),
            " ".join(sn.get("trait_ids", [])),
        ]
    ).strip()


def build_tfidf_index(corpus: list[dict]):
    if not isinstance(corpus, list):
        raise ValueError("corpus must be a list")
    sig = f"len:{len(corpus)}|ids:{'|'.join(str(c.get('citation_id','')) for c in corpus[:20])}"
    if _INDEX_CACHE.get("sig") == sig and _INDEX_CACHE.get("vectorizer") is not None:
        return _INDEX_CACHE["vectorizer"], _INDEX_CACHE["matrix"]

    texts = [_index_text(c) for c in corpus]
    if not texts:
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform([""])
    else:
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(texts)

    _INDEX_CACHE["sig"] = sig
    _INDEX_CACHE["vectorizer"] = vec
    _INDEX_CACHE["matrix"] = mat
    return vec, mat


def retrieve_evidence(query: str, trait_pack: dict, k: int = 12) -> list[dict]:
    corpus = load_corpus()
    if not corpus:
        return []

    vec, mat = build_tfidf_index(corpus)
    q = str(query or "").strip()
    qv = vec.transform([q])
    base_scores = (mat @ qv.T).toarray().reshape(-1)

    trait_id = str((trait_pack or {}).get("trait_id", "")).strip()
    trait_name = str((trait_pack or {}).get("trait_name", "")).strip().lower()
    category = str((trait_pack or {}).get("category", "")).strip().lower()
    subcategory = str((trait_pack or {}).get("subcategory", "")).strip().lower()
    keywords = [str(x).strip().lower() for x in ((trait_pack or {}).get("keywords", []) if isinstance((trait_pack or {}).get("keywords", []), list) else []) if str(x).strip()]
    genes = set()
    rsids = set()
    variants = (trait_pack or {}).get("variants", [])
    if isinstance(variants, list):
        for v in variants:
            if isinstance(v, dict):
                g = str(v.get("gene", "")).strip().upper()
                if g:
                    genes.add(g)
                r = str(v.get("rsid", "")).strip().lower()
                if r:
                    rsids.add(r)

    # First pass: keep only likely candidates when possible.
    first_pass = []
    for sn in corpus:
        sn_genes = set(sn.get("genes", [])) if isinstance(sn.get("genes", []), list) else set()
        sn_rsids = set(sn.get("rsids", [])) if isinstance(sn.get("rsids", []), list) else set()
        sn_traits = set(sn.get("trait_ids", [])) if isinstance(sn.get("trait_ids", []), list) else set()
        if (trait_id and trait_id in sn_traits) or (genes and genes & sn_genes) or (rsids and rsids & sn_rsids):
            first_pass.append(sn)
    if not first_pass:
        first_pass = corpus

    idx_map = {id(sn): i for i, sn in enumerate(corpus)}
    scored = []
    for sn in first_pass:
        i = idx_map.get(id(sn), 0)
        score = float(base_scores[i]) if i < len(base_scores) else 0.0
        sn_genes = set(sn.get("genes", [])) if isinstance(sn.get("genes", []), list) else set()
        if genes and (genes & sn_genes):
            score += 2.0
        sn_rsids = set(sn.get("rsids", [])) if isinstance(sn.get("rsids", []), list) else set()
        if rsids and (rsids & sn_rsids):
            score += 2.0
        sn_traits = set(sn.get("trait_ids", [])) if isinstance(sn.get("trait_ids", []), list) else set()
        if trait_id and trait_id in sn_traits:
            score += 3.0
        text_blob = " ".join(
            [
                str(sn.get("title", "")).lower(),
                str(sn.get("text", "")).lower(),
            ]
        )
        overlap_terms = [term for term in [trait_name, category, subcategory] + keywords if term and term in text_blob]
        if overlap_terms:
            score += min(1.5, 0.2 * len(overlap_terms))
        obj = dict(sn)
        obj["score"] = score
        scored.append(obj)

    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return scored[: max(1, int(k or 6))]


def evidence_quality(snippets: list[dict]) -> dict:
    snips = [s for s in (snippets or []) if isinstance(s, dict)]
    if not snips:
        return {
            "has_real_citations": False,
            "num_snippets": 0,
            "placeholder_only": True,
            "quality": "Low",
        }

    def is_placeholder(s: dict) -> bool:
        cid = str(s.get("citation_id", "")).strip()
        text = str(s.get("text", "")).lower()
        return cid.startswith("CURATED:") or ("placeholder" in text)

    real = [s for s in snips if not is_placeholder(s)]
    placeholder_only = len(real) == 0

    if len(real) >= 3:
        quality = "High"
    elif len(real) >= 1:
        quality = "Medium"
    else:
        quality = "Low"

    return {
        "has_real_citations": len(real) > 0,
        "num_snippets": len(snips),
        "placeholder_only": placeholder_only,
        "quality": quality,
    }
