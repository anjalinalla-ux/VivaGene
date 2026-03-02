from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


EVIDENCE_DIR = Path("evidence_corpus")

CATEGORY_KEYWORDS = {
    "Neurobehavior": ["stress", "anxiety", "attention", "dopamine", "serotonin", "circadian", "sleep"],
    "Nutrition": ["lipid", "glucose", "folate", "vitamin D", "caffeine", "lactose", "omega-3"],
    "Fitness": ["VO2", "endurance", "strength", "recovery", "injury", "inflammation"],
    "Liver": ["NAFLD", "ALT", "AST", "hepatic steatosis"],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(trait_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(trait_id or "").strip())
    return s or "unknown_trait"


def get_cached_evidence(trait_id) -> dict | None:
    path = EVIDENCE_DIR / f"{_safe_name(trait_id)}.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def save_cached_evidence(trait_id, evidence_dict) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / f"{_safe_name(trait_id)}.json"
    payload = evidence_dict if isinstance(evidence_dict, dict) else {}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def europe_pmc_search(query: str, max_results: int = 8) -> list[dict]:
    q = str(query or "").strip()
    if not q:
        return []
    params = {
        "query": q,
        "format": "json",
        "resultType": "core",
        "pageSize": max(1, int(max_results or 8)),
    }
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "VivaGene/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    rows = (((data or {}).get("resultList") or {}).get("result") or [])
    return rows if isinstance(rows, list) else []


def extract_snippets(paper_json) -> list[str]:
    def clean_text(v: str) -> str:
        s = re.sub(r"<[^>]+>", " ", str(v or ""))
        s = re.sub(r"\b(Background|Objectives|Methods|Results|Conclusion|Conclusions)\b\s*[:\-]?", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def clip_words(v: str, n: int = 40) -> str:
        words = str(v or "").split()
        return " ".join(words[:n]).strip()

    p = paper_json if isinstance(paper_json, dict) else {}
    abstract = clean_text(p.get("abstractText", ""))
    title = clean_text(p.get("title", ""))
    if abstract:
        sentences = re.split(r"(?<=[.!?])\s+", abstract)
        out = [clip_words(s.strip(), 40) for s in sentences if s.strip()][:2]
        return out if out else [clip_words(abstract, 40)]
    if title:
        return [clip_words(title, 40)]
    return []


def normalize_citation(paper) -> dict:
    p = paper if isinstance(paper, dict) else {}
    pmid = str(p.get("pmid", "")).strip()
    pmcid = str(p.get("pmcid", "")).strip()
    doi = str(p.get("doi", "")).strip()
    title = str(p.get("title", "")).strip() or "Untitled"
    year = str(p.get("pubYear", "")).strip()
    journal = str(p.get("journalTitle", "")).strip()
    if pmid:
        url = f"https://europepmc.org/article/MED/{pmid}"
    elif pmcid:
        url = f"https://europepmc.org/article/PMC/{pmcid}"
    elif doi:
        url = f"https://doi.org/{doi}"
    else:
        url = "https://europepmc.org/"
    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "year": year,
        "journal": journal,
        "url": url,
    }


def _keyword_for_category(category: str) -> str:
    cat = str(category or "").strip()
    words = CATEGORY_KEYWORDS.get(cat, [])
    return words[0] if words else "genetic association"


def build_trait_queries(trait: dict) -> list[str]:
    t = trait if isinstance(trait, dict) else {}
    gene = str(t.get("gene", "")).strip()
    rsid = str(t.get("rsid", "")).strip()
    trait_name = str(t.get("trait_name", t.get("title", ""))).strip()
    keyword = _keyword_for_category(t.get("category", ""))
    queries = []
    if gene and rsid:
        queries.append(f"{gene} {rsid} {trait_name}".strip())
    if gene:
        queries.append(f"{gene} polymorphism {keyword}".strip())
    if rsid:
        queries.append(f"{rsid} association {keyword}".strip())
    dedup = []
    for q in queries:
        if q and q not in dedup:
            dedup.append(q)
    return dedup


def ensure_trait_evidence(trait: dict, min_citations: int = 2, min_snippets: int = 2, max_results: int = 8) -> dict:
    t = trait if isinstance(trait, dict) else {}
    trait_id = str(t.get("trait_id", "")).strip() or "unknown_trait"
    cached = get_cached_evidence(trait_id)
    if isinstance(cached, dict):
        cits = cached.get("citations", []) if isinstance(cached.get("citations", []), list) else []
        snips = cached.get("snippets", []) if isinstance(cached.get("snippets", []), list) else []
        if len(cits) >= min_citations and len(snips) >= min_snippets:
            return cached

    citations = []
    snippets = []
    seen_ident = set()
    queries = build_trait_queries(t)

    for q in queries:
        try:
            papers = europe_pmc_search(q, max_results=max_results)
        except Exception:
            papers = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            cit = normalize_citation(p)
            ident = cit.get("pmid") or cit.get("pmcid") or cit.get("doi") or cit.get("title")
            if ident in seen_ident:
                continue
            seen_ident.add(ident)
            citations.append(cit)
            for s in extract_snippets(p):
                snippets.append({"text": s, "citation": cit})
            if len(citations) >= max(2, min_citations) and len(snippets) >= max(2, min_snippets):
                break
        if len(citations) >= max(2, min_citations) and len(snippets) >= max(2, min_snippets):
            break
        time.sleep(0.2)

    payload = {
        "trait_id": trait_id,
        "queries": queries,
        "citations": citations,
        "snippets": snippets,
        "retrieved_at": _now_iso(),
    }
    save_cached_evidence(trait_id, payload)
    return payload
