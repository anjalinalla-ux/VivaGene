#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from study_pack_loader import load_all_trait_packs

EVIDENCE_DIR = Path("evidence_corpus")
INDEX_PATH = EVIDENCE_DIR / "index.jsonl"
SEARCH_LOG_PATH = EVIDENCE_DIR / "search_log.jsonl"
EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_jsonl_read(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def _existing_trait_counts(rows):
    counts = {}
    seen = set()
    for r in rows:
        tid = str(r.get("trait_id", "")).strip()
        if not tid:
            continue
        ident = (
            str(r.get("pmid", "")).strip()
            or str(r.get("pmcid", "")).strip()
            or str(r.get("doi", "")).strip()
            or str(r.get("title", "")).strip()
        )
        key = f"{tid}|{ident}"
        if key in seen:
            continue
        seen.add(key)
        counts[tid] = counts.get(tid, 0) + 1
    return counts, seen


def _pick_trait_keyphrase(trait: dict) -> str:
    keys = []
    for k in trait.get("keywords", []) if isinstance(trait.get("keywords", []), list) else []:
        s = str(k).strip()
        if s:
            keys.append(s)
    return keys[0] if keys else str(trait.get("subcategory", "")).strip()


def _build_queries(trait: dict):
    trait_name = str(trait.get("trait_name", "")).strip()
    key_phrase = _pick_trait_keyphrase(trait)
    gene = ""
    rsid = ""
    for v in trait.get("variants", []) if isinstance(trait.get("variants", []), list) else []:
        if not isinstance(v, dict):
            continue
        if not gene:
            gene = str(v.get("gene", "")).strip()
        if not rsid:
            rsid = str(v.get("rsid", "")).strip()

    queries = []
    if gene and rsid:
        queries.append(f'("{gene}" AND {rsid})')
    if gene and (trait_name or key_phrase):
        phrase = trait_name or key_phrase
        queries.append(f'("{gene}" AND ("{phrase}"))')
    if rsid:
        queries.append(rsid)
    if gene:
        queries.append(f'"{gene}"')

    deduped = []
    for q in queries:
        if q and q not in deduped:
            deduped.append(q)
    return deduped, gene, rsid


def _fetch_europe_pmc(query: str, page_size: int = 10):
    params = {
        "query": query,
        "format": "json",
        "pageSize": int(page_size),
        "resultType": "core",
    }
    url = EUROPE_PMC_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "VivaGene/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    results = (((data or {}).get("resultList") or {}).get("result") or [])
    return results if isinstance(results, list) else [], url


def _to_snippet(result: dict, trait_id: str, gene: str, rsid: str, query: str):
    pmid = str(result.get("pmid", "")).strip() or None
    pmcid = str(result.get("pmcid", "")).strip() or None
    doi = str(result.get("doi", "")).strip() or None
    title = str(result.get("title", "")).strip()
    journal = str(result.get("journalTitle", "")).strip() or None
    year_raw = str(result.get("pubYear", "")).strip()
    try:
        year = int(year_raw) if year_raw else None
    except Exception:
        year = None
    snippet = str(result.get("abstractText", "")).strip() or title
    snippet = snippet[:1200]
    source = str(result.get("source", "")).strip() or "EuropePMC"
    ident = pmid or pmcid or doi
    if pmid:
        url = f"https://europepmc.org/article/MED/{pmid}"
    elif pmcid:
        url = f"https://europepmc.org/article/PMC/{pmcid}"
    elif doi:
        url = f"https://doi.org/{doi}"
    else:
        url = "https://europepmc.org/search"

    return {
        "trait_id": trait_id,
        "gene": gene or "",
        "rsid": rsid or "",
        "query": query,
        "source": source,
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "year": year,
        "snippet": snippet,
        "url": url,
        "retrieved_at": _now_iso(),
        "_dedupe_ident": ident or title,
    }


def _iter_target_traits(trait_id: str | None = None):
    packs = load_all_trait_packs(base_dir="trait_study_packs")
    for t in packs:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("trait_id", "")).strip()
        if not tid:
            continue
        if trait_id and tid != trait_id:
            continue
        yield t


def build_evidence_corpus(max_traits: int | None = None, trait_id: str | None = None):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.touch(exist_ok=True)
    SEARCH_LOG_PATH.touch(exist_ok=True)

    existing_rows = _safe_jsonl_read(INDEX_PATH)
    trait_counts, dedupe_seen = _existing_trait_counts(existing_rows)

    processed = 0
    created = 0
    query_attempts = 0
    query_hits = 0

    for trait in _iter_target_traits(trait_id=trait_id):
        if max_traits is not None and processed >= max(0, int(max_traits)):
            break
        tid = str(trait.get("trait_id", "")).strip()
        if not tid:
            continue
        if trait_counts.get(tid, 0) >= 3:
            processed += 1
            continue

        queries, gene, rsid = _build_queries(trait)
        if not queries:
            queries = [str(trait.get("trait_name", tid)).strip()]

        local_added = 0
        for query in queries:
            if trait_counts.get(tid, 0) >= 3:
                break
            query_attempts += 1
            error = None
            hits = 0
            request_url = ""
            results = []
            try:
                results, request_url = _fetch_europe_pmc(query=query, page_size=10)
                hits = len(results)
                query_hits += hits
            except Exception as exc:
                error = str(exc)

            _append_jsonl(
                SEARCH_LOG_PATH,
                {
                    "trait_id": tid,
                    "gene": gene or "",
                    "rsid": rsid or "",
                    "query": query,
                    "request_url": request_url,
                    "hits": int(hits),
                    "retrieved_at": _now_iso(),
                    "error": error,
                },
            )

            if error:
                time.sleep(random.uniform(0.2, 0.5))
                continue

            for result in results:
                if trait_counts.get(tid, 0) >= 3:
                    break
                if not isinstance(result, dict):
                    continue
                row = _to_snippet(result, tid, gene, rsid, query)
                key = f"{tid}|{row.get('_dedupe_ident', '')}"
                if key in dedupe_seen:
                    continue
                dedupe_seen.add(key)
                row.pop("_dedupe_ident", None)
                _append_jsonl(INDEX_PATH, row)
                trait_counts[tid] = trait_counts.get(tid, 0) + 1
                created += 1
                local_added += 1
                if local_added >= 3:
                    break

            time.sleep(random.uniform(0.2, 0.5))

        processed += 1

    return {
        "processed_traits": processed,
        "snippets_created": created,
        "query_attempts": query_attempts,
        "query_hits": query_hits,
        "index_path": str(INDEX_PATH),
        "search_log_path": str(SEARCH_LOG_PATH),
    }


def main():
    parser = argparse.ArgumentParser(description="Build/refresh local evidence corpus from Europe PMC.")
    parser.add_argument("--max_traits", type=int, default=None, help="Maximum number of traits to process this run.")
    parser.add_argument("--trait_id", type=str, default=None, help="Optional single trait_id to refresh.")
    args = parser.parse_args()

    summary = build_evidence_corpus(max_traits=args.max_traits, trait_id=args.trait_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
