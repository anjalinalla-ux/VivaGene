import json
from urllib.parse import quote_plus
from urllib.request import urlopen


EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _safe_str(value):
    return (value or "") if isinstance(value, str) else str(value or "")


def search_europe_pmc(query: str, page_size: int = 10) -> list[dict]:
    """Search Europe PMC and return normalized paper dicts."""
    q = _safe_str(query).strip()
    if not q:
        return []

    size = max(1, min(int(page_size or 10), 100))
    url = (
        f"{EUROPE_PMC_SEARCH_URL}?query={quote_plus(q)}"
        f"&format=json&pageSize={size}&resultType=core"
    )

    try:
        with urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    results = payload.get("resultList", {}).get("result", [])
    if not isinstance(results, list):
        return []

    out = []
    for r in results:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "title": _safe_str(r.get("title")),
                "authorString": _safe_str(r.get("authorString")),
                "journalTitle": _safe_str(r.get("journalTitle")),
                "pubYear": _safe_str(r.get("pubYear")),
                "abstractText": _safe_str(r.get("abstractText")),
                "id": _safe_str(r.get("id")),
                "source": _safe_str(r.get("source")),
                "doi": _safe_str(r.get("doi")),
                "pmid": _safe_str(r.get("pmid")),
                "pmcid": _safe_str(r.get("pmcid")),
            }
        )
    return out


def build_query(gene: str, trait_name: str, keywords: list[str]) -> str:
    """Build a simple robust Europe PMC query."""
    g = _safe_str(gene).strip()
    trait = _safe_str(trait_name).strip()
    kw = [k.strip() for k in (keywords or []) if _safe_str(k).strip()]

    trait_terms = [trait] + kw
    trait_clause = " OR ".join(f'"{t}"' if " " in t else t for t in trait_terms if t)
    if not trait_clause:
        trait_clause = "trait"

    gene_clause = g if g else "gene"
    return f"({gene_clause}) AND ({trait_clause}) AND (human OR GWAS OR meta-analysis)"


def retrieve_evidence_for_trait(trait: dict, max_papers: int = 6) -> list[dict]:
    """Retrieve and deduplicate papers for a trait context."""
    t = trait if isinstance(trait, dict) else {}
    gene = _safe_str(t.get("gene")).strip()
    trait_name = _safe_str(t.get("trait_name") or t.get("title")).strip()
    keywords = []
    if isinstance(t.get("keywords"), list):
        keywords = t.get("keywords")
    elif isinstance(t.get("mechanism_keywords"), list):
        keywords = t.get("mechanism_keywords")

    query = build_query(gene=gene, trait_name=trait_name, keywords=keywords)
    papers_raw = search_europe_pmc(query, page_size=max(10, int(max_papers or 6)))

    papers = []
    seen = set()
    for p in papers_raw[: max(1, int(max_papers or 6))]:
        pmid = _safe_str(p.get("pmid"))
        pmcid = _safe_str(p.get("pmcid"))
        doi = _safe_str(p.get("doi"))
        dedupe_key = pmid or pmcid or doi or _safe_str(p.get("title"))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        papers.append(
            {
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "title": _safe_str(p.get("title")),
                "year": _safe_str(p.get("pubYear")),
                "abstractText": _safe_str(p.get("abstractText")),
                "journalTitle": _safe_str(p.get("journalTitle")),
                "authorString": _safe_str(p.get("authorString")),
            }
        )
    return papers


# Backward-compatible wrappers used by older app code.
def build_gene_query(gene: str, trait_name: str, keywords: list[str]) -> str:
    return build_query(gene=gene, trait_name=trait_name, keywords=keywords)


def get_evidence_packets(trait: dict, max_papers: int = 6) -> dict:
    t = trait if isinstance(trait, dict) else {}
    return {
        "gene": _safe_str(t.get("gene")),
        "trait_name": _safe_str(t.get("trait_name") or t.get("title")),
        "papers": [
            {
                "pmid": p.get("pmid", ""),
                "pmcid": p.get("pmcid", ""),
                "doi": p.get("doi", ""),
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "abstract": p.get("abstractText", ""),
            }
            for p in retrieve_evidence_for_trait(t, max_papers=max_papers)
        ],
    }
