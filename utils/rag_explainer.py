from __future__ import annotations

import re


DISCLAIMER = "This is educational, not medical advice."


def _strip_html(text: str) -> str:
    s = re.sub(r"<[^>]+>", " ", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _citation_tag(c: dict) -> str:
    if not isinstance(c, dict):
        return ""
    pmid = str(c.get("pmid", "")).strip()
    pmcid = str(c.get("pmcid", "")).strip()
    doi = str(c.get("doi", "")).strip()
    if pmid:
        return f"[PMID:{pmid}]"
    if pmcid:
        return f"[PMCID:{pmcid}]"
    if doi:
        return f"[DOI:{doi}]"
    return ""


def _safe_trait_values(trait: dict) -> dict:
    t = trait if isinstance(trait, dict) else {}
    return {
        "trait_name": str(t.get("trait_name", "This trait")).strip() or "This trait",
        "category": str(t.get("category", "")).strip() or "trait category",
        "gene": str(t.get("gene", "")).strip() or "the reported gene",
        "rsid": str(t.get("rsid", "")).strip() or "the reported variant",
        "genotype": str(t.get("user_genotype", "")).strip() or "the observed genotype",
        "bucket": str(t.get("bucket", t.get("effect_level", ""))).strip() or "a non-typical signal",
        "effect": str(t.get("effect_label", t.get("effect_level", ""))).strip() or "a trait tendency",
        "coverage": float(t.get("coverage", 0.0) or 0.0) if isinstance(t.get("coverage", 0.0), (int, float)) else 0.0,
    }


def _normalize_sources(citations: list[dict], max_items: int = 3) -> list[dict]:
    out = []
    seen = set()
    for c in citations if isinstance(citations, list) else []:
        if not isinstance(c, dict):
            continue
        pmid = str(c.get("pmid", "")).strip()
        pmcid = str(c.get("pmcid", "")).strip()
        doi = str(c.get("doi", "")).strip()
        title = _strip_html(c.get("title", "Study")) or "Study"
        year = str(c.get("year", "")).strip()
        url = str(c.get("url", c.get("source_url", ""))).strip()
        ident = pmid or pmcid or doi or title
        if ident in seen:
            continue
        seen.add(ident)
        out.append(
            {
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "title": title,
                "year": year,
                "url": url,
            }
        )
        if len(out) >= max_items:
            break
    return out


def _snippet_fragments(snippets: list[dict], max_items: int = 2) -> list[str]:
    frags = []
    for s in snippets if isinstance(snippets, list) else []:
        if not isinstance(s, dict):
            continue
        text = s.get("text", s.get("snippet_text", ""))
        cleaned = _strip_html(text)
        if cleaned:
            frags.append(" ".join(cleaned.split()[:28]))
        if len(frags) >= max_items:
            break
    return frags


def generate_trait_explanation(trait: dict, evidence_snippets: list[dict], mode: str, llm_fn=None) -> dict:
    values = _safe_trait_values(trait)
    mode_norm = str(mode or "patient").strip().lower()
    is_doctor = mode_norm.startswith("doctor")

    snippets = [s for s in (evidence_snippets or []) if isinstance(s, dict)]
    citations = _normalize_sources([s.get("citation", s) for s in snippets], max_items=3)
    tags = [_citation_tag(c) for c in citations]
    tags = [t for t in tags if t]

    if not snippets:
        if is_doctor:
            summary = (
                "We found your DNA match for this trait, but we could not retrieve enough research text right now to explain it safely. "
                "Evidence retrieval returned 0 snippets."
            )
        else:
            summary = (
                "We found your DNA match for this trait, but we could not retrieve enough research text right now to explain it safely. "
                "Please try again later."
            )
        return {
            "explanation": f"{summary} {DISCLAIMER}",
            "sources": citations,
            "status": "missing",
            "used_fallback": True,
        }

    f1, f2 = (_snippet_fragments(snippets, max_items=2) + ["", ""])[:2]
    if llm_fn is not None:
        citation_lines = []
        for i, s in enumerate(snippets[:3], start=1):
            c = citations[i - 1] if i - 1 < len(citations) else {}
            tag = _citation_tag(c) or f"[SRC:{i}]"
            snippet_txt = _strip_html(s.get("text", s.get("snippet_text", "")))
            title = _strip_html(c.get("title", "Study"))
            citation_lines.append(f"{tag} {title}: {snippet_txt}")
        prompt = (
            "You are a careful genetics educator. Use ONLY the provided snippets.\n"
            "No diagnosis, no treatment, no medication advice.\n"
            "If unsupported, say: Insufficient evidence retrieved.\n"
            f"Mode: {'doctor' if is_doctor else 'patient'}.\n"
            f"Trait: {values['trait_name']} ({values['category']}), gene {values['gene']}, rsid {values['rsid']}, genotype {values['genotype']}, signal {values['effect']}.\n"
            "Evidence snippets:\n" + "\n".join(citation_lines) + "\n"
            "Write 1-3 short sentences."
        )
        try:
            llm_text = str(llm_fn(prompt) or "").strip()
        except Exception:
            llm_text = ""
        if llm_text:
            clean = _strip_html(llm_text)
            if not is_doctor:
                clean = f"{clean} {DISCLAIMER}"
            return {
                "explanation": clean,
                "sources": citations,
                "status": "found",
                "used_fallback": False,
            }

    if is_doctor:
        c1 = tags[0] if tags else ""
        c2 = tags[1] if len(tags) > 1 else c1
        lines = [
            f"Based on {values['gene']} {values['rsid']} genotype {values['genotype']}, the current signal is {values['effect']} ({values['bucket']}). {c1}".strip(),
            f"Coverage for this trait is {values['coverage'] * 100:.0f}%, so interpretation remains probabilistic rather than diagnostic. {c2}".strip(),
        ]
        if f1:
            lines.append(f"Evidence summary: {f1}. {c1}".strip())
        explanation = " ".join(lines[:5]).strip()
    else:
        lines = [
            f"Based on {values['gene']} {values['rsid']} genotype {values['genotype']}, this suggests {values['effect']}.",
            f"This may show up as differences in {values['category']} patterns from person to person.",
        ]
        if f1:
            lines.append(f"Research snippets suggest: {f1}.")
        explanation = " ".join(lines[:3]).strip()
        explanation = f"{explanation} {DISCLAIMER}"

    explanation = _strip_html(explanation)
    return {
        "explanation": explanation,
        "sources": citations,
        "status": "found",
        "used_fallback": False,
    }
