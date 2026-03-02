from __future__ import annotations

import re


SAFE_NEXT_STEPS = [
    "If concerns remain, discuss these findings with a clinician or genetic counselor.",
    "Consider tracking relevant experiences and sharing them with a qualified professional.",
    "Learn more from trusted sources such as NIH, MedlinePlus, CDC, or NSGC.",
]


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


def _first_sentence(text: str) -> str:
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", raw)
    return parts[0].strip() if parts else raw


def _strip_html_and_sections(text: str) -> str:
    raw = str(text or "")
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\b(Background|Objectives|Methods|Results|Conclusion|Conclusions)\b\s*[:\-]?", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _clean_mode_text(text: str) -> str:
    t = _strip_html_and_sections(text)
    # Remove repeated citation tags
    t = re.sub(r"(\[PMID:[^\]]+\]){2,}", r"\1", t)
    return t.strip()


def build_explanation(trait, evidence, mode) -> dict:
    t = trait if isinstance(trait, dict) else {}
    e = evidence if isinstance(evidence, dict) else {}
    citations = e.get("citations", []) if isinstance(e.get("citations", []), list) else []
    snippets = e.get("snippets", []) if isinstance(e.get("snippets", []), list) else []
    queries = e.get("queries", []) if isinstance(e.get("queries", []), list) else []
    mode_text = str(mode or "Patient (simple)")
    is_doctor = "Doctor" in mode_text

    if len(citations) < 2 or len(snippets) < 2:
        return {
            "summary": "Evidence pending — no explanatory claims shown yet (RAG safety).",
            "life_impact": "",
            "patient_summary": "Evidence pending — no explanatory claims shown yet (RAG safety).",
            "doctor_summary": "Evidence pending — no explanatory claims shown yet (RAG safety).",
            "next_steps": SAFE_NEXT_STEPS[:2],
            "citations": citations,
            "claim_map": [],
            "queries_used": queries,
        }

    tag1 = _citation_tag(citations[0]) if citations else ""
    tag2 = _citation_tag(citations[1]) if len(citations) > 1 else tag1
    effect = str(t.get("effect_label", "")).strip() or str(t.get("effect_level", "")).strip() or "a non-typical signal"
    category = str(t.get("category", "")).strip().lower()
    gene = str(t.get("gene", "")).strip() or "the reported gene"
    rsid = str(t.get("rsid", "")).strip() or "the reported rsID"
    genotype = str(t.get("user_genotype", "")).strip() or "the observed genotype"
    bucket = str(t.get("effect_level", "")).strip() or str(t.get("bucket", "")).strip() or "a non-typical signal"

    if is_doctor:
        summary = f"Based on {gene} {rsid} genotype {genotype}, this profile suggests {effect} ({bucket}) in association studies, with non-deterministic effect size. {tag1}".strip()
        life = f"This may correspond to measurable variation in {category or 'this domain'} at the group level, though individual expression and generalizability can vary. {tag2}".strip()
        mech = f"Current evidence supports association-level interpretation rather than causal diagnosis. {tag2}".strip()
        summary = f"{summary} {mech}".strip()
    else:
        summary = f"Based on {gene} {rsid} genotype {genotype}, this suggests {effect}. {tag1}".strip()
        life = f"In daily life, this may show up as differences in {category or 'related patterns'}, but responses can vary from person to person. {tag2}".strip()

    summary = _clean_mode_text(summary)
    life = _clean_mode_text(life)

    claim_map = []
    if tag1:
        claim_map.append({"claim": summary, "pmids": [tag1.replace("[", "").replace("]", "")]})
    if life and tag2:
        claim_map.append({"claim": life, "pmids": [tag2.replace("[", "").replace("]", "")]})

    patient_summary = _clean_mode_text(
        f"Based on {gene} {rsid} genotype {genotype}, this suggests {effect}. {tag1}".strip()
    )
    doctor_summary = _clean_mode_text(
        f"Based on {gene} {rsid} genotype {genotype}, this profile suggests {effect} ({bucket}) with probabilistic, non-diagnostic interpretation. {tag1}".strip()
    )

    return {
        "summary": summary,
        "life_impact": life,
        "patient_summary": patient_summary,
        "doctor_summary": doctor_summary,
        "next_steps": SAFE_NEXT_STEPS,
        "citations": citations[:3],
        "claim_map": claim_map,
        "queries_used": queries,
    }
