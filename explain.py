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
            "next_steps": SAFE_NEXT_STEPS[:2],
            "citations": citations,
            "claim_map": [],
            "queries_used": queries,
        }

    tag1 = _citation_tag(citations[0]) if citations else ""
    tag2 = _citation_tag(citations[1]) if len(citations) > 1 else tag1
    effect = str(t.get("effect_label", "")).strip() or str(t.get("effect_level", "")).strip() or "a non-typical signal"
    category = str(t.get("category", "")).strip().lower()
    s1 = _first_sentence((snippets[0] or {}).get("text", ""))
    s2 = _first_sentence((snippets[1] or {}).get("text", "")) if len(snippets) > 1 else ""

    if is_doctor:
        summary = (
            f"This trait shows {effect}, with supporting association signals in retrieved literature; findings should be interpreted as probabilistic and context-dependent. {tag1}"
        ).strip()
        life = (
            f"Retrieved evidence suggests this signal may correspond to observable variation in {category or 'this domain'}, but effect size and external validity can vary. {tag2}"
        ).strip()
        if "[PMID:" not in summary and "INSUFFICIENT_EVIDENCE" not in summary:
            summary = "Evidence pending — no explanatory claims shown yet (RAG safety)."
            life = ""
    else:
        summary = (
            f"Research linked to this trait suggests {effect} may appear in some people, but this is not deterministic. {tag1}"
        ).strip()
        life = (
            f"In daily life, this may show up as differences in {category or 'related patterns'} from one person to another. {tag2}"
        ).strip()

    if s1:
        summary = f"{summary} {_first_sentence(s1)} {tag1}".strip()
    if s2 and life:
        life = f"{life} {_first_sentence(s2)} {tag2}".strip()

    claim_map = []
    if tag1:
        claim_map.append({"claim": summary, "pmids": [tag1.replace("[", "").replace("]", "")]})
    if life and tag2:
        claim_map.append({"claim": life, "pmids": [tag2.replace("[", "").replace("]", "")]})

    return {
        "summary": summary,
        "life_impact": life,
        "next_steps": SAFE_NEXT_STEPS,
        "citations": citations[:3],
        "claim_map": claim_map,
        "queries_used": queries,
    }
