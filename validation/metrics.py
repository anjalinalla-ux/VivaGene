from __future__ import annotations

import re
from typing import Iterable


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(t) >= 3}


def extract_atomic_claims(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "")) if s.strip()]
    keep = []
    verbs = (" is ", " suggests", " associated", " linked", " may ", " might ")
    for s in sentences:
        low = f" {s.lower()} "
        if any(v in low for v in verbs):
            keep.append(s)
    return keep


def claim_supported_by_evidence(claim: str, retrieved_snippets: Iterable[dict]) -> bool:
    c_toks = _tokens(claim)
    if not c_toks:
        return False

    for sn in retrieved_snippets:
        if not isinstance(sn, dict):
            continue
        text = f"{sn.get('title','')} {sn.get('text','')} {' '.join(sn.get('genes',[]))} {' '.join(sn.get('trait_ids',[]))}"
        s_toks = _tokens(text)
        if not s_toks:
            continue

        inter = len(c_toks & s_toks)
        union = len(c_toks | s_toks)
        jacc = (inter / union) if union else 0.0
        if jacc >= 0.12:
            return True

        genes = [g.lower() for g in (sn.get("genes", []) if isinstance(sn.get("genes", []), list) else [])]
        entity_hit = any(g in claim.lower() for g in genes if g)
        if entity_hit and inter >= 2:
            return True

    return False


def hallucination_rate(output_text: str, evidence_snippets: list[dict]) -> dict:
    claims = extract_atomic_claims(output_text)
    if not claims:
        return {"num_claims": 0, "num_supported": 0, "num_unsupported": 0, "unsupported_rate": 0.0}

    supported = sum(1 for c in claims if claim_supported_by_evidence(c, evidence_snippets))
    unsupported = len(claims) - supported
    return {
        "num_claims": len(claims),
        "num_supported": supported,
        "num_unsupported": unsupported,
        "unsupported_rate": (unsupported / len(claims)) if claims else 0.0,
    }
