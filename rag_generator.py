from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import streamlit as st
except Exception:
    st = None


def _get_secret(name: str, default: str = "") -> str:
    if st is not None:
        try:
            value = st.secrets.get(name, os.getenv(name, default))
            return str(value or "").strip()
        except Exception:
            pass
    return str(os.getenv(name, default) or "").strip()


OPENROUTER_API_KEY = _get_secret("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = _get_secret("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_CLIENT = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    if (OpenAI is not None and OPENROUTER_API_KEY)
    else None
)


def format_evidence_block(snippets: list[dict]) -> str:
    lines = []
    for i, s in enumerate([x for x in (snippets or []) if isinstance(x, dict)], start=1):
        title = str(s.get("title", "")).strip() or "Untitled"
        year = str(s.get("year", "")).strip() or "n.d."
        cid = str(s.get("citation_id", "")).strip() or "UNKNOWN"
        text = str(s.get("text", "")).strip().replace("\n", " ")
        text = text[:400]
        lines.append(f"[{i}] ({cid}) {title} ({year}): {text}")
    return "\n".join(lines)


def _call_openrouter_json(system: str, user: str) -> str:
    if not OPENROUTER_CLIENT:
        return ""
    try:
        resp = OPENROUTER_CLIENT.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _safe_json_parse(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def enforce_citations(output_json: dict, snippets: list[dict]) -> dict:
    out = dict(output_json) if isinstance(output_json, dict) else {}
    out.setdefault("trait_id", "")
    out.setdefault("explanation_1", "")
    out.setdefault("explanation_2", "")
    out.setdefault("citations_used", [])
    out.setdefault("unsupported", False)
    out.setdefault("refusal_reason", "")

    s1 = str(out.get("explanation_1", "")).strip()
    s2 = str(out.get("explanation_2", "")).strip()
    text_sentences = [s for s in [s1, s2] if s]

    if out.get("unsupported"):
        return out

    if not text_sentences:
        out["unsupported"] = True
        out["refusal_reason"] = "Citation validation failed"
        out["citations_used"] = []
        return out

    max_idx = len([s for s in (snippets or []) if isinstance(s, dict)])
    used_ids = []

    for s in text_sentences:
        refs = re.findall(r"\[(\d+)\]", s)
        if not refs:
            out["unsupported"] = True
            out["refusal_reason"] = "Citation validation failed"
            out["citations_used"] = []
            return out
        for r in refs:
            idx = int(r)
            if idx < 1 or idx > max_idx:
                out["unsupported"] = True
                out["refusal_reason"] = "Citation validation failed"
                out["citations_used"] = []
                return out
            cid = str(snippets[idx - 1].get("citation_id", "")).strip()
            if cid and cid not in used_ids:
                used_ids.append(cid)

    out["citations_used"] = used_ids
    out["unsupported"] = False
    out["refusal_reason"] = ""
    return out


def generate_trait_explanation_rag(trait_summary: dict, snippets: list[dict], quality: dict) -> dict:
    trait_id = str((trait_summary or {}).get("trait_id", "")).strip()

    quality_label = str((quality or {}).get("quality", "Low")).strip()
    if quality_label == "Low":
        return {
            "trait_id": trait_id,
            "explanation_1": "Insufficient evidence in local corpus.",
            "explanation_2": "",
            "citations_used": [],
            "unsupported": True,
            "refusal_reason": "Insufficient evidence in local corpus",
        }

    evidence_block = format_evidence_block(snippets)
    system = (
        "You are an evidence-bound scientific explainer. You may ONLY use the provided snippets. "
        "If a claim is not supported by snippets, do not include it. Produce JSON only."
    )
    user = f"""
Trait summary JSON:
{json.dumps(trait_summary, indent=2)}

Evidence snippets:
{evidence_block}

Return JSON only with keys:
{{
  "trait_id": "...",
  "explanation_1": "One sentence with citations like [1][2].",
  "explanation_2": "Optional second sentence with citations.",
  "citations_used": [],
  "unsupported": false,
  "refusal_reason": ""
}}

Rules:
- 1-2 sentences max, general-population friendly.
- No lifestyle advice.
- No medical diagnosis/treatment claims.
- Every sentence must include at least one [n] citation marker.
"""

    raw = _call_openrouter_json(system, user)
    parsed = _safe_json_parse(raw)
    if not parsed:
        parsed = {
            "trait_id": trait_id,
            "explanation_1": "Insufficient evidence in local corpus.",
            "explanation_2": "",
            "citations_used": [],
            "unsupported": True,
            "refusal_reason": "Citation validation failed",
        }
    parsed.setdefault("trait_id", trait_id)
    validated = enforce_citations(parsed, snippets)
    return validated
