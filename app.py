import streamlit as st
import csv
import json
import re
import traceback
from pathlib import Path
from genomics_interpreter import (
    TRAIT_DB_PATH,
    TRAITS_JSON_PATH,
    load_traits_catalog,
    build_prs_report_from_upload,
    generate_text_report,
    generate_html_report,
)
import os
from utils.trait_quality import compute_trait_completeness, trait_has_variants
from rag_retrieval import load_study_packs, retrieve_evidence
from polygenic import compute_prs
from evidence_guard import enforce_evidence_only
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
BASE_DIR = Path(__file__).resolve().parent
STUDY_PACK_DIR = BASE_DIR / "trait_study_packs"

def normalize_report(report_obj):
    """Ensure downstream renderers receive a dict with .get().

    Some helper functions may return a list of trait dicts, or a dict whose
    `traits` field contains non-dict entries. This normalizes to:
    {"summary": {...}, "traits": [<dict>, ...]}
    """

    # Case 1: report_obj is already a list of traits
    if isinstance(report_obj, list):
        traits = [t for t in report_obj if isinstance(t, dict)]
        categories = sorted({t.get("category", "") for t in traits if t.get("category")})
        return {
            "summary": {
                "num_traits_found": len(traits),
                "categories": categories,
            },
            "traits": traits,
        }

    # Case 2: report_obj is a dict-like report
    if isinstance(report_obj, dict):
        report_obj.setdefault("summary", {})
        raw_traits = report_obj.get("traits", [])

        # Force traits to be a list of dicts only
        traits = [t for t in raw_traits if isinstance(t, dict)] if isinstance(raw_traits, list) else []
        report_obj["traits"] = traits

        # Build / backfill summary fields
        if not isinstance(report_obj["summary"], dict):
            report_obj["summary"] = {}

        report_obj["summary"].setdefault("num_traits_found", len(traits))
        report_obj["summary"].setdefault(
            "categories",
            sorted({t.get("category", "") for t in traits if t.get("category")}),
        )

        return report_obj

    # Fallback
    return {"summary": {"num_traits_found": 0, "categories": []}, "traits": []}


FLAG_ORDER = {"High": 0, "Medium": 1, "Low": 2}
EVIDENCE_ORDER = {"High": 0, "Moderate": 1, "Emerging": 2}


def sort_traits_for_display(traits):
    items = [t for t in (traits or []) if isinstance(t, dict)]

    def evidence_rank(value):
        v = str(value or "").strip()
        if not v:
            return 99
        v_norm = v.capitalize()
        if v.lower() == "strong":
            v_norm = "High"
        return EVIDENCE_ORDER.get(v_norm, 99)

    def key_fn(t):
        flag = str(t.get("flag_level", "")).strip().capitalize()
        return (
            FLAG_ORDER.get(flag, 99),
            evidence_rank(t.get("evidence_strength")),
            str(t.get("trait_name", "")).strip().lower(),
        )

    return sorted(items, key=key_fn)


def build_filtered_report(report_obj, selected_traits):
    report = normalize_report(dict(report_obj) if isinstance(report_obj, dict) else {})
    selected = [t for t in (selected_traits or []) if isinstance(t, dict)]
    selected = sort_traits_for_display(selected)
    categories = sorted({t.get("track", t.get("category", "")) for t in selected if t.get("track", t.get("category", ""))})
    report["traits"] = selected
    report["summary"] = report.get("summary", {})
    report["summary"]["num_traits_found"] = len(selected)
    report["summary"]["categories"] = categories
    report["summary"]["total_matched_ready"] = len(selected)
    report["summary"]["major_matched_count"] = sum(1 for t in selected if str(t.get("priority", "")).strip().lower() == "major")
    report["summary"]["standard_matched_count"] = sum(1 for t in selected if str(t.get("priority", "")).strip().lower() != "major")
    report["traits_major"] = [t for t in selected if str(t.get("priority", "")).strip().lower() == "major"]
    report["traits_standard"] = [t for t in selected if str(t.get("priority", "")).strip().lower() != "major"]
    rag_all = report_obj.get("rag_evidence", {}) if isinstance(report_obj, dict) else {}
    if isinstance(rag_all, dict):
        report["rag_evidence"] = {t.get("trait_id", ""): rag_all.get(t.get("trait_id", ""), []) for t in selected if t.get("trait_id", "")}
    else:
        report["rag_evidence"] = {}
    return report


def is_dev_mode():
    env_flag = str(os.environ.get("DEV_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}
    secret_flag = False
    try:
        secret_flag = bool(st.secrets.get("DEV_MODE", False))
    except Exception:
        secret_flag = False
    return env_flag or secret_flag


@st.cache_data(show_spinner=False)
def cached_load_study_packs(dir_path: str):
    return load_study_packs(dir_path)


def get_evidence_packets_cached(trait: dict, max_papers: int = 6):
    t = trait if isinstance(trait, dict) else {}
    trait_id = str(t.get("trait_id", "")).strip()
    gene = str(t.get("gene", "")).strip()
    trait_name = str(t.get("trait_name", t.get("title", ""))).strip()
    packs = cached_load_study_packs(str(STUDY_PACK_DIR))
    local = retrieve_evidence(trait_id=trait_id, study_packs=packs, k=max(1, int(max_papers or 6)))
    papers = []
    for p in local.get("passages", []) or []:
        if not isinstance(p, dict):
            continue
        citation_id = str(p.get("citation_id", "") or "")
        pmid = citation_id.replace("PMID:", "") if citation_id.startswith("PMID:") else ""
        pmcid = citation_id.replace("PMC:", "") if citation_id.startswith("PMC:") else ""
        doi = citation_id.replace("DOI:", "") if citation_id.startswith("DOI:") else ""
        papers.append(
            {
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "title": str(p.get("title", "") or ""),
                "year": str(p.get("year", "") or ""),
                "abstract": str(p.get("quote", "") or ""),
            }
        )
    return {"gene": gene, "trait_name": trait_name, "papers": papers}


def attach_rag_evidence_to_report(report: dict, max_papers_per_gene: int = 6):
    rep = normalize_report(dict(report) if isinstance(report, dict) else {})
    rag_evidence = {}
    major_traits = [t for t in rep.get("traits_major", []) if isinstance(t, dict)]
    for trait in major_traits:
        q = trait.get("_quality", {})
        if not isinstance(q, dict) or q.get("label") != "Ready":
            continue
        trait_id = str(trait.get("trait_id", "")).strip()
        if not trait_id:
            continue

        genes = set()
        variants = trait.get("variants")
        if isinstance(variants, list):
            for v in variants:
                if not isinstance(v, dict):
                    continue
                g = str(v.get("gene", "")).strip()
                if g:
                    genes.add(g)
        g_single = str(trait.get("gene", "")).strip()
        if g_single:
            genes.add(g_single)

        packets = []
        for gene in sorted(genes):
            trait_for_query = dict(trait)
            trait_for_query["gene"] = gene
            packet = get_evidence_packets_cached(trait_for_query, max_papers=max_papers_per_gene)
            packets.append(packet)
        rag_evidence[trait_id] = packets

    rep["rag_evidence"] = rag_evidence
    return rep


def attach_polygenic_evidence_scaffold(report: dict, user_variants: list[dict]):
    """Attach local study-pack PRS + local evidence + trust metadata."""
    rep = normalize_report(dict(report) if isinstance(report, dict) else {})
    rep.setdefault("polygenic", {})
    rep.setdefault("evidence", {})
    rep.setdefault("trust", {"coverage": 0.0, "confidence": "Low", "citations": []})

    packs = cached_load_study_packs(str(STUDY_PACK_DIR))
    if not packs:
        rep["trust"] = {
            "coverage": 0.0,
            "confidence": "Low",
            "citations": [],
            "notes": "Study packs are missing; polygenic and local RAG scaffold not applied.",
        }
        return rep, "Local study packs not found. Continuing with single-SNP report only."

    citations = []
    seen_citations = set()
    coverages = []

    for pack in packs:
        if not isinstance(pack, dict):
            continue
        trait_id = str(pack.get("trait_id", "")).strip()
        if not trait_id:
            continue

        prs = compute_prs(pack, user_variants or [])
        rep["polygenic"][trait_id] = prs
        coverages.append(float(prs.get("coverage", 0.0) or 0.0))

        ev = retrieve_evidence(trait_id=trait_id, study_packs=packs, k=3)
        rep["evidence"][trait_id] = ev
        for p in ev.get("passages", []) or []:
            if not isinstance(p, dict):
                continue
            cid = str(p.get("citation_id", "")).strip()
            if not cid or cid in seen_citations:
                continue
            seen_citations.add(cid)
            citations.append(cid)

    avg_coverage = (sum(coverages) / len(coverages)) if coverages else 0.0
    if avg_coverage >= 0.6:
        confidence = "High"
    elif avg_coverage >= 0.2:
        confidence = "Medium"
    else:
        confidence = "Low"

    rep["trust"] = {
        "coverage": round(avg_coverage, 4),
        "confidence": confidence,
        "citations": citations[:20],
    }
    return rep, ""

# Expect the key to be set as OPENROUTER_API_KEY
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
OPENROUTER_CLIENT = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    if (OpenAI is not None and OPENROUTER_API_KEY)
    else None
)

# Show errors in the UI instead of failing silently
st.set_option("client.showErrorDetails", True)

def generate_ai_summary(report: dict) -> str:
    if not OPENROUTER_CLIENT:
        return {"overview": "", "top_insights": [], "evidence": {}}

    report = normalize_report(report)
    traits = [t for t in report.get("traits", []) if isinstance(t, dict) and str(t.get("priority", "")).strip().lower() == "major"]
    rag_evidence = report.get("rag_evidence", {}) if isinstance(report.get("rag_evidence"), dict) else {}

    prompt = f"""
You are a careful genetics educator.

Rules:
- Educational only
- No medical advice
- No diagnosis
- No disease risk prediction
- Use cautious language (may, might, could)
- Return strict JSON only, no markdown, no extra text.
- Use Major traits only.
- Top insights must be 4-7 items when possible.
- confidence must be one of: High, Medium, Low.
- If no papers for a trait, include a limitations note saying:
  "No direct paper match was retrieved"

Return EXACTLY this JSON shape:
{{
  "overview": "3-5 short paragraphs, friendly, educational",
  "top_insights": [
    {{
      "trait_id":"...",
      "title":"...",
      "explanation":"...",
      "try_this":[ "...", "..." ],
      "confidence":"Medium"
    }}
  ],
  "evidence": {{
    "TRAIT_ID": {{
      "citations":[ {{"pmid":"...", "title":"...", "year":2022}} ],
      "limitations":"..."
    }}
  }}
}}

Major matched trait objects (JSON):
{json.dumps(traits, indent=2)}

Evidence packets by trait_id (JSON):
{json.dumps(rag_evidence, indent=2)}
"""

    raw = generate_with_fallback_model(prompt)
    return parse_ai_summary_json(raw)


def parse_ai_summary_json(text: str):
    def empty_payload():
        return {"overview": "", "top_insights": [], "evidence": {}}

    raw = (text or "").strip()
    if not raw:
        return empty_payload()

    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return empty_payload()

    overview = obj.get("overview", "")
    if not isinstance(overview, str):
        overview = str(overview or "")
    top_insights = obj.get("top_insights", [])
    evidence = obj.get("evidence", {})
    if not isinstance(top_insights, list):
        top_insights = []
    if not isinstance(evidence, dict):
        evidence = {}

    normalized_insights = []
    for i in top_insights:
        if not isinstance(i, dict):
            continue
        trait_id = str(i.get("trait_id", "")).strip()
        title = str(i.get("title", "")).strip()
        explanation = str(i.get("explanation", "")).strip()
        try_this = i.get("try_this", [])
        if not isinstance(try_this, list):
            try_this = []
        try_this = [str(x).strip() for x in try_this if str(x).strip()][:3]
        confidence = str(i.get("confidence", "Medium")).strip().capitalize()
        if confidence not in {"High", "Medium", "Low"}:
            confidence = "Medium"
        normalized_insights.append(
            {
                "trait_id": trait_id,
                "title": title,
                "explanation": explanation,
                "try_this": try_this,
                "confidence": confidence,
            }
        )

    normalized_evidence = {}
    for tid, block in evidence.items():
        if not isinstance(block, dict):
            continue
        citations = block.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        normalized_citations = []
        for c in citations:
            if not isinstance(c, dict):
                continue
            normalized_citations.append(
                {
                    "pmid": str(c.get("pmid", "")).strip(),
                    "pmcid": str(c.get("pmcid", "")).strip(),
                    "doi": str(c.get("doi", "")).strip(),
                    "title": str(c.get("title", "")).strip(),
                    "year": str(c.get("year", "")).strip(),
                }
            )
        normalized_evidence[str(tid)] = {
            "citations": normalized_citations,
            "limitations": str(block.get("limitations", "")).strip(),
        }

    return {"overview": overview.strip(), "top_insights": normalized_insights, "evidence": normalized_evidence}


def insights_to_chat_summary(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    insights = payload.get("top_insights", [])
    if not isinstance(insights, list) or not insights:
        return ""
    lines = []
    for i in insights:
        if not isinstance(i, dict):
            continue
        title = str(i.get("title", "")).strip()
        exp = str(i.get("explanation", "")).strip()
        if title or exp:
            lines.append(f"- {title}: {exp}".strip(": "))
    return "\n".join(lines)


def generate_with_fallback_model(prompt: str) -> str:
    """Generate text via OpenRouter using an OpenAI-compatible client."""
    if not OPENROUTER_CLIENT:
        return ""
    response = OPENROUTER_CLIENT.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful genetics educator. Educational content only. "
                    "Do not provide medical advice, diagnosis, or treatment recommendations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return (response.choices[0].message.content or "").strip()


def is_red_flag_trait(trait: dict) -> bool:
    if not isinstance(trait, dict):
        return False
    flag_level = str(trait.get("flag_level", "")).strip().lower()
    priority = str(trait.get("priority", "")).strip().lower()
    effect_level = str(trait.get("effect_level", "")).strip().lower()
    return (
        flag_level == "high"
        or priority == "major"
        or effect_level in {"elevated", "high"}
    )


def parse_trait_annotation_json(text: str):
    payload = {"life_impact": "", "red_flag": "", "citations": []}
    raw = (text or "").strip()
    if not raw:
        return payload
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return payload
    payload["life_impact"] = str(obj.get("life_impact", "")).strip()
    payload["red_flag"] = str(obj.get("red_flag", "")).strip()
    citations = obj.get("citations", [])
    if isinstance(citations, list):
        out = []
        for c in citations[:3]:
            if not isinstance(c, dict):
                continue
            out.append(
                {
                    "pmid": str(c.get("pmid", "")).strip(),
                    "pmcid": str(c.get("pmcid", "")).strip(),
                    "doi": str(c.get("doi", "")).strip(),
                    "title": str(c.get("title", "")).strip(),
                    "year": str(c.get("year", "")).strip(),
                }
            )
        payload["citations"] = out
    return payload


def extract_citations_from_packets(evidence_packets, max_items: int = 3):
    citations = []
    seen = set()
    for packet in evidence_packets if isinstance(evidence_packets, list) else []:
        papers = packet.get("papers", []) if isinstance(packet, dict) else []
        for p in papers:
            if not isinstance(p, dict):
                continue
            pmid = str(p.get("pmid", "")).strip()
            pmcid = str(p.get("pmcid", "")).strip()
            doi = str(p.get("doi", "")).strip()
            key = pmid or pmcid or doi or str(p.get("title", "")).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "doi": doi,
                    "title": str(p.get("title", "")).strip(),
                    "year": str(p.get("year", "")).strip(),
                }
            )
            if len(citations) >= max_items:
                return citations
    return citations


def generate_trait_annotation(trait: dict, evidence_packets):
    t = trait if isinstance(trait, dict) else {}
    citations = extract_citations_from_packets(evidence_packets, max_items=3)
    default_life = (
        f"{t.get('trait_name', 'This trait')} may be associated with variation in response patterns, "
        "but observed effects can vary across people and contexts."
    )
    default_red = ""
    if is_red_flag_trait(t):
        default_red = (
            "This is marked as a red-flag trait because the current signal may indicate a stronger-than-average effect."
        )
    if not citations and "Evidence retrieval pending." not in default_life:
        default_life += " Evidence retrieval pending."

    if not OPENROUTER_CLIENT:
        return {"life_impact": default_life, "red_flag": default_red, "citations": citations}

    evidence_snippets = []
    for c in citations:
        ident = c.get("pmid") or c.get("pmcid") or c.get("doi") or ""
        evidence_snippets.append(
            {
                "id": ident,
                "title": c.get("title", ""),
                "year": c.get("year", ""),
            }
        )

    prompt = f"""
Trait:
{json.dumps(t, indent=2)}

Evidence snippets:
{json.dumps(evidence_snippets, indent=2)}

Return JSON ONLY with:
{{
 "life_impact": "1–2 sentences: how this might show up in everyday life (plain language).",
 "red_flag": "0–1 sentence ONLY if this trait is flagged High/Major/Elevated; otherwise empty string.",
 "citations": [
   {{"pmid":"", "pmcid":"", "doi":"", "title":"", "year":""}}
 ]
}}

Rules:
- No bullet lists.
- No advice or what to do.
- Educational only. No medical advice. No diagnosis. No treatment. No lifestyle fixes.
- Use cautious language (may/might/can vary).
- Do NOT restate the provided explanation verbatim; rewrite clearly for a general audience.
- If no evidence, citations = [] and include: "Evidence retrieval pending."
"""
    raw = generate_with_fallback_model(prompt)
    parsed = parse_trait_annotation_json(raw)
    if not parsed.get("life_impact"):
        parsed["life_impact"] = default_life
    if is_red_flag_trait(t) and not parsed.get("red_flag"):
        parsed["red_flag"] = default_red
    if not is_red_flag_trait(t):
        parsed["red_flag"] = ""
    parsed_citations = parsed.get("citations", [])
    if not isinstance(parsed_citations, list) or not parsed_citations:
        parsed["citations"] = citations
    else:
        parsed["citations"] = parsed_citations[:3]
    if not parsed["citations"] and "Evidence retrieval pending." not in parsed["life_impact"]:
        parsed["life_impact"] = f"{parsed['life_impact']} Evidence retrieval pending."
    return parsed


def build_local_lifestyle_plan(report: dict) -> str:
    report = normalize_report(report)
    categories = report.get("summary", {}).get("categories", [])
    cat_line = ", ".join(categories) if categories else "your matched traits"
    return (
        "Big picture: Your profile suggests useful lifestyle experiments around "
        f"{cat_line}. This is educational only, not medical advice.\n\n"
        "Sleep: Keep a consistent sleep-wake window, reduce late caffeine, and track how rested you feel.\n\n"
        "Focus and learning: Use short deep-work blocks (25-45 min), then brief breaks, and note which timing works best.\n\n"
        "Movement and recovery: Pair moderate activity with regular recovery days; adjust intensity based on sleep and stress.\n\n"
        "Caffeine and stimulants: Start with smaller doses earlier in the day and avoid late-afternoon intake.\n\n"
        "Everyday habits: Change one variable at a time for 2-3 weeks and keep simple notes so you can spot patterns."
    )


def build_local_chat_reply(user_input: str, report: dict) -> str:
    report = normalize_report(report)
    n = report.get("summary", {}).get("num_traits_found", 0)
    categories = report.get("summary", {}).get("categories", [])
    cats = ", ".join(categories) if categories else "the available traits"
    return (
        f"I could not reach an AI provider right now, so here is a local guidance response. "
        f"Your report includes {n} matched traits across {cats}. "
        f"For your question \"{user_input}\", try one small, trackable habit change this week, "
        f"monitor sleep/energy/focus daily, and adjust gradually. This is educational only, not medical advice."
    )

# ---------- Page config ----------
st.set_page_config(
    page_title="VivaGene",
    page_icon="🧬",
    layout="wide",
)

# ---------- Global styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg-dark: #020617;
        --bg-main: #020617;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-gold: #fbbf24;
        --text-muted: #9ca3af;
        --text-main: #e5e7eb;
    }
    body {
        background: var(--bg-main);
        color: var(--text-main);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg-main);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .main .block-container {
        padding-top: 0;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* TOP NAV */

    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 0 10px;
        margin-bottom: 6px;
    }

    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* G shaped DNA style mark */

    .logo-mark {
        width: 32px;
        height: 32px;
        border-radius: 999px;
        background: #020617;
        border: 2px solid #3b82f6;
        position: relative;
        box-shadow: 0 6px 14px rgba(15,23,42,0.45);
    }

    .logo-g-arc {
        position: absolute;
        inset: 4px;
        border-radius: 999px;
        border: 2px solid #3b82f6;
        border-right-color: transparent;
    }

    .logo-g-cut {
        position: absolute;
        right: 4px;
        top: 50%;
        width: 8px;
        height: 2px;
        background: #020617;
        transform: translateY(-50%);
    }

    .logo-helix-line {
        position: absolute;
        top: 6px;
        bottom: 6px;
        left: 50%;
        width: 2px;
        border-radius: 999px;
        transform: translateX(-7px);
        background: linear-gradient(180deg, #3b82f6, #22c55e);
    }

    .logo-helix-line:nth-child(2) {
        transform: translateX(5px);
        background: linear-gradient(180deg, #8b5cf6, #fbbf24);
    }

    .logo-rung {
        position: absolute;
        left: 50%;
        width: 14px;
        height: 2px;
        background: #e5e7eb;
        border-radius: 999px;
        transform: translateX(-50%);
    }

    .logo-rung:nth-child(1) { top: 9px; }
    .logo-rung:nth-child(2) { top: 16px; }
    .logo-rung:nth-child(3) { top: 23px; }

    .logo-text-main {
        font-weight: 700;
        letter-spacing: 0.12em;
        font-size: 0.86rem;
        text-transform: uppercase;
    }

    .logo-text-sub {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: -2px;
    }

    /* Nav buttons (top right) */

    .nav-container {
        display: flex;
        justify-content: flex-end;
        gap: 18px;
        padding-top: 4px;
    }

    .stButton>button {
        background: transparent;
        border: none;
        color: #e5e7eb; /* light text on dark background */
        padding: 2px 0;
        font-size: 0.9rem;
        font-weight: 500;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        transition: color 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
    }

    .stButton>button:hover {
        color: #ffffff; /* pure white on hover */
        border-bottom-color: #38bdf8;
        transform: translateY(-1px);
        cursor: pointer;
    }

    .nav-active > button {
        border-bottom-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* HERO BAND */

    .hero-band {
        background: radial-gradient(circle at 0% 0%, #0f172a, #020617 48%);
        color: #e5e7eb;
        border-radius: 0 0 40px 40px;
        padding: 40px 56px 64px;
        margin: 0 -3rem 40px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.55);
    }

    @media (max-width: 800px) {
        .hero-band {
            margin: 0 -1rem 32px;
            padding: 28px 24px 44px;
        }
    }

    .hero-grid {
        display: flex;
        gap: 56px;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }

    .hero-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.5);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 12px;
        color: #e5e7eb;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }

    .hero-title span {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
        -webkit-background-clip: text;
        color: transparent;
    }

    .hero-sub {
        font-size: 0.98rem;
        max-width: 560px;
        color: #cbd5f5;
        margin-bottom: 1.0rem;
    }

    .hero-note {
        font-size: 0.8rem;
        color: #9ca3af;
        max-width: 540px;
    }

    /* DNA CARD (G-Helix inspired) */

    .hero-dna-card {
        min-width: 260px;
        max-width: 320px;
        border-radius: 24px;
        background:
            radial-gradient(circle at 10% 0%, #1f2937, transparent 55%),
            radial-gradient(circle at 90% 100%, rgba(251,191,36,0.25), transparent 50%),
            #020617;
        border: 1px solid rgba(148, 163, 184, 0.45);
        padding: 18px 18px 20px;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .hero-dna-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 30px rgba(15,23,42,0.7);
        border-color: rgba(251,191,36,0.8);
    }

    .hero-dna-title {
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .hero-dna-sub {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 12px;
    }

    .dna-helix {
        position: relative;
        height: 180px;
        margin-top: 4px;
    }

    .dna-strand {
        position: absolute;
        width: 2px;
        height: 100%;
        border-radius: 999px;
        left: 50%;
        transform: translateX(-26px);
        background: linear-gradient(180deg, #3b82f6, #f97316);
    }

    .dna-strand:nth-child(2) {
        transform: translateX(26px);
        background: linear-gradient(180deg, #8b5cf6, #22c55e);
    }

    .dna-rung {
        position: absolute;
        width: 64px;
        height: 2px;
        background: rgba(148, 163, 184, 0.7);
        left: 50%;
        transform: translateX(-50%);
        border-radius: 999px;
    }

    .dna-rung:nth-child(3) { top: 12%; }
    .dna-rung:nth-child(4) { top: 28%; }
    .dna-rung:nth-child(5) { top: 44%; }
    .dna-rung:nth-child(6) { top: 60%; }
    .dna-rung:nth-child(7) { top: 76%; }

    /* GENERAL SECTIONS */

    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        color: var(--text-main);
    }

    .section-sub {
        font-size: 0.94rem;
        color: #6b7280;
        margin-bottom: 1.0rem;
    }

    .home-section {
        margin-top: 0.5rem;
        margin-bottom: 2.5rem;
    }

    .newsletter-modal {
        border-radius: 16px;
        padding: 14px 16px 12px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        box-shadow: 0 20px 40px rgba(15,23,42,0.12);
        color: var(--text-main);
    }

    .newsletter-title {
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    .newsletter-sub {
        font-size: 0.86rem;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }

    .small-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    /* FEATURE / INFO CARDS */

    .feature-row {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .feature-card {
        flex: 1 1 220px;
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        padding: 14px 16px 16px;
        box-shadow: 0 8px 18px rgba(148, 163, 184, 0.18);
        color: var(--text-main);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(148,163,184,0.35);
        border-color: #cbd5f5;
    }

/* FIX HOW-SECTION TITLES */
.how-title,
.how-label,
.how-title *,
.how-label * {
    color: #111827 !important;
}

/* FIX NEWSLETTER SECTION TEXT */
.newsletter-title,
.newsletter-title *,
.newsletter-sub,
.newsletter-sub * {
    color: #111827 !important;
}

    .feature-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }

    .feature-title {
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: var(--text-main);
    }

    .feature-body {
        font-size: 0.86rem;
        color: #4b5563;
    }

/* FORCE DARK TEXT INSIDE LIGHT CARDS */
.feature-card,
.feature-card * {
    color: #111827 !important;
}

.how-step,
.how-step * {
    color: #111827 !important;
}

.bio-card,
.bio-card * {
    color: #111827 !important;
}

.who-card,
.who-card * {
    color: #111827 !important;
}
    /* HOW IT WORKS STRIP */

    .how-band {
        margin-top: 2.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #f9fafb, #eef2ff);
        border: 1px solid #e0e7ff;
        padding: 16px 18px 18px;
        color: var(--text-main);
    }

    .how-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b7280;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }

    .how-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .how-row {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
    }

    .how-step {
        flex: 1 1 190px;
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7ff;
        padding: 10px 12px 12px;
        font-size: 0.84rem;
        color: #4b5563;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .how-step:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 22px rgba(148,163,184,0.35);
        border-color: #c7d2fe;
    }

    .how-step-number {
        font-size: 0.75rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.1rem;
    }

    .how-step-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: var(--text-main);
    }

    /* FOUNDER + WHO IT'S FOR + ROADMAP */

    .bio-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        padding: 14px 16px 16px;
        font-size: 0.9rem;
        box-shadow: 0 8px 18px rgba(148, 163, 184, 0.18);
        color: var(--text-main);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .bio-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(148,163,184,0.35);
        border-color: #cbd5f5;
    }

    .bio-name {
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: var(--text-main);
    }

    .bio-role {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }

    .who-row {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }

    .who-card {
        flex: 1 1 220px;
        background: #f9fafb;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        padding: 10px 12px 12px;
        font-size: 0.86rem;
        color: #4b5563;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .who-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 22px rgba(148,163,184,0.35);
        border-color: #d1d5db;
    }

    .who-title {
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: var(--text-main);
    }

    .roadmap-card {
        background: #020617;
        color: #e5e7eb;
        border-radius: 18px;
        padding: 14px 16px 16px;
        border: 1px solid rgba(148,163,184,0.5);
        font-size: 0.86rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .roadmap-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 32px rgba(15,23,42,0.8);
        border-color: rgba(59,130,246,0.7);
    }

    .roadmap-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9ca3af;
        margin-bottom: 0.2rem;
    }

    .roadmap-title {
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    /* FINAL FIX: Ensure How-Title is dark */
    .how-title {
        color: #111827 !important;
    }
/* Global site footer */
.site-footer {
    margin-top: 40px;
    padding: 18px 0 10px;
    border-top: 1px solid rgba(148, 163, 184, 0.4);
    font-size: 0.8rem;
    color: #9ca3af;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    opacity: 0.9;
}
.site-footer a {
    color: #9ca3af;
    text-decoration: none;
}
.site-footer a:hover {
    text-decoration: underline;
    color: #e5e7eb;
}

/* Smooth section fade-in */
.hero,
.home-section,
.section-light,
.how-band,
.chat-box,
.feature-card {
    animation: fadeUp 0.35s ease-out;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Button hover transitions */
.stButton > button {
    transition: background-color 0.18s ease, color 0.18s ease,
                transform 0.18s ease, box-shadow 0.18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.35);
}

/* Card hover transitions */
.feature-card,
.how-step,
.trait-card,
.chat-box {
    transition: transform 0.18s ease-out, box-shadow 0.18s ease-out,
                border-color 0.18s ease-out, background-color 0.18s ease-out;
}

.feature-card:hover,
.how-step:hover,
.trait-card:hover,
.chat-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.4);
    border-color: rgba(148, 163, 184, 0.7);
}

/* Navbar hover refinement */
.nav-btn,
.nav-btn-active {
    transition: color 0.16s ease, border-bottom-color 0.16s ease,
                transform 0.16s ease;
}

.nav-btn:hover,
.nav-btn-active:hover {
    transform: translateY(-1px);
}

/* ---------- Microanimations + Page Transitions ---------- */

/* Page fade / slide on navigation (Streamlit rerun, but looks like a transition) */
.page-wrap {
  animation: pageFade 240ms ease-out;
}

@keyframes pageFade {
  from { opacity: 0; transform: translateY(6px); filter: blur(1px); }
  to   { opacity: 1; transform: translateY(0); filter: blur(0); }
}

/* Premium button microinteractions */
.stButton > button {
  transition: transform 180ms ease, box-shadow 180ms ease, background-color 180ms ease, color 180ms ease;
  will-change: transform;
}

.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 26px rgba(15, 23, 42, 0.45);
}

.stButton > button:active {
  transform: translateY(0px) scale(0.98);
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.35);
}

/* Card + panel hover polish */
.feature-card, .how-step, .trait-card, .chat-box, .section-light {
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
  will-change: transform;
}

.feature-card:hover, .how-step:hover, .trait-card:hover, .chat-box:hover, .section-light:hover {
  transform: translateY(-3px);
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.45);
  border-color: rgba(148, 163, 184, 0.75);
}

/* Navbar microinteraction (bolder + subtle lift) */
.nav-btn {
  font-weight: 600 !important;
  letter-spacing: 0.2px;
  transition: color 160ms ease, border-bottom-color 160ms ease, transform 160ms ease;
}

.nav-btn:hover {
  transform: translateY(-1px);
}

html {
  scroll-behavior: smooth;
}

/* NAV BUTTON CONTRAST FIX: make top-nav buttons readable on dark background */
.nav-button-container .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 2px 0 !important;
}

.nav-button-container .stButton > button:hover {
    color: #ffffff !important;
    border-bottom-color: #38bdf8 !important;
    transform: translateY(-1px);
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Nav state ----------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

# Page transition key (increments on navigation change)
if "page_transition_key" not in st.session_state:
    st.session_state.page_transition_key = 0


def set_page(page_name: str):
    st.session_state.active_page = page_name
    st.session_state.page_transition_key += 1

# ---------- TOP NAV BAR ----------
with st.container():
    col_left, col_right = st.columns([1.4, 1.6])

    with col_left:
        st.markdown(
            """
            <style>
            .nav-logo-container {
                display: flex;
                align-items: center;
                gap: 14px;
            }
            .logo-title {
                font-weight: 700;
                letter-spacing: 0.12em;
                font-size: 1.05rem;
                text-transform: uppercase;
            }
            .logo-sub {
                font-size: 0.75rem;
                color: #9ca3af;
                margin-top: -4px;
            }
            .nav-button-container {
                display: flex;
                justify-content: flex-end;
                gap: 20px;
                padding-top: 8px;
            }
            .nav-btn {
    background: transparent !important;
    border: none !important;
    font-size: 0.95rem !important;
    color: #e5e7eb !important;              /* light text on dark nav */
    border-bottom: 2px solid transparent !important;
    padding-bottom: 4px !important;
}

.nav-btn:hover {
    color: #ffffff !important;              /* bright white on hover */
    border-bottom-color: #38bdf8 !important; /* cyan underline */
    transform: translateY(-1px);
}

.nav-btn-active {
    background: transparent !important;
    color: #ffffff !important;              /* active tab always bright */
    border-bottom: 2px solid #3b82f6 !important; /* blue underline */
    padding-bottom: 4px !important;
transition: color 0.16s ease, border-bottom-color 0.16s ease,
                transform 0.16s ease;
}

/* G Helix rotating DNA mark */
.g-helix-orbit {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    position: relative;
    border: 1px solid rgba(148, 163, 184, 0.6);
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: helix-rotate 4.5s linear infinite;
    background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.3), transparent 55%);
}

.g-helix-strand {
    position: absolute;
    width: 2px;
    height: 80%;
    border-radius: 999px;
    background: linear-gradient(to bottom, #38bdf8, #a855f7);
    opacity: 0.95;
}

.g-helix-strand-left {
    transform: rotate(18deg) translateX(-7px);
}

.g-helix-strand-right {
    transform: rotate(-18deg) translateX(7px);
}

.g-helix-rung {
    position: absolute;
    width: 26px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(to right, #38bdf8, #a855f7);
    opacity: 0.9;
}

.g-helix-rung.rung-1 { top: 18%; }
.g-helix-rung.rung-2 { top: 35%; }
.g-helix-rung.rung-3 { top: 55%; }
.g-helix-rung.rung-4 { top: 72%; }

@keyframes helix-rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.image("genai_logo.png", width=58)

        st.markdown(
            """
            <div class="nav-logo-container">
                <div>
                    <div class="logo-title">VIVAGENE</div>
                    <div class="logo-sub">Genomics, explained with evidence.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("<div class='nav-button-container'>", unsafe_allow_html=True)
        pages = ["Home", "Upload & Report", "Lifestyle Q&A", "Trait Explorer", "Trait Science", "About", "Contact"]
        labels = ["Home", "Upload", "Lifestyle Q&A", "Trait explorer", "Trait science", "About", "Contact"]
        if str(os.environ.get("SHOW_VALIDATION", "0")).strip() == "1":
            pages.append("Validation (Dev)")
            labels.append("Validation (Dev)")
        nav_cols = st.columns(len(labels))

        for i, p in enumerate(pages):
            if nav_cols[i].button(labels[i], key=f"nav_btn_{i}"):
                set_page(p)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

page = st.session_state.active_page

# Wrapper to trigger CSS animation on page change
st.markdown(
    f"<div class='page-wrap' data-k='{st.session_state.page_transition_key}'>",
    unsafe_allow_html=True,
)
              
           
# ---------- Newsletter state ----------
if "hide_newsletter" not in st.session_state:
    st.session_state.hide_newsletter = False


def newsletter_block(location_text: str):
    if st.session_state.hide_newsletter:
        return

    with st.container():
        st.markdown(
            f"<div class='small-label'>Stay in the loop</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='newsletter-modal'>"
            "<div class='newsletter-title'>Join the VivaGene early access list</div>"
            "<div class='newsletter-sub'>Get updates as the platform evolves and, in a full launch, "
            "receive your reports securely by email.</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns([3, 1])
        with cols[0]:
            email = st.text_input("Email address", key=f"newsletter_{location_text}")
        with cols[1]:
            st.write("")
            if st.button("Notify me", key=f"notify_{location_text}"):
                if email.strip():
                    st.success("Thanks for subscribing. In a real deployment this would save to a mailing list.")
                    st.session_state.hide_newsletter = True
                else:
                    st.warning("Please enter an email.")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Trait DB helper ----------
def load_trait_rows(path_csv: str, path_json: str):
    """Load trait rows for explorer, preferring nested JSON schema."""
    rows = []

    # Preferred: data/traits.json (nested -> flatten for table/search/filter).
    try:
        with open(path_json, encoding="utf-8") as f:
            traits = json.load(f)
        if isinstance(traits, list):
            for trait in traits:
                if not isinstance(trait, dict):
                    continue
                quality = trait.get("_quality")
                if not isinstance(quality, dict):
                    quality = compute_trait_completeness(trait)
                quality_label = quality.get("label", "Placeholder")
                completeness_score = int(quality.get("score", 0))
                missing_fields_count = len(quality.get("missing", [])) if isinstance(quality.get("missing"), list) else 0
                priority = trait.get("priority", "Standard")
                flag_level = trait.get("flag_level", "Low")
                user_visibility_default = bool(trait.get("user_visibility_default", False))
                trait_name = trait.get("trait_name", trait.get("title", ""))
                variants = trait.get("variants", []) or []
                if not variants:
                    rows.append(
                        {
                            "trait_id": trait.get("trait_id", ""),
                            "title": trait_name,
                            "trait_name": trait_name,
                            "track": trait.get("track", trait.get("category", "")),
                            "subcategory": trait.get("subcategory", "General"),
                            "rsid": "",
                            "gene": "",
                            "genotype": "",
                            "effect_label": "",
                            "effect_level": "",
                            "explanation": "",
                            "evidence_strength": "",
                            "priority": priority,
                            "flag_level": flag_level,
                            "user_visibility_default": user_visibility_default,
                            "quality_label": quality_label,
                            "completeness_score": completeness_score,
                            "missing_fields_count": missing_fields_count,
                        }
                    )
                    continue
                for var in variants:
                    if not isinstance(var, dict):
                        continue
                    genotypes = var.get("genotype_effects", {}) or var.get("genotypes", {}) or {}
                    if not genotypes:
                        rows.append(
                            {
                                "trait_id": trait.get("trait_id", ""),
                                "title": trait_name,
                                "trait_name": trait_name,
                                "track": trait.get("track", trait.get("category", "")),
                                "subcategory": trait.get("subcategory", "General"),
                                "rsid": var.get("rsid", ""),
                                "gene": var.get("gene", ""),
                                "genotype": "",
                                "effect_label": "",
                                "effect_level": "",
                                "explanation": "",
                                "evidence_strength": var.get("evidence_strength", ""),
                                "priority": priority,
                                "flag_level": flag_level,
                                "user_visibility_default": user_visibility_default,
                                "quality_label": quality_label,
                                "completeness_score": completeness_score,
                                "missing_fields_count": missing_fields_count,
                            }
                        )
                        continue
                    for gt, gx in genotypes.items():
                        if not isinstance(gx, dict):
                            continue
                        rows.append(
                            {
                                "trait_id": trait.get("trait_id", ""),
                                "title": trait_name,
                                "trait_name": trait_name,
                                "track": trait.get("track", trait.get("category", "")),
                                "subcategory": trait.get("subcategory", "General"),
                                "rsid": var.get("rsid", ""),
                                "gene": var.get("gene", ""),
                                "genotype": str(gt).upper(),
                                "effect_label": gx.get("effect_label", ""),
                                "effect_level": gx.get("effect_level", ""),
                                "explanation": gx.get("explanation", ""),
                                "evidence_strength": var.get("evidence_strength", ""),
                                "priority": priority,
                                "flag_level": flag_level,
                                "user_visibility_default": user_visibility_default,
                                "quality_label": quality_label,
                                "completeness_score": completeness_score,
                                "missing_fields_count": missing_fields_count,
                            }
                        )
            if rows:
                return rows
    except FileNotFoundError:
        pass
    except Exception as e:
        st.error(f"Could not load trait database from {path_json}: {e}")

    # Fallback: legacy CSV
    try:
        with open(path_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row.setdefault("title", row.get("trait_name", ""))
                row.setdefault("track", row.get("category", ""))
                row.setdefault("subcategory", "General")
                row.setdefault("priority", "Standard")
                row.setdefault("flag_level", "Low")
                row.setdefault("user_visibility_default", False)
                quality = compute_trait_completeness(
                    {
                        "trait_name": row.get("title", ""),
                        "user_question": "",
                        "what_it_means": "",
                        "why_it_matters": "",
                        "limitations": "",
                        "tags": [],
                        "variants": [row] if row.get("rsid") else [],
                        "research_query_hint": "",
                        "citation_seeds": [],
                        "mechanism_keywords": [],
                    }
                )
                row["quality_label"] = quality.get("label", "Placeholder")
                row["completeness_score"] = int(quality.get("score", 0))
                row["missing_fields_count"] = len(quality.get("missing", [])) if isinstance(quality.get("missing"), list) else 0
                rows.append(row)
    except Exception as e:
        st.error(f"Could not load trait database from {path_csv}: {e}")
    return rows


def get_trait_quality_counts(path_json: str):
    """Return total and label counts using precomputed trait _quality metadata."""
    counts = {"total": 0, "Ready": 0, "Draft": 0, "Placeholder": 0, "no_variants": 0}
    try:
        traits = load_traits_catalog(path_json)
        counts["total"] = len(traits)
        for t in traits:
            q = t.get("_quality", {})
            label = q.get("label", "Placeholder")
            if label in counts:
                counts[label] += 1
            if not trait_has_variants(t):
                counts["no_variants"] += 1
    except Exception:
        pass
    return counts


def get_trait_db_stats(path_json: str):
    stats = {"traits": 0, "variants": 0, "ready": 0, "draft": 0, "placeholder": 0}
    try:
        traits = load_traits_catalog(path_json)
        stats["traits"] = len(traits)
        for t in traits:
            if not isinstance(t, dict):
                continue
            variants = t.get("variants", [])
            if isinstance(variants, list):
                stats["variants"] += len([v for v in variants if isinstance(v, dict)])
            q = t.get("_quality", {})
            label = q.get("label", "Placeholder") if isinstance(q, dict) else "Placeholder"
            if label == "Ready":
                stats["ready"] += 1
            elif label == "Draft":
                stats["draft"] += 1
            else:
                stats["placeholder"] += 1
    except Exception:
        pass
    return stats

# ---------- HOME ----------
if page == "Home":
    st.markdown(
        """
        <div class="hero-band">
          <div class="hero-grid">
            <div>
              <div class="hero-chip">Evidence-first Genomics</div>
              <div class="hero-title">
                Understand your genetics with <span>clarity.</span>
              </div>
              <div class="hero-sub">
                VivaGene translates raw genotype files into structured trait signals and plain-language explanations grounded in citations.
                It is designed for careful educational use and clear communication of confidence and coverage.
              </div>
              <div class="hero-note">
                VivaGene currently supports a curated trait panel across Nutrition, Neurobehavior, and Fitness.
                Every explanation is linked to evidence and shown with confidence + coverage. Genes are not destiny.
              </div>
            </div>
            <div class="hero-dna-card">
              <div class="hero-dna-title">Evidence-first Genomics</div>
              <div class="hero-dna-sub">
                A lightweight variant interpreter feeds a language model that explains patterns
                in clear language while keeping the underlying science visible.
              </div>
              <div class="dna-helix">
                <div class="dna-strand"></div>
                <div class="dna-strand"></div>
                <div class="dna-rung"></div>
                <div class="dna-rung"></div>
                <div class="dna-rung"></div>
                <div class="dna-rung"></div>
                <div class="dna-rung"></div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary CTAs under hero
    cta_col1, cta_col2 = st.columns([0.4, 0.4])
    with cta_col1:
        if st.button("Generate your report", key="hero_start_report"):
            st.session_state.active_page = "Upload & Report"
    with cta_col2:
        if st.button("Try Lifestyle Q&A", key="hero_chatbot"):
            st.session_state.active_page = "Lifestyle Q&A"

    # Why VivaGene exists
    st.markdown('<div class="home-section dark-section">', unsafe_allow_html=True)
    col_left, col_right = st.columns([1.4, 1.0])

    with col_left:
        st.markdown('<div class="section-title">Why VivaGene exists</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - Turn raw SNP data into something a person can actually read.  
            - Help people see how certain traits may connect to sleep, nutrition, focus, or training response.  
            - Give students and future clinicians a safe way to practice genomic thinking without making medical claims.  
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        newsletter_block("home")
    st.markdown("</div>", unsafe_allow_html=True)

    # Why choose VivaGene
    st.markdown('<div class="home-section dark-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why choose VivaGene?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A focused, lifestyle aware, education first platform instead of a generic consumer report.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-row">
          <div class="feature-card">
            <div class="feature-label">Founder story</div>
            <div class="feature-title">Designed by a future AI geneticist</div>
            <div class="feature-body">
              VivaGene is led by Anjali, a high school student who plans to work at the intersection of genetics,
              AI, and clinical decision making. The project reflects real experience in rare disease research and
              clinical volunteering, translated into a tool that explains DNA in a calm and structured way.
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-label">Transparent</div>
            <div class="feature-title">Every trait is traceable</div>
            <div class="feature-body">
              Under the hood is a curated trait table. Each interpretation links to specific rsIDs and genes,
              rather than opaque risk scores, so you can always see the structure behind the narrative.
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-label">Lifestyle focus</div>
            <div class="feature-title">DNA as one piece of the puzzle</div>
            <div class="feature-body">
              The goal is not to label you. It is to highlight patterns that might interact with sleep, nutrition,
              attention, or training, always framed as possibilities and ideas, never as prescriptions.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="home-section">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="how-band">
          <div class="how-label">How it works</div>
          <div class="how-title">From raw file to a lifestyle aware report.</div>
          <div class="how-row">
            <div class="how-step">
              <div class="how-step-number">01</div>
              <div class="how-step-title">Upload a raw DNA file</div>
              <div>Use a 23andMe style text export or start with the supported file format to see the flow.</div>
            </div>
            <div class="how-step">
              <div class="how-step-number">02</div>
              <div class="how-step-title">Match variants to traits</div>
              <div>A genetics engine scans rsIDs and matches them to a curated panel related to traits such as caffeine response, sleep, or recovery.</div>
            </div>
            <div class="how-step">
              <div class="how-step-number">03</div>
              <div class="how-step-title">Generate an evidence overview</div>
              <div>The language model converts structured traits into a human-readable summary while keeping evidence boundaries explicit.</div>
            </div>
            <div class="how-step">
              <div class="how-step-number">04</div>
              <div class="how-step-title">Review the full report</div>
              <div>Explore a printable HTML report with trait cards, explanations, and caveats that you can revisit over time.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="home-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How VivaGene stays trustworthy</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - Coverage shown (% variants found)  
        - Confidence shown (High/Medium/Low)  
        - Citations displayed  
        - Refuses unsupported claims  
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Secondary CTA near How it works
    spacer_left, cta_mid, spacer_right = st.columns([0.35, 0.3, 0.35])
    with cta_mid:
        if st.button("Start your report", key="how_start_report"):
            st.session_state.active_page = "Upload & Report"

    # Founder profile / Who this is for / Roadmap
    st.markdown('<div class="home-section dark-section">', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1.1])

    with col_a:
        st.markdown('<div class="section-title">Founder profile</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="bio-card">
              <div class="bio-name">Anjali Nalla</div>
              <div class="bio-role">Student and aspiring AI geneticist</div>
              <div>
                I'm Anjali Nalla, a high school student with hands-on experience in genetics, neuroscience, and clinical settings.
                My background includes Johns Hopkins affiliated research focused on rare disease biology, internships that explore
                both medicine and research, and formal laboratory certifications that trained me in careful, real-world lab practice.
                I am especially interested in genetic counseling and neuro oncology, and in how AI can support thoughtful,
                evidence-aware conversations about risk and lifestyle. VivaGene is my way of connecting everything I am
                learning into a single pipeline: from raw genomic data, to structured trait interpretation, to clear explanations
                that help people understand their biology in a calm and responsible way.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown('<div class="section-title">Who this is for</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="who-row">
              <div class="who-card">
                <div class="who-title">People curious about DNA and lifestyle</div>
                <div>You want to understand how genetics may relate to your sleep, focus, training, or nutrition
                without expecting a diagnosis or a quick fix.</div>
              </div>
              <div class="who-card">
                <div class="who-title">Students in biology or pre medicine</div>
                <div>You would like a learning environment where you can practice reading trait reports, understand limitations,
                and think about how you would communicate genomics in the future.</div>
              </div>
              <div class="who-card">
                <div class="who-title">Future clinicians and AI builders</div>
                <div>You are interested in combining structured genomic data, language models, and human centered design
                in a responsible way.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown('<div class="section-title">Roadmap</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="roadmap-card">
              <div class="roadmap-label">Next steps</div>
              <div class="roadmap-title">Beyond this research project</div>
              <div>
                · Expanding the trait panel with additional genotypes and pathways that relate to lifestyle. <br/>
                · Account creation so users can save reports under a VivaGene login. <br/>
                · Email delivery of reports from a dedicated VivaGene address. <br/>
                · A cautious chatbot that lets users ask follow up questions about their traits and habits, framed as education
                  and ideas rather than medical advice.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div style="margin-top: 0.8rem; font-size: 0.86rem; color: #e5e7eb;">
          <strong>Founder highlights</strong><br/>
          · Johns Hopkins affiliated research in rare disease biology<br/>
          · Clinical volunteering on a neuro floor and in hospice settings<br/>
          · Internships that combine medicine, psychology, and research<br/>
          · Formal lab safety and benchwork certifications<br/>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Safety, privacy, and limitations (Home)
    st.markdown('<div class="home-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Safety, privacy, and limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - VivaGene is an educational research project. It does not diagnose, treat, or predict disease.  
        - This tool focuses on a small set of lifestyle-related traits, not your full genomic risk.  
        - In this current deployment, files are processed for the session and not saved to a user database.  
        - Genetics is only one part of the picture alongside sleep, nutrition, stress, and environment.  
        - For any medical questions or concerns, always speak with a licensed clinician or genetic counselor.  
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # FAQ (Home - short)
    st.markdown('<div class="home-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick questions</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **What kind of DNA files can I use?**  
        Plain text genotype files with columns like rsid, chromosome, position, and genotype. A 23andMe-style export is a common example.  

        **Will this tell me if I have a disease?**  
        No. VivaGene only looks at a small set of lifestyle-oriented traits and does not provide medical risk predictions.  

        **Is this a replacement for my doctor or genetic counselor?**  
        No. This is a learning tool. It can help you think of questions, but it cannot replace professional advice.  
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Upload & Report":
    st.markdown('<div class="section-title">Generate a personal trait report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Upload a compatible raw DNA text file. VivaGene will interpret a subset of variants, build a trait '
        'profile, and ask an AI model to generate a plain language overview with a focus on traits that may interact with lifestyle.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("AI, chatbot, and data policy"):
        st.markdown(
            """
            - Educational insights only; not medical advice  
            - No diagnosis or disease prediction  
            - Explanations are bounded by available evidence and confidence  
            - Citations are shown when available  
            - Unsupported claims are refused intentionally  
            """
        )

    newsletter_block("upload")

    if is_dev_mode():
        stats = get_trait_db_stats(TRAITS_JSON_PATH)
        st.markdown("#### DEV CHECK")
        st.caption(
            "Traits loaded: "
            f"{stats.get('traits', 0)} | Variants loaded: {stats.get('variants', 0)} | "
            f"Ready: {stats.get('ready', 0)} | Draft: {stats.get('draft', 0)} | Placeholder: {stats.get('placeholder', 0)}"
        )

    col1, col2 = st.columns([1.2, 0.9])

    with col1:
        st.markdown("#### Upload raw data")
        uploaded = st.file_uploader("Raw genotype file (.txt, 23andMe style)", type=["txt"])

        st.caption(
            "The file should contain at least rsid and genotype columns."
        )

        st.markdown("#### Where should results go?")
        email_for_result = st.text_input(
            "Optional email (for a future version that emails the PDF)",
            placeholder="you@example.com",
        )

        categories_selected = st.multiselect(
            "Trait categories",
            options=["Neurobehavior", "Nutrition", "Fitness", "Liver"],
            default=["Neurobehavior", "Nutrition", "Fitness"],
        )
        include_optional_liver = st.checkbox("Include optional liver traits", value=False)
        red_flag_only = st.checkbox("Show red-flag traits only", value=True)

        generate = st.button("Run analysis")

        if generate:
            if not uploaded:
                st.warning("Please upload a raw genotype .txt file first.")
                st.stop()

            temp_path = BASE_DIR / "uploaded_genome.txt"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getvalue())
            genotype_path = str(temp_path)

            try:
                report = build_prs_report_from_upload(
                    genotype_path=genotype_path,
                    categories_selected=categories_selected,
                    include_optional_liver=include_optional_liver,
                    red_flag_only=red_flag_only,
                )
                report = normalize_report(report)
                if not isinstance(report, dict):
                    raise ValueError("Report generation failed: expected a dict report object.")
                report = attach_rag_evidence_to_report(report)

                # Save for chatbot
                st.session_state.last_report = report

                if not report.get("traits"):
                    st.warning("No traits were matched. Check that the file uses the expected rsIDs and genotypes.")
                else:
                    major_traits = sort_traits_for_display(report.get("traits_major", []))
                    selected_traits = sort_traits_for_display(report.get("traits", []))
                    filtered_report = build_filtered_report(report, selected_traits)
                    major_only_report = build_filtered_report(report, major_traits if major_traits else selected_traits)

                    enriched_traits = [dict(t) for t in selected_traits if isinstance(t, dict)]

                    filtered_report = build_filtered_report(report, enriched_traits)

                    trust_summary = report.get("trust_summary", {}) if isinstance(report.get("trust_summary", {}), dict) else {}
                    if int(trust_summary.get("traits_refused", 0) or 0) > 0:
                        st.warning(
                            "Some traits could not be explained because the evidence corpus does not yet contain sufficient supporting snippets. This is intentional to prevent unsupported claims."
                        )

                    summary_lines = []
                    for t in enriched_traits:
                        if not isinstance(t, dict):
                            continue
                        title = str(t.get("trait_name", "")).strip()
                        life = str(t.get("explanation", "")).strip()
                        if title and life:
                            summary_lines.append(f"- {title}: {life}")
                    st.session_state.last_ai_summary = "\n".join(summary_lines[:8])

                    cats = report.get("summary", {}).get("categories", []) if isinstance(report.get("summary", {}), dict) else []
                    n_traits = int(report.get("summary", {}).get("num_traits_found", len(enriched_traits)) or len(enriched_traits))
                    cov_vals = [float(t.get("coverage", 0.0) or 0.0) for t in enriched_traits if isinstance(t, dict) and isinstance(t.get("coverage", None), (int, float))]
                    cov_txt = f"{(sum(cov_vals) / len(cov_vals)) * 100:.0f}%" if cov_vals else "—"
                    st.markdown("### Your VivaGene Trait Summary")
                    st.caption(
                        f"Categories: {', '.join(cats) if cats else '—'} · Traits found: {n_traits} · Coverage: {cov_txt}"
                    )

                    st.markdown("### Evidence-based overview")
                    st.caption("This overview is educational. It does not diagnose disease or replace medical care.")

                    st.markdown("### Trait cards")
                    traits_by_category = {}
                    for t in enriched_traits:
                        cat = t.get("category", t.get("track", "General"))
                        traits_by_category.setdefault(cat, []).append(t)

                    for category in sorted(traits_by_category.keys()):
                        st.markdown(f"#### {category}")
                        for trait in traits_by_category.get(category, []):
                            with st.container():
                                trait_name = trait.get("trait_name", trait.get("trait_id", "Trait"))
                                gene = trait.get("gene", "")
                                rsid = trait.get("rsid", "")
                                genotype = trait.get("user_genotype", "")
                                effect_label = trait.get("effect_label", "")
                                effect_level = trait.get("effect_level", "")
                                evidence_strength = trait.get("evidence_strength", "")
                                life_impact = trait.get("explanation", "") or "Insufficient evidence in the local corpus to explain this trait yet."
                                citations = trait.get("citations", []) if isinstance(trait.get("citations", []), list) else []
                                trust = trait.get("trust", {}) if isinstance(trait.get("trust", {}), dict) else {}

                                st.markdown(f"**{trait_name}**")
                                st.caption(f"Category: {category}")
                                st.caption(f"{gene} · {rsid} · Genotype {genotype}")
                                st.markdown(
                                    f"Effect: **{effect_label or 'Observed variant'}** "
                                    f"`{str(effect_level or 'Unknown').upper()}`"
                                )
                                st.write(life_impact)
                                st.caption(f"Evidence level: {evidence_strength or 'Emerging'}")
                                if trust:
                                    cov = trust.get("coverage", trait.get("coverage", None))
                                    cov_txt = f"{float(cov) * 100:.0f}%" if isinstance(cov, (int, float)) else "n/a"
                                    st.caption(
                                        f"Trust Panel · Bucket: {trust.get('bucket', effect_level)} · "
                                        f"Confidence: {trust.get('confidence', trait.get('confidence', 'Low'))} · "
                                        f"Coverage: {cov_txt} · Evidence quality: {trust.get('evidence_quality', 'Low')}"
                                    )
                                    bias_note = str(trust.get("bias_note", "")).strip()
                                    if bias_note:
                                        st.caption(f"Bias note: {bias_note}")
                                with st.expander("Evidence & citations"):
                                    if citations:
                                        st.markdown("Sources:")
                                        for c in citations[:5]:
                                            if isinstance(c, dict):
                                                title_c = str(c.get("title", "")).strip() or str(c.get("source", "")).strip() or "Source"
                                                year_c = str(c.get("year", "")).strip()
                                                source_c = str(c.get("journal", "")).strip() or str(c.get("source", "")).strip()
                                                if not source_c:
                                                    url_c = str(c.get("url", "")).strip()
                                                    if url_c:
                                                        source_c = url_c.split("/")[2] if "://" in url_c and len(url_c.split("/")) > 2 else url_c
                                                year_bit = f" ({year_c})" if year_c else ""
                                                tail = f" — {source_c}" if source_c else ""
                                                st.markdown(f"- {title_c}{year_bit}{tail}")
                                            else:
                                                text_c = str(c).strip()
                                                if "://" in text_c:
                                                    parts = text_c.split("/")
                                                    domain = parts[2] if len(parts) > 2 else text_c
                                                    st.markdown(f"- {domain}")
                                                else:
                                                    st.markdown(f"- {text_c}")
                                    else:
                                        st.write("No paper retrieved for this trait yet.")

                    text_report = generate_text_report(filtered_report)

                    with st.expander("View technical text summary"):
                        st.text(text_report)

                    st.markdown("### Detailed trait report")
                    html_report = generate_html_report(filtered_report, ai_summary=None)
                    st.components.v1.html(html_report, height=850, scrolling=True)

                    if email_for_result.strip():
                        st.info(
                            f"In a future deployed version, this report could also be sent securely to {email_for_result.strip()} "
                            "from a VivaGene email address."
                        )

            except Exception as e:
                st.error(f"Something went wrong while generating the report: {e}")
                with st.expander("Debug traceback"):
                    st.code(traceback.format_exc(), language="text")

    with col2:
        st.markdown("#### What your report looks like")
        st.markdown(
            """
            You will receive:
            
            - An evidence-based overview of your matched traits  
            - Individual trait cards showing rsIDs, genes, and genotype  
            - Short explanations of what each trait may mean in everyday life  
            - A scannable HTML layout that you can revisit later  
            """
        )
        st.markdown("---")
        st.markdown("#### What VivaGene does")
        st.markdown(
            """
            - Parses a raw genotype file line by line  
            - Matches known variants against a curated trait database  
            - Builds a structured JSON representation of the trait profile  
            - Uses a language model to write an educational overview  
            - Renders a printable HTML report  
            """
        )
        st.markdown("---")
        st.markdown("#### What VivaGene does not do")
        st.markdown(
            """
            - It does not diagnose conditions  
            - It does not replace medical care  
            - It does not cover the entire genome  
            """
        )
        st.caption(
            "Always speak with a licensed clinician or genetic counselor for health decisions."
        )

# ---------- LIFESTYLE CHATBOT ----------
elif page == "Lifestyle Q&A":
    st.markdown('<div class="section-title">Lifestyle Q&A</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Ask questions about your trait report, habits, and what you might want to pay attention to. '
        'The chatbot is designed for education and lifestyle ideas, not for medical advice.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Chatbot policy")
    st.markdown(
        """
        - Educational and informational only  
        - No diagnosis or disease prediction  
        - No medication or treatment guidance  
        - If you mention symptoms, the chatbot will advise professional care  
        - Uses cautious language  
        - When available, citations are shown and the bot refuses unsupported claims  
        """
    )
    ack = st.checkbox("I understand this is not medical advice", value=False, key="chatbot_policy_ack")

    st.markdown("#### Lifestyle overview plan")

    # Show previously generated plan if available
    existing_plan = st.session_state.get("lifestyle_plan")
    if existing_plan:
        with st.expander("View your current lifestyle plan", expanded=True):
            st.markdown(existing_plan)

    st.caption(
        "This plan is educational only. It suggests gentle habit ideas based on your traits and should not be treated as medical guidance."
    )

    generate_plan_clicked = st.button("Generate / refresh my lifestyle plan", disabled=not ack)

    report = st.session_state.get("last_report")
    ai_summary = st.session_state.get("last_ai_summary")

    if not ack:
        st.info("Please acknowledge to continue.")

    if report is None:
        st.info("Generate a report on the Upload page first so the chatbot has context about your traits.")
    else:
        # Optionally generate a structured lifestyle overview plan
        if generate_plan_clicked:
            lifestyle_context = f"Trait JSON:\n{report}\n\nAI summary of traits:\n{ai_summary or ''}"
            lifestyle_system = (
                "You are a careful genetics-informed lifestyle coach. "
                "Given a structured trait report and an AI summary, create a short, non-medical lifestyle plan. "
                "Organize the plan into sections such as Sleep, Focus & Learning, Movement & Recovery, Caffeine & Stimulants, "
                "and Everyday Habits. For each section, list 3-5 gentle, practical ideas that could be helpful for someone with these traits. "
                "Use tentative language (may, might, could) and remind the reader that this is not medical advice."
            )

            try:
                prompt = f"""
{lifestyle_system}

Context:
{lifestyle_context}
"""
                lifestyle_plan = generate_with_fallback_model(prompt)
                if not lifestyle_plan.strip():
                    lifestyle_plan = build_local_lifestyle_plan(report)
                st.session_state["lifestyle_plan"] = lifestyle_plan

                with st.expander("View your current lifestyle plan", expanded=True):
                    st.markdown(lifestyle_plan)
            except Exception as e:
                st.error(f"There was an error generating the lifestyle plan: {e}")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show previous messages
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)

        user_input = st.chat_input("Ask a question about your traits or lifestyle", disabled=not ack)
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            system_prompt = (
                "You are a genetics informed lifestyle coach. "
                "You receive a JSON report of traits and a short AI summary. "
                "You may discuss possible lifestyle ideas related to sleep, focus, caffeine, training, and general wellness. "
                "You must avoid medical advice, diagnosis, or treatment recommendations. "
                "Use careful language like may, might, and could, and encourage the user to talk with a clinician "
                "or genetic counselor for any medical questions."
            )

            context_snippet = f"Trait JSON:\n{report}\n\nSummary:\n{ai_summary or ''}"

            with st.chat_message("assistant"):
                try:
                    prompt = f"""
{system_prompt}

Context:
{context_snippet}

User question:
{user_input}
"""
                    reply = generate_with_fallback_model(prompt)
                    if not reply.strip():
                        reply = build_local_chat_reply(user_input, report)
                except Exception as e:
                    reply = f"There was an error calling the model: {e}"

                st.markdown(reply)
                st.session_state.chat_history.append(("assistant", reply))

 # ---------- TRAIT EXPLORER ----------
elif page == "Trait Explorer":
    st.markdown('<div class="section-title">Trait explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Browse and search the underlying trait database used by the interpreter. '
        'This view is for learning and debugging, not for medical interpretation.</div>',
        unsafe_allow_html=True,
    )

    # Load trait rows (JSON preferred, CSV fallback)
    trait_rows = load_trait_rows(TRAIT_DB_PATH, TRAITS_JSON_PATH)

    if not trait_rows:
        st.info("No trait rows could be loaded from the database.")
    else:
        # Build simple filter options
        all_tracks = sorted({r.get("track", "") for r in trait_rows if r.get("track")})
        all_subcategories = sorted({r.get("subcategory", "") for r in trait_rows if r.get("subcategory")})
        all_evidence = sorted({r.get("evidence_strength", "") for r in trait_rows if r.get("evidence_strength")})
        all_quality_labels = ["Ready", "Draft", "Placeholder"]

        col_search, col_track, col_subcat, col_ev, col_quality = st.columns([1.2, 0.9, 0.9, 0.8, 0.9])
        with col_search:
            query = st.text_input("Search by rsID, gene, or title")
        with col_track:
            track_filter = st.multiselect("Filter by track", options=all_tracks)
        with col_subcat:
            subcat_filter = st.multiselect("Filter by subcategory", options=all_subcategories)
        with col_ev:
            ev_filter = st.multiselect("Filter by evidence", options=all_evidence)
        with col_quality:
            quality_filter = st.multiselect("Filter by quality", options=all_quality_labels)

        show_liver = st.checkbox("Show Liver traits", value=True)
        min_score = st.slider("Minimum completeness score", min_value=0, max_value=100, value=0, step=5)

        # Apply filters
        def match_row(row):
            text = (row.get("rsid", "") + " " + row.get("gene", "") + " " + row.get("title", row.get("trait_name", ""))).lower()
            if query and query.lower() not in text:
                return False
            if track_filter and row.get("track") not in track_filter:
                return False
            if subcat_filter and row.get("subcategory") not in subcat_filter:
                return False
            if ev_filter and row.get("evidence_strength") not in ev_filter:
                return False
            if quality_filter and row.get("quality_label") not in quality_filter:
                return False
            if int(row.get("completeness_score", 0)) < min_score:
                return False
            if not show_liver and (row.get("track") or "").strip().lower() == "liver":
                return False
            return True

        filtered = [r for r in trait_rows if match_row(r)]

        st.markdown(f"Showing **{len(filtered)}** of **{len(trait_rows)}** traits.")

        if filtered:
            # Show a compact table
            display_cols = [c for c in ["trait_id", "title", "track", "subcategory", "quality_label", "completeness_score", "missing_fields_count", "rsid", "gene", "genotype", "effect_label", "effect_level", "evidence_strength"] if c in filtered[0]]
            st.dataframe([{k: r.get(k) for k in display_cols} for r in filtered], use_container_width=True, hide_index=True)

            # Lightweight "click" behavior via selector to show trait status.
            unique_traits = []
            seen = set()
            for row in filtered:
                key = (row.get("trait_id", ""), row.get("title", row.get("trait_name", "")))
                if key in seen:
                    continue
                seen.add(key)
                unique_traits.append(row)
            selected_idx = st.selectbox(
                "Inspect trait",
                options=list(range(len(unique_traits))),
                format_func=lambda i: f"{unique_traits[i].get('trait_id','')} - {unique_traits[i].get('title', unique_traits[i].get('trait_name',''))}",
            )
            selected_trait = unique_traits[selected_idx]
            selected_label = selected_trait.get("quality_label", "Placeholder")
            if selected_label == "Placeholder":
                st.info("Coming soon: This trait is scaffolded for structure but not yet populated with variant rules and evidence.")
            elif selected_label == "Draft":
                st.warning("Draft: This trait exists but may have limited variants or incomplete explanations.")
            else:
                st.success("Ready: This trait is active in the interpreter.")

            st.markdown("---")
            st.markdown("#### JSON model preview for the first filtered trait")
            first = selected_trait
            # Convert a CSV row into a JSON-like trait object
            json_trait = {
                "rsid": first.get("rsid"),
                "gene": first.get("gene"),
                "trait_category": first.get("track"),
                "trait_subcategory": first.get("subcategory"),
                "trait_name": first.get("title", first.get("trait_name")),
                "genotype": first.get("genotype"),
                "variant_effect": first.get("effect_label"),
                "effect_level": first.get("effect_level"),
                "evidence_level": first.get("evidence_strength"),
                "explanation": first.get("explanation"),
                "mechanism": first.get("mechanism") or "",  # optional column
                "lifestyle_links": [],  # can be populated later
                "notes": "For education only; not a diagnosis.",
            }

            st.code(json.dumps(json_trait, indent=2), language="json")

            # Optional: export full JSON model
            if st.button("Export full trait database as JSON model"):
                json_model = []
                for row in filtered:
                    obj = {
                        "rsid": row.get("rsid"),
                        "gene": row.get("gene"),
                        "trait_category": row.get("track"),
                        "trait_subcategory": row.get("subcategory"),
                        "trait_name": row.get("title", row.get("trait_name")),
                        "genotype": row.get("genotype"),
                        "variant_effect": row.get("effect_label"),
                        "effect_level": row.get("effect_level"),
                        "evidence_level": row.get("evidence_strength"),
                        "explanation": row.get("explanation"),
                        "mechanism": row.get("mechanism") or "",
                        "lifestyle_links": [],
                        "notes": "For education only; not a diagnosis.",
                    }
                    json_model.append(obj)

                try:
                    with open("trait_database_model.json", "w", encoding="utf-8") as f:
                        json.dump(json_model, f, ensure_ascii=False, indent=2)
                    st.success("Exported JSON model to trait_database_model.json in the project folder.")
                except Exception as e:
                    st.error(f"Could not write JSON model file: {e}")
        else:
            st.info("No traits matched your filters.")


# ---------- TRAIT SCIENCE ----------
elif page == "Trait Science":
    st.markdown('<div class="section-title">Science behind the traits</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A high-level overview of how VivaGene turns individual genetic variants into trait interpretations.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### What is a SNP?", unsafe_allow_html=True)
    st.markdown(
        """
        A single nucleotide polymorphism (SNP) is a position in the genome where people commonly differ by a single base.
        VivaGene uses SNPs that have been studied in the literature and are associated with traits such as caffeine
        metabolism, sleep patterns, exercise response, and certain neurobehavioral tendencies.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How traits are defined", unsafe_allow_html=True)
    st.markdown(
        """
        Each trait in the internal database links:
        
        - One or more rsIDs (SNP identifiers)  
        - The gene or genomic region  
        - A description of the reported effect (for example, typical vs. increased sensitivity)  
        - A qualitative evidence label that reflects the strength and consistency of published findings  
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What this panel focuses on", unsafe_allow_html=True)
    st.markdown(
        """
        The current research project focuses on a small set of traits that connect to everyday lifestyle questions, such as:
        
        - Sleep timing and sleep depth tendencies  
        - Response to caffeine and stimulants  
        - Exercise and recovery related traits  
        - Certain sensory and neurobehavior-related features  
        
        These are chosen because they are easier to connect to day-to-day habits and are safer to discuss in an
        educational context than serious disease risk.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Why nothing here is a diagnosis", unsafe_allow_html=True)
    st.markdown(
        """
        Most common genetic variants have small effects that interact with many other factors like sleep, stress,
        environment, and medical history. VivaGene treats traits as gentle hints rather than answers. Any serious
        medical concern should always be discussed with a healthcare professional.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Data model sketch", unsafe_allow_html=True)
    st.markdown(
        """
        Internally, each trait can be represented as a small JSON object that connects the genetic signal to a lifestyle-oriented
        explanation. A simplified example structure might look like:
        """,
        unsafe_allow_html=True,
    )

    st.code(
        '{\n'
        '  "rsid": "rs1234",\n'
        '  "gene": "ADORA2A",\n'
        '  "trait_category": "Caffeine sensitivity",\n'
        '  "variant_effect": "Increased sensitivity to caffeine",\n'
        '  "mechanism": "Adenosine receptor signaling differences",\n'
        '  "evidence_level": "Moderate",\n'
        '  "lifestyle_links": [\n'
        '    "May feel stronger effects from caffeine",\n'
        '    "Might benefit from limiting caffeine later in the day"\n'
        '  ],\n'
        '  "notes": "For education only; not a diagnosis."\n'
        '}',
        language="json",
    )

    st.markdown(
        """
        Over time, this kind of structure can be expanded into a richer knowledge graph that links traits to sources, study quality,
        and more nuanced lifestyle ideas, while still keeping the interface simple for end users.
        """,
        unsafe_allow_html=True,
    )

# ---------- ABOUT ----------
elif page == "About":
    st.markdown('<div class="section-title">About VivaGene</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'VivaGene is an educational research project built to explore how AI can help interpret '
        'small scale genomic trait panels in a transparent and lifestyle aware way.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Concept", unsafe_allow_html=True)
        st.markdown(
            """
            This project combines three layers:
            1. A structured genetics engine that maps specific SNPs to trait interpretations  
            2. A curated database with gene, rsID, and literature based explanations  
            3. A language model that converts this structure into a narrative overview focused on education and lifestyle  
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Design principles", unsafe_allow_html=True)
        st.markdown(
            """
            - Educational first: explain, do not prescribe  
            - Transparent: the underlying trait data stays visible  
            - Lifestyle aware: DNA is framed as one input among sleep, habits, and environment  
            - Human centered: final decisions belong to people, not models  
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Technical sketch", unsafe_allow_html=True)
    st.code(
        """User file (.txt) 
   ↓
Parse SNPs and genotypes
   ↓
Match against trait_database.csv
   ↓
Build JSON report object
   ↓
Call Gemini model for overview text
   ↓
Render HTML report (and optional PDF)""",
        language="text",
    )

    # Safety and limitations (About)
    st.markdown('<div class="section-title">Safety and limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        VivaGene is designed as an educational tool. It highlights potential trait patterns and how they might relate to
        lifestyle, but it does not measure risk for disease or replace professional genetic counseling. All interpretations
        are simplified and should be viewed as conversation starters, not conclusions.
        """,
        unsafe_allow_html=True,
    )

    # FAQ (About - extended)
    st.markdown('<div class="section-title">Frequently asked questions</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Does VivaGene store my DNA data?**  
        In this research project, files are handled within your session. A production deployment would include a clear privacy
        policy and options for data deletion or local-only processing.

        **Can this tell me what conditions I have or will develop?**  
        No. The trait panel is limited and focuses on common variants related to lifestyle and tendencies. Medical genetics
        requires much deeper analysis and professional interpretation.

        **Is AI making up results?**  
        The AI model is used to turn structured trait data into readable text. The underlying rsIDs, genes, and effect
        labels come from a curated table, not from the model inventing variants.

        **Who is this for?**  
        Students, early researchers, and curious individuals who want to practice thinking about genomics and lifestyle
        in a careful, low-stakes way.
        """,
        unsafe_allow_html=True,
    )

# ---------- CONTACT ----------
elif page == "Contact":
    st.markdown('<div class="section-title">Contact</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Questions, feedback, or research collaboration? '
        'Use the form below to leave a message.</div>',
        unsafe_allow_html=True,
    )
    st.caption("For privacy, do not send raw genotype files through this form.")

    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        role = st.selectbox(
            "I am primarily a…",
            ["Student", "Educator", "Researcher or clinician", "Developer", "Other"],
        )
        message = st.text_area("Message", height=140)
        submitted = st.form_submit_button("Send message")

        if submitted:
            if not (name.strip() and email.strip() and message.strip()):
                st.warning("Please fill in name, email, and a brief message.")
            else:
                st.success(
                    "Thank you for reaching out. In a production environment this would send your message to the project owner."
              )

# ---------- VALIDATION DEV ----------
elif page == "Validation (Dev)" and str(os.environ.get("SHOW_VALIDATION", "0")).strip() == "1":
    st.markdown('<div class="section-title">Validation (Dev)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Local validation artifacts for benchmarking and paper figures.</div>',
        unsafe_allow_html=True,
    )
    from pathlib import Path as _Path
    out_dir = _Path("validation_outputs").resolve()
    if not out_dir.exists():
        st.info("No validation outputs found yet. Run `python -m validation.run_validation` first.")
    else:
        pngs = sorted(out_dir.glob("*.png"))
        csvs = sorted(out_dir.glob("*.csv"))
        jsons = sorted(out_dir.glob("*.json"))

        if pngs:
            st.markdown("#### Plots")
            for p in pngs:
                st.markdown(f"**{p.name}**")
                st.image(str(p), use_container_width=True)
        if csvs or jsons:
            st.markdown("#### Downloads")
            for f in csvs + jsons:
                data = f.read_bytes()
                st.download_button(
                    label=f"Download {f.name}",
                    data=data,
                    file_name=f.name,
                    mime="application/octet-stream",
                    key=f"dl_{f.name}",
                )

st.markdown("</div>", unsafe_allow_html=True)
# ---------- Global footer ----------
st.markdown(
    """
    <div class="site-footer">
      <div>
        VivaGene · Educational genomics insights · Not medical advice.
      </div>
      <div>
        Built by Anjali Nalla · Independent research project.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")
