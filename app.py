import streamlit as st
import csv
import json
import re
import subprocess
import sys
import textwrap
import base64
import hashlib
import time
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from genomics_interpreter import (
    TRAIT_DB_PATH,
    TRAITS_JSON_PATH,
    load_traits_catalog,
    build_prs_report_from_upload,
    generate_html_report,
    parse_genotype_file,
)
import os
from utils.trait_quality import compute_trait_completeness, trait_has_variants
from rag_retrieval import load_study_packs, retrieve_evidence
from polygenic import compute_prs
from evidence_guard import enforce_evidence_only
from utils.evidence_retrieval import retrieve_trait_evidence
from utils.rag_explainer import generate_trait_explanation
from study_pack_loader import load_all_trait_packs
from utils.prompts import LLM_SAFETY_SYSTEM
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
BASE_DIR = Path(__file__).resolve().parent
STUDY_PACK_DIR = BASE_DIR / "trait_study_packs"
ASSETS_DIR = BASE_DIR / "assets"
VIVAGENE_LOGO_PATH = ASSETS_DIR / "vivagene_logo.svg"
VIVAGENE_MARK_PATH = ASSETS_DIR / "vivagene_mark.png"
CACHE_UPLOAD_DIR = BASE_DIR / ".cache_uploads"
ENGINE_VERSION = "v1"


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, os.getenv(name, default))
    except Exception:
        value = os.getenv(name, default)
    return str(value or "").strip()


def sanitize_svg(svg: str) -> str:
    if not isinstance(svg, str):
        return ""
    cleaned = re.sub(r"<\?xml[^>]*\?>", "", svg, flags=re.IGNORECASE)
    cleaned = re.sub(r"<!DOCTYPE[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<script[\s\S]*?</script>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<foreignObject[\s\S]*?</foreignObject>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s(on\w+)\s*=\s*\"[^\"]*\"", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s(on\w+)\s*=\s*'[^']*'", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s(xlink:href|href)\s*=\s*\"https?://[^\"]*\"", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s(xlink:href|href)\s*=\s*'https?://[^']*'", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"javascript:", "", cleaned, flags=re.IGNORECASE)
    if "<svg" not in cleaned.lower():
        return ""
    return cleaned.strip()


def load_svg_safe(path: str) -> str:
    try:
        raw = Path(path).read_text(encoding="utf-8")
        return sanitize_svg(raw)
    except Exception:
        return ""


def safe_html(html: str):
    if html is None:
        html = ""
    if not isinstance(html, str):
        html = str(html)
    st.markdown(html.strip(), unsafe_allow_html=True)


def begin_page_wrap():
    safe_html(f"<div class='page-wrap' data-k='{st.session_state.get('page_transition_key', 0)}'>")


def end_page_wrap():
    safe_html("</div>")


def render_vg_helix_svg() -> str:
    if not VIVAGENE_MARK_PATH.exists():
        return "<div class='vg-mark-wrap'><div class='trait-muted'>VivaGene Helix Mark (missing asset)</div></div>"
    try:
        encoded = base64.b64encode(VIVAGENE_MARK_PATH.read_bytes()).decode("utf-8")
        uri = f"data:image/png;base64,{encoded}"
        return (
            '<div class="vg-mark-wrap mark-card" aria-hidden="true">'
            f'<img src="{uri}" alt="VivaGene Helix Mark"/>'
            "</div>"
        )
    except Exception:
        return "<div class='vg-mark-wrap'><div class='trait-muted'>VivaGene Helix Mark (missing asset)</div></div>"

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


def normalize_category(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    neuro = ["sleep", "stress", "anxiety", "focus", "attention", "mood", "dopamine", "serotonin", "neuro", "behavior", "cognition", "hpa", "circadian", "sensory"]
    nutrition = ["nutrition", "vitamin", "folate", "b12", "iron", "glucose", "lipid", "omega", "taste", "caffeine", "alcohol", "metabolism", "appetite"]
    fitness = ["fitness", "strength", "endurance", "muscle", "recovery", "vo2", "training", "injury", "actn3", "ace"]
    liver = ["liver", "nafld", "fatty liver", "alt", "ast", "pnpla3", "tm6sf2"]
    if any(k in text for k in neuro):
        return "Neurobehavior"
    if any(k in text for k in nutrition):
        return "Nutrition"
    if any(k in text for k in fitness):
        return "Fitness"
    if any(k in text for k in liver):
        return "Liver"
    return ""


def two_sentence_text(text: str) -> str:
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", raw)
    out = [p for p in parts if p.strip()][:2]
    return " ".join(out).strip()


def _clip_text(s: str, limit: int) -> str:
    txt = " ".join(str(s or "").split())
    if len(txt) <= limit:
        return txt
    return txt[:limit].rstrip() + "..."


def _safe_json_obj(text: str):
    raw = str(text or "").strip()
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


def _stable_traits_hash(items: list[dict]) -> str:
    payload = []
    for t in items if isinstance(items, list) else []:
        if not isinstance(t, dict):
            continue
        payload.append(
            {
                "trait_id": t.get("trait_id", ""),
                "trait_name": t.get("trait_name", ""),
                "gene": t.get("gene", ""),
                "rsid": t.get("rsid", ""),
                "genotype": t.get("user_genotype", ""),
                "signal": t.get("effect_level", t.get("bucket", "")),
                "effect": t.get("effect_label", ""),
                "coverage": t.get("coverage", 0.0),
                "evidence": t.get("evidence_strength", ""),
                "citations": t.get("sources", t.get("citations", [])),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def fallback_explanation_for_trait(trait: dict, category: str) -> str:
    effect = str(trait.get("effect_label", "")).strip() or str(trait.get("effect_level", "")).strip() or "a trait tendency"
    cat_line = {
        "Neurobehavior": "attention, stress response, or sleep rhythm patterns",
        "Nutrition": "stimulant, appetite, or nutrient-response patterns",
        "Fitness": "training response or recovery patterns",
        "Liver": "metabolic processing patterns",
    }.get(category, "day-to-day trait patterns")
    return (
        f"This variant pattern is associated with {effect} in published studies. "
        f"In daily life, this may relate to {cat_line}, though individual response can vary."
    )


def citation_to_link(c: dict):
    if not isinstance(c, dict):
        return ("Source", "")
    ident = str(c.get("identifier", "")).strip()
    title = str(c.get("title", "")).strip() or ident or "Source"
    doi = str(c.get("doi", "")).strip()
    pmid = str(c.get("pmid", "")).strip()
    pmcid = str(c.get("pmcid", "")).strip()
    src = str(c.get("source_url", "")).strip()
    if doi:
        return (title, f"https://doi.org/{doi}")
    if pmid:
        return (title, f"https://europepmc.org/article/MED/{pmid}")
    if pmcid:
        return (title, f"https://europepmc.org/article/PMC/{pmcid}")
    if ident.upper().startswith("DOI:"):
        return (title, f"https://doi.org/{ident.split(':', 1)[1].strip()}")
    if ident.upper().startswith("PMID:"):
        return (title, f"https://europepmc.org/article/MED/{ident.split(':', 1)[1].strip()}")
    if ident.upper().startswith("PMCID:"):
        return (title, f"https://europepmc.org/article/PMC/{ident.split(':', 1)[1].strip()}")
    return (title, src)


def _html_escape(text: str) -> str:
    s = str(text or "")
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def build_category_panel_html(category: str, traits: list[dict]) -> str:
    cards = []
    for trait in traits:
        if not isinstance(trait, dict):
            continue
        trait_name = _html_escape(trait.get("trait_name", trait.get("trait_id", "Trait")))
        gene = _html_escape(trait.get("gene", ""))
        rsid = _html_escape(trait.get("rsid", ""))
        genotype = _html_escape(trait.get("user_genotype", ""))
        effect_line = _html_escape(str(trait.get("effect_level", "")).strip() or str(trait.get("effect_label", "")).strip() or "Observed")
        expl = _html_escape(trait.get("_display_explanation", ""))
        cites = trait.get("citations", []) if isinstance(trait.get("citations", []), list) else []
        if cites:
            cite_lines = []
            for c in cites[:3]:
                label, url = citation_to_link(c)
                label_h = _html_escape(label)
                url_h = _html_escape(url)
                if url_h:
                    cite_lines.append(f"<li><a href='{url_h}' target='_blank'>{label_h}</a></li>")
                else:
                    cite_lines.append(f"<li>{label_h}</li>")
            citations_html = "<ul class='trait-citations'>" + "".join(cite_lines) + "</ul>"
        else:
            citations_html = "<div class='trait-citations-empty'>No citations available yet for this trait.</div>"
        cards.append(
            textwrap.dedent(
                f"""
                <div class='trait-mini-card'>
                  <div class='trait-mini-title'>{trait_name}</div>
                  <div class='trait-mini-meta'>Gene {gene} · rsID {rsid} · Genotype {genotype}</div>
                  <div class='trait-mini-meta'>Signal {effect_line}</div>
                  <div class='trait-mini-expl'>{expl}</div>
                  <div class='trait-mini-cite-label'>Citations</div>
                  {citations_html}
                </div>
                """
            ).strip()
        )
    count = len([t for t in traits if isinstance(t, dict)])
    cards_html = "\n".join(cards)
    return textwrap.dedent(
        f"""
        <div class='results-panel'>
          <div class='results-panel-header'>
            <span>{_html_escape(category)}</span>
            <span class='results-count'>{count}</span>
          </div>
          <div class='results-panel-body'>
            {cards_html}
          </div>
        </div>
        """
    ).strip()


def build_tab_panel_html(traits: list[dict], mode_label: str) -> str:
    cards = []
    for trait in traits:
        if not isinstance(trait, dict):
            continue
        name = _html_escape(trait.get("trait_name", trait.get("trait_id", "Trait")))
        gene = _html_escape(trait.get("gene", ""))
        rsid = _html_escape(trait.get("rsid", ""))
        gt = _html_escape(trait.get("user_genotype", "not found"))
        coverage = trait.get("coverage", 0.0)
        cov_txt = f"{float(coverage) * 100:.0f}%" if isinstance(coverage, (int, float)) else "0%"
        evidence_level = _html_escape(trait.get("evidence_strength", "Unknown"))
        signal = _html_escape(str(trait.get("effect_level", "")).strip() or str(trait.get("effect_label", "")).strip())
        summary = _html_escape(trait.get("_final_summary", ""))
        life = _html_escape(trait.get("_final_life_impact", ""))
        explanation_html = f"{summary} {life}".strip() or "Evidence pending — no explanatory claims shown yet (RAG safety)."
        has_variant = bool(trait.get("has_any_variant", False))
        variant_note = "" if has_variant else "<div class='trait-missing'>No variants found in uploaded file.</div>"
        queries = trait.get("_queries_used", []) if isinstance(trait.get("_queries_used", []), list) else []
        queries_html = ""
        if not summary:
            qtxt = ", ".join(_html_escape(q) for q in queries[:3]) if queries else "No query metadata available."
            queries_html = f"<div class='trait-query'>Queries used: {qtxt}</div>"
        citations = trait.get("_final_citations", []) if isinstance(trait.get("_final_citations", []), list) else []
        if citations:
            cite_items = []
            for c in citations[:3]:
                label, url = citation_to_link(c)
                lbl = _html_escape(label)
                if url:
                    cite_items.append(f"<li><a href='{_html_escape(url)}' target='_blank'>{lbl}</a></li>")
                else:
                    cite_items.append(f"<li>{lbl}</li>")
            cites_html = "<ul class='trait-citations'>" + "".join(cite_items) + "</ul>"
        else:
            cites_html = "<div class='trait-citations-empty'>Citations pending.</div>"
        cards.append(
            textwrap.dedent(
                f"""
                <div class='trait-mini-card'>
                  <div class='trait-mini-title'>{name}</div>
                  <div class='trait-mini-meta'>Gene {gene} · rsID {rsid} · Genotype {gt}</div>
                  <div class='trait-mini-meta'>Coverage {cov_txt} · Evidence level {evidence_level}</div>
                  <div class='trait-mini-meta'>Signal {signal}</div>
                  {variant_note}
                  <div class='trait-mini-expl'>{explanation_html}</div>
                  {queries_html}
                  <div class='trait-mini-cite-label'>Sources</div>
                  {cites_html}
                </div>
                """
            ).strip()
        )
    cards_html = "\n".join(cards) if cards else "<div class='trait-citations-empty'>No matching variants from your file for this category yet.</div>"
    return textwrap.dedent(
        f"""
        <div class='single-results-panel'>
          <div class='single-results-head'>Mode: {_html_escape(mode_label)}</div>
          <div class='single-results-body'>
            {cards_html}
          </div>
        </div>
        """
    ).strip()


def render_trait_card(trait: dict, mode_label: str):
    t = trait if isinstance(trait, dict) else {}
    name = _html_escape(t.get("trait_name", "Trait"))
    gene = _html_escape(t.get("gene", ""))
    rsid = _html_escape(t.get("rsid", ""))
    gt = _html_escape(t.get("user_genotype", ""))
    evidence = _html_escape(t.get("evidence_strength", ""))
    signal = _html_escape(
        str(t.get("signal_bucket", "")).strip()
        or str(t.get("effect_level", "")).strip()
        or str(t.get("effect_label", "")).strip()
    )
    coverage = t.get("coverage", "")
    if isinstance(coverage, (int, float)):
        coverage_txt = f"{float(coverage) * 100:.0f}%"
    else:
        coverage_txt = _html_escape(str(coverage or ""))

    is_patient = str(mode_label).strip().lower().startswith("patient")
    explanation = (
        t.get("explanation_patient")
        if is_patient
        else t.get("explanation_doctor")
    )
    if not explanation:
        explanation = (
            f"{str(t.get('_final_summary', '')).strip()} {str(t.get('_final_life_impact', '')).strip()}"
        ).strip()
    explanation = str(explanation or "").replace("<b>", "").replace("</b>", "")
    explanation = re.sub(r"<[^>]+>", " ", explanation)
    explanation = _html_escape(" ".join(explanation.split()))

    lifestyle_one = str(t.get("_lifestyle_one_liner", "")).strip()
    lifestyle_bullets = t.get("_lifestyle_bullets", [])
    if not isinstance(lifestyle_bullets, list):
        lifestyle_bullets = []
    lifestyle_bullets = [str(x).strip() for x in lifestyle_bullets if str(x).strip()][:3]
    sources = t.get("_lifestyle_citations", t.get("_final_citations", t.get("citations", [])))
    if not isinstance(sources, list):
        sources = []

    html = f"""
    <div class="trait-card">
      <div class="trait-title">{name}
        {f'<span class="chip">{signal}</span>' if signal else ''}
      </div>
      <div class="trait-meta">
        Gene: <b>{gene}</b> · rsID: <b>{rsid}</b> · Genotype: <b>{gt}</b><br/>
        {f'Coverage: <b>{coverage_txt}</b> · ' if coverage_txt else ''}Evidence: <b>{evidence or 'Unknown'}</b>
      </div>
      <div class="trait-body"><b>What this may mean in daily life</b></div>
    """
    if lifestyle_one:
        html += f"<div class='trait-body'>{_html_escape(lifestyle_one)}</div>"
    if lifestyle_bullets:
        html += "<div class='trait-body'><ul>"
        for b in lifestyle_bullets:
            html += f"<li>{_html_escape(b)}</li>"
        html += "</ul></div>"
    if not lifestyle_one and not lifestyle_bullets:
        html += f"<div class='trait-body'>{explanation or 'Evidence pending — no explanatory claims shown yet (RAG safety).'}</div>"

    if sources:
        html += "<div class='trait-sources'><b>Sources</b><ul>"
        for s in sources[:3]:
            if isinstance(s, dict):
                title = _html_escape(s.get("title", "Study"))
                url = _html_escape(s.get("url", s.get("source_url", "")))
                pmid = _html_escape(s.get("pmid", ""))
                if is_patient:
                    label = title
                else:
                    label = f"{title} (PMID: {pmid})" if pmid else title
                if url:
                    html += f"<li><a href='{url}' target='_blank'>{label}</a></li>"
                else:
                    html += f"<li>{label}</li>"
            else:
                html += f"<li>{_html_escape(str(s))}</li>"
        html += "</ul></div>"
    else:
        queries = t.get("_queries_used", [])
        qtxt = ", ".join(_html_escape(str(q)) for q in queries[:2]) if isinstance(queries, list) and queries else "query pending"
        html += (
            "<div class='trait-sources trait-muted'>"
            "Evidence not found in the local corpus yet. This trait will display citations once the study pack is expanded."
            f" Retrieval status: {qtxt}."
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def compute_category_overlap(parsed_rsids: set[str], include_liver: bool) -> dict:
    packs = load_all_trait_packs(
        base_dir="trait_study_packs",
        categories=["Neurobehavior", "Nutrition", "Fitness", "Liver"],
        include_optional=include_liver,
    )
    by_cat = {"Neurobehavior": set(), "Nutrition": set(), "Fitness": set(), "Liver": set()}
    for p in packs:
        if not isinstance(p, dict):
            continue
        cat = str(p.get("category", "")).strip()
        if cat not in by_cat:
            continue
        for v in p.get("variants", []) if isinstance(p.get("variants", []), list) else []:
            if not isinstance(v, dict):
                continue
            r = str(v.get("rsid", "")).strip()
            if r:
                by_cat[cat].add(r)
    out = {}
    for cat, rsids in by_cat.items():
        overlap = parsed_rsids & rsids
        out[cat] = {
            "panel_rsids": len(rsids),
            "overlap_count": len(overlap),
            "overlap_sample": sorted(list(overlap))[:10],
        }
    return out


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


def generate_pdf_report_bytes(report_obj: dict, mode_used: str = "") -> bytes:
    report = normalize_report(report_obj if isinstance(report_obj, dict) else {})
    traits = [t for t in report.get("traits", []) if isinstance(t, dict)]
    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("VivaGene Report", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Educational genomics insights only. This report does not provide medical advice or diagnosis.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 8))
    if mode_used:
        story.append(Paragraph(f"Explanation mode: {mode_used}", styles["BodyText"]))
        story.append(Spacer(1, 6))

    cats = summary.get("categories", []) if isinstance(summary.get("categories", []), list) else []
    coverage_vals = [float(t.get("coverage", 0.0) or 0.0) for t in traits if isinstance(t.get("coverage"), (int, float))]
    avg_cov = (sum(coverage_vals) / len(coverage_vals)) if coverage_vals else 0.0
    found = sum(1 for t in traits if str(t.get("evidence_status", "")).strip().lower() == "found")
    missing = max(0, len(traits) - found)

    story.append(
        Paragraph(
            f"Matched traits: {len(traits)} | Evidence found: {found} | Evidence missing: {missing} | Coverage: {avg_cov * 100:.0f}%",
            styles["BodyText"],
        )
    )
    story.append(Paragraph(f"Categories: {', '.join(cats) if cats else '—'}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    table_data = [["Trait", "Category", "Gene/rsID", "Genotype", "Evidence"]]
    for t in traits:
        table_data.append(
            [
                str(t.get("trait_name", "")).strip() or str(t.get("trait_id", "")).strip(),
                str(t.get("category", t.get("track", ""))).strip(),
                f"{str(t.get('gene', '')).strip()} / {str(t.get('rsid', '')).strip()}",
                str(t.get("user_genotype", "")).strip(),
                str(t.get("evidence_status", "missing")).strip().capitalize(),
            ]
        )

    table = Table(table_data, colWidths=[150, 90, 120, 85, 70], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    for t in traits:
        trait_name = str(t.get("trait_name", "")).strip() or str(t.get("trait_id", "Trait")).strip()
        story.append(Paragraph(f"<b>{trait_name}</b>", styles["Heading4"]))
        story.append(
            Paragraph(
                f"{str(t.get('gene', '')).strip()} · {str(t.get('rsid', '')).strip()} · Genotype {str(t.get('user_genotype', '')).strip()}",
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                f"Effect: {str(t.get('effect_label', '')).strip()} ({str(t.get('effect_level', '')).strip()}) | Evidence: {str(t.get('evidence_strength', '')).strip() or 'Emerging'}",
                styles["BodyText"],
            )
        )
        status = str(t.get("evidence_status", "missing")).strip().lower()
        explanation = str(t.get("explanation", "")).strip()
        citations = t.get("_final_citations", t.get("citations", []))
        if not isinstance(citations, list):
            citations = []
        if status == "found" and explanation:
            story.append(Paragraph(explanation, styles["BodyText"]))
            if citations:
                story.append(Paragraph("Citations:", styles["BodyText"]))
                for c in citations[:3]:
                    if isinstance(c, dict):
                        ident = str(c.get("identifier", "")).strip() or str(c.get("pmid", "")).strip() or str(c.get("pmcid", "")).strip() or str(c.get("doi", "")).strip()
                        title = str(c.get("title", "")).strip() or "Source"
                        year = str(c.get("year", "")).strip()
                        url = str(c.get("source_url", c.get("url", ""))).strip()
                        row = f"- {ident} | {title} {f'({year})' if year else ''}"
                        if url:
                            row += f" | {url}"
                        story.append(Paragraph(row, styles["BodyText"]))
        else:
            prov = t.get("search_provenance", {}) if isinstance(t.get("search_provenance", {}), dict) else {}
            query_txt = str(prov.get("query", "")).strip()
            prov_txt = f" Query logged: {query_txt}" if query_txt else ""
            story.append(
                Paragraph(
                    "Explanation withheld (no evidence retrieved)." + prov_txt,
                    styles["BodyText"],
                )
            )
        story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()


def run_evidence_builder(max_traits: int | None = None, trait_id: str | None = None):
    cmd = [sys.executable, "scripts/build_evidence_corpus.py"]
    if max_traits is not None:
        cmd.extend(["--max_traits", str(int(max_traits))])
    if trait_id:
        cmd.extend(["--trait_id", str(trait_id)])
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def is_dev_mode():
    env_flag = str(os.environ.get("DEV_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}
    secret_flag = False
    try:
        secret_flag = bool(st.secrets.get("DEV_MODE", False))
    except Exception:
        secret_flag = False
    return env_flag or secret_flag


def genotype_hash_from_bytes(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes or b"").hexdigest()


def persist_uploaded_file(file_bytes: bytes, genotype_hash: str) -> Path:
    CACHE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_UPLOAD_DIR / f"{genotype_hash}.txt"
    if not path.exists():
        path.write_bytes(file_bytes or b"")
    return path


@st.cache_resource(show_spinner=False)
def cached_load_study_packs(dir_path: str):
    return load_study_packs(dir_path)


@st.cache_data(show_spinner=False)
def cached_parse_genotype(genotype_hash: str):
    path = CACHE_UPLOAD_DIR / f"{genotype_hash}.txt"
    if not path.exists():
        return []
    rows = parse_genotype_file(str(path))
    return rows if isinstance(rows, list) else []


@st.cache_data(show_spinner=False)
def cached_build_base_report(genotype_hash: str, categories_selected: tuple[str, ...], include_optional_liver: bool):
    path = CACHE_UPLOAD_DIR / f"{genotype_hash}.txt"
    if not path.exists():
        return {"summary": {"num_traits_found": 0, "categories": []}, "traits": []}
    report = build_prs_report_from_upload(
        genotype_path=str(path),
        categories_selected=list(categories_selected),
        include_optional_liver=bool(include_optional_liver),
        red_flag_only=False,
    )
    return normalize_report(report)


@st.cache_data(show_spinner=False, ttl=604800)
def cached_retrieve_trait_evidence(trait_payload: dict):
    t = trait_payload if isinstance(trait_payload, dict) else {}
    return retrieve_trait_evidence(
        t,
        min_citations=2,
        min_snippets=2,
        max_results=8,
        retries=1,
    )


@st.cache_data(show_spinner=False, ttl=604800)
def cached_generate_trait_explanation(trait_payload: dict, evidence_snippets: list[dict], mode_key: str):
    return generate_trait_explanation(
        trait_payload if isinstance(trait_payload, dict) else {},
        evidence_snippets if isinstance(evidence_snippets, list) else [],
        mode_key,
        llm_fn=generate_with_fallback_model,
    )


def build_category_evidence_bundle(category_name: str, traits_in_cat: list[dict]) -> tuple[list[dict], list[dict]]:
    traits_payload = []
    evidence_payload = []
    max_traits = 8
    max_snippets_per_trait = 2
    char_budget = 2800
    used_chars = 0

    for t in [x for x in (traits_in_cat or []) if isinstance(x, dict)][:max_traits]:
        traits_payload.append(
            {
                "trait_name": str(t.get("trait_name", "")).strip(),
                "gene": str(t.get("gene", "")).strip(),
                "rsid": str(t.get("rsid", "")).strip(),
                "genotype": str(t.get("user_genotype", "")).strip(),
                "signal_bucket": str(t.get("effect_level", t.get("bucket", ""))).strip(),
                "evidence_strength": str(t.get("evidence_strength", "")).strip(),
                "coverage": float(t.get("coverage", 0.0) or 0.0) if isinstance(t.get("coverage", 0.0), (int, float)) else 0.0,
            }
        )

        snippets = t.get("_evidence_snippets", [])
        if not isinstance(snippets, list):
            snippets = []
        for s in snippets[:max_snippets_per_trait]:
            if not isinstance(s, dict):
                continue
            citation = s.get("citation", s if isinstance(s, dict) else {})
            row = {
                "title": str((citation or {}).get("title", s.get("title", ""))).strip(),
                "year": str((citation or {}).get("year", s.get("year", ""))).strip(),
                "journal": str((citation or {}).get("journal", "")).strip(),
                "pmid": str((citation or {}).get("pmid", s.get("pmid", ""))).strip(),
                "url": str((citation or {}).get("url", s.get("url", ""))).strip(),
                "snippet_text": _clip_text(s.get("text", s.get("snippet_text", "")), 220),
            }
            projected = used_chars + len(row["snippet_text"])
            if projected > char_budget:
                break
            used_chars = projected
            evidence_payload.append(row)

    return traits_payload, evidence_payload


def _category_summary_fallback(category_name: str, traits_payload: list[dict], evidence_payload: list[dict], mode_key: str) -> str:
    n_traits = len(traits_payload or [])
    n_ev = len(evidence_payload or [])
    if n_ev == 0:
        return (
            f"{category_name}: evidence is currently limited in this category. "
            "Some traits have limited evidence available right now. "
            "This category's evidence corpus is still loading. Trait cards are shown below. "
            "Not medical advice."
        )
    if mode_key == "doctor":
        return (
            f"{category_name} overview: {n_traits} traits were processed with {n_ev} supporting snippets. "
            "Evidence is limited for some traits, so interpretation remains cautious and non-diagnostic. "
            "Not medical advice."
        )
    return (
        f"{category_name} overview: we found {n_traits} traits with some supporting research snippets. "
        "Some traits have limited evidence available right now. "
        "These points are educational and not medical advice."
    )


@st.cache_data(show_spinner=False, ttl=86400)
def cached_generate_category_summary(
    category_name: str,
    traits_payload: list[dict],
    evidence_payload: list[dict],
    mode_key: str,
):
    if not evidence_payload:
        return _category_summary_fallback(category_name, traits_payload, evidence_payload, mode_key)

    system_prompt = (
        "You are VivaGene. You must only write statements supported by the provided evidence snippets. "
        "If a claim is not supported, you MUST say evidence is limited and do not infer. "
        "No medical advice, no diagnosis, no treatment. Use cautious language: may/might/could. "
        "Always attach citations like [PMID:12345678] to any scientific claim."
    )
    user_prompt = (
        f"Mode: {'Patient' if mode_key == 'patient' else 'Doctor'}\n"
        f"Category: {category_name}\n"
        f"Trait summary table (compact JSON):\n{json.dumps(traits_payload, ensure_ascii=False)}\n\n"
        f"Evidence snippets (compact):\n{json.dumps(evidence_payload, ensure_ascii=False)}\n\n"
        f"Write a category summary that explains the main takeaways in {'patient' if mode_key == 'patient' else 'doctor'} style. "
        "Keep it understandable. Only include claims supported by snippets and cite them.\n"
        "Output format: Start with 1 sentence overview, then 4-6 bullet points, end with one line saying not medical advice."
    )
    raw = generate_with_fallback_messages(system_prompt, user_prompt)
    txt = str(raw or "").strip()
    if not txt:
        return _category_summary_fallback(category_name, traits_payload, evidence_payload, mode_key)
    if "not medical advice" not in txt.lower():
        txt = txt.rstrip() + "\n\nNot medical advice."
    if not evidence_payload and "limited evidence" not in txt.lower():
        txt = txt.rstrip() + "\nSome traits have limited evidence available right now."
    return txt


def _normalize_citations_from_trait(trait_row: dict) -> list[dict]:
    out = []
    seen = set()
    for c in trait_row.get("sources", trait_row.get("citations", [])) if isinstance(trait_row.get("sources", trait_row.get("citations", [])), list) else []:
        if not isinstance(c, dict):
            continue
        pmid = str(c.get("pmid", "")).strip()
        title = str(c.get("title", "")).strip() or "Study"
        key = pmid or title
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"pmid": pmid, "title": title})
    return out[:3]


def _fallback_trait_lifestyle(trait_row: dict) -> dict:
    effect = str(trait_row.get("effect_label", "")).strip() or str(trait_row.get("signal_bucket", "")).strip() or "a trait tendency"
    category = str(trait_row.get("category", "")).strip().lower() or "daily patterns"
    return {
        "one_liner": f"This signal may relate to {effect} in day-to-day {category} patterns.",
        "bullets": [
            f"May show subtle differences linked to {effect}.",
            "Might vary based on sleep, stress, environment, and routine.",
            "Could change over time and is not deterministic.",
        ],
        "citations": _normalize_citations_from_trait(trait_row),
    }


@st.cache_data(show_spinner=False, ttl=86400)
def cached_generate_category_trait_explainers(category_name: str, mode_key: str, traits_payload: list[dict]):
    if not isinstance(traits_payload, list) or not traits_payload:
        return {}
    user_prompt = (
        f"Mode: {'Patient' if mode_key == 'patient' else 'Doctor'}\n"
        f"Category: {category_name}\n"
        f"Trait summary table (compact JSON):\n{json.dumps(traits_payload, ensure_ascii=False)}\n\n"
        "For each trait_id, return JSON only:\n"
        "{ \"traits\": { \"TRAIT_ID\": { \"one_liner\": \"...\", \"bullets\": [\"...\",\"...\",\"...\"], "
        "\"citations\": [{\"pmid\":\"...\",\"title\":\"...\"}] } } }\n"
        "Rules:\n"
        "- 8th-grade language for patient mode.\n"
        "- Doctor mode may be more scientific.\n"
        "- 1 short one-liner + 1-3 lifestyle bullets per trait.\n"
        "- Use only provided citations. If no citations, include low-evidence wording and keep citations empty.\n"
        "- No medical advice, diagnosis, treatment, or prescriptions.\n"
    )
    raw = generate_with_fallback_messages(LLM_SAFETY_SYSTEM, user_prompt)
    obj = _safe_json_obj(raw)
    traits = obj.get("traits", {}) if isinstance(obj.get("traits", {}), dict) else {}
    out = {}
    for t in traits_payload:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("trait_id", "")).strip()
        model_row = traits.get(tid, {}) if isinstance(traits, dict) else {}
        if not isinstance(model_row, dict):
            model_row = {}
        one_liner = " ".join(str(model_row.get("one_liner", "")).split()).strip()
        bullets = model_row.get("bullets", [])
        if not isinstance(bullets, list):
            bullets = []
        bullets = [" ".join(str(b).split()).strip() for b in bullets if str(b).strip()][:3]
        cits = model_row.get("citations", [])
        if not isinstance(cits, list):
            cits = []
        allowed = {str(c.get("pmid", "")).strip() for c in _normalize_citations_from_trait(t) if str(c.get("pmid", "")).strip()}
        valid_cits = []
        for c in cits:
            if not isinstance(c, dict):
                continue
            pmid = str(c.get("pmid", "")).strip()
            title = str(c.get("title", "")).strip() or "Study"
            if pmid and allowed and pmid not in allowed:
                continue
            valid_cits.append({"pmid": pmid, "title": title})
        if not one_liner:
            fb = _fallback_trait_lifestyle(t)
            one_liner = fb["one_liner"]
            bullets = fb["bullets"]
            valid_cits = fb["citations"]
        out[tid] = {"one_liner": one_liner, "bullets": bullets, "citations": valid_cits}
    return out


def generate_category_summary(category_name: str, traits: list[dict], mode: str) -> dict:
    mode_key = "doctor" if str(mode or "").lower().startswith("doctor") else "patient"
    traits_payload = []
    for t in traits if isinstance(traits, list) else []:
        if not isinstance(t, dict):
            continue
        traits_payload.append(
            {
                "trait_id": str(t.get("trait_id", "")).strip(),
                "trait_name": str(t.get("trait_name", "")).strip(),
                "effect_label": str(t.get("effect_label", "")).strip(),
                "signal_bucket": str(t.get("effect_level", t.get("bucket", ""))).strip(),
                "evidence_strength": str(t.get("evidence_strength", "")).strip(),
                "coverage": float(t.get("coverage", 0.0) or 0.0) if isinstance(t.get("coverage", 0.0), (int, float)) else 0.0,
                "gene": str(t.get("gene", "")).strip(),
                "rsids": [str(t.get("rsid", "")).strip()],
                "citations": _normalize_citations_from_trait(t),
            }
        )
    _, evidence_payload = build_category_evidence_bundle(category_name, traits)
    txt = cached_generate_category_summary(category_name, traits_payload, evidence_payload, mode_key)
    lines = [line.strip("- ").strip() for line in str(txt or "").splitlines() if line.strip()]
    bullets = [line for line in lines if len(line) > 2][:6]
    if not bullets:
        bullets = [
            f"{category_name} traits may influence everyday patterns, but effects can vary by person.",
            "Some traits have limited evidence available right now.",
            "Insights below are educational and should be interpreted cautiously.",
            "Genes are one factor among sleep, stress, environment, and habits.",
        ]
    if len(bullets) < 3:
        bullets = bullets + [
            "Some traits have limited evidence available right now.",
            "These insights are educational and not diagnostic.",
        ]
    cits = []
    seen = set()
    for t in traits_payload:
        for c in t.get("citations", []):
            pmid = str(c.get("pmid", "")).strip()
            title = str(c.get("title", "")).strip() or "Study"
            key = pmid or title
            if key and key not in seen:
                seen.add(key)
                cits.append({"pmid": pmid, "title": title})
    if mode_key == "doctor":
        cits = cits[:6]
    else:
        cits = cits[:3]
    return {"category": category_name, "bullets": bullets[:6], "citations": cits}


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

# Prefer Streamlit secrets in Cloud, fallback to environment variables locally.
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = get_secret("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
OPENROUTER_CLIENT = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    if (OpenAI is not None and OPENROUTER_API_KEY)
    else None
)


def get_llm_client(show_error: bool = True):
    provider = get_secret("LLM_PROVIDER", "openrouter").lower()
    model_default = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"
    model = get_secret("LLM_MODEL", model_default)
    openrouter_key = get_secret("OPENROUTER_API_KEY", "")
    openai_key = get_secret("OPENAI_API_KEY", "")
    client = None
    active_provider = provider

    if OpenAI is None:
        st.error("OpenAI client library is not available. Install dependencies from requirements.txt.")
        return None, "", active_provider

    if provider == "openrouter":
        if openrouter_key:
            client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        elif openai_key:
            active_provider = "openai"
            client = OpenAI(api_key=openai_key)
            model = get_secret("LLM_MODEL", "gpt-4o-mini")
    else:
        if openai_key:
            client = OpenAI(api_key=openai_key)
            model = get_secret("LLM_MODEL", "gpt-4o-mini")
        elif openrouter_key:
            active_provider = "openrouter"
            client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
            model = get_secret("LLM_MODEL", "openai/gpt-4o-mini")

    if client is None and show_error and not st.session_state.get("_llm_missing_warned", False):
        st.error(
            "LLM credentials are missing. Set Streamlit Cloud secrets: OPENROUTER_API_KEY (preferred) "
            "or OPENAI_API_KEY. Optional: LLM_PROVIDER and LLM_MODEL."
        )
        st.session_state["_llm_missing_warned"] = True
    return client, model, active_provider

# Hide detailed tracebacks in UI; show friendly errors instead.
st.set_option("client.showErrorDetails", False)

def generate_ai_summary(report: dict) -> str:
    client, _, _ = get_llm_client()
    if not client:
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
    client, model, _ = get_llm_client(show_error=False)
    if not client:
        return ""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
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
                temperature=0.5,
                timeout=30,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "503" in msg) and attempt < 2:
                time.sleep(1.2 * (attempt + 1))
                continue
            return ""
    return ""


def generate_with_fallback_messages(system_prompt: str, user_prompt: str) -> str:
    client, model, _ = get_llm_client(show_error=False)
    if not client:
        return ""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": str(system_prompt or "")},
                    {"role": "user", "content": str(user_prompt or "")},
                ],
                temperature=0.3,
                timeout=30,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "503" in msg) and attempt < 2:
                time.sleep(1.2 * (attempt + 1))
                continue
            return ""
    return ""


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

    client, _, _ = get_llm_client()
    if not client:
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


def _collect_report_sources(report: dict, limit: int = 10) -> list[str]:
    rep = normalize_report(report if isinstance(report, dict) else {})
    lines = []
    seen = set()
    for t in rep.get("traits", []):
        if not isinstance(t, dict):
            continue
        source_list = t.get("sources", t.get("citations", []))
        if not isinstance(source_list, list):
            continue
        for c in source_list:
            if not isinstance(c, dict):
                continue
            pmid = str(c.get("pmid", "")).strip()
            title = str(c.get("title", "")).strip() or "Study"
            year = str(c.get("year", "")).strip()
            key = pmid or title
            if not key or key in seen:
                continue
            seen.add(key)
            ident = f"PMID:{pmid}" if pmid else title
            lines.append(f"- {ident} — {title}{f' ({year})' if year else ''}")
            if len(lines) >= limit:
                return lines
    return lines


def generate_lifestyle_plan(report: dict, mode: str = "patient") -> str:
    rep = normalize_report(report if isinstance(report, dict) else {})
    traits = [t for t in rep.get("traits", []) if isinstance(t, dict)][:25]
    mode_key = str(mode or "patient").strip().lower()
    clinician_line = "If you have concerns, discuss this with a clinician or a genetic counselor."
    safe_system = (
        "You are a cautious lifestyle educator. No medical advice, no diagnosis, no treatment, "
        "no supplement or medication instructions. Provide only gentle, general ideas."
    )
    context = {
        "mode": mode_key,
        "traits": [
            {
                "trait_name": t.get("trait_name", ""),
                "category": t.get("category", ""),
                "bucket": t.get("effect_level", t.get("bucket", "")),
                "summary": t.get("_final_summary", ""),
            }
            for t in traits
        ],
    }
    prompt = f"""
{safe_system}

Generate a short plan with sections:
Sleep, Focus, Nutrition, Fitness, Liver (only if present).
Each section: 3-5 gentle ideas. Cautious language only.
Never diagnose or prescribe.
End with: "{clinician_line}"
Mode: {mode_key}
Context JSON:
{json.dumps(context, ensure_ascii=False)}
"""
    text = generate_with_fallback_model(prompt)
    if not text.strip():
        text = build_local_lifestyle_plan(rep)
    if mode_key.startswith("doctor"):
        src = _collect_report_sources(rep, limit=8)
        if src:
            text = text.rstrip() + "\n\nSuggested reading\n" + "\n".join(src)
    return text


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
        --bg-primary: #0b0f19;
        --bg-secondary: #111827;
        --bg-elevated: #161f2e;
        --border-subtle: #1f2937;
        --text-primary: #e5e7eb;
        --text-secondary: #9ca3af;
        --accent: #3b82f6;
        --accent-muted: #2563eb;
        --radius-sm: 6px;
        --radius-md: 12px;
        --radius-lg: 18px;
        --transition-base: 220ms cubic-bezier(.4,0,.2,1);

        /* Backward-compatible aliases for existing selectors */
        --color-bg-primary: var(--bg-primary);
        --color-bg-secondary: var(--bg-secondary);
        --color-border: var(--border-subtle);
        --color-accent: var(--accent);
        --color-accent-muted: var(--accent-muted);
        --color-text-primary: var(--text-primary);
        --color-text-secondary: var(--text-secondary);
        --radius-sm: 6px;
        --radius-md: 12px;
        --radius-lg: 18px;
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --spacing-xl: 40px;
        --transition-fast: 120ms cubic-bezier(.4,0,.2,1);
        --transition-base: 220ms cubic-bezier(.4,0,.2,1);
        --shadow-soft: 0 8px 24px rgba(0,0,0,0.35);
        --shadow-soft-hover: 0 12px 30px rgba(0,0,0,0.45);
        --color-success-bg: #10251b;
        --color-success-text: #9cd9b4;
        --color-warn-bg: #2a2112;
        --color-warn-text: #f2d39a;
    }

    html { scroll-behavior: smooth; }
    * { box-sizing: border-box; }

    body {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        font-size: 15px;
        line-height: 1.6;
        font-weight: 400;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: var(--bg-primary);
    }

    .main .block-container {
        padding-top: 0;
        padding-bottom: var(--spacing-xl);
        max-width: 1200px;
    }
    h1 { font-size: 32px; line-height: 1.2; font-weight: 600; margin: 0 0 var(--spacing-md); color: var(--color-text-primary); }
    h2 { font-size: 24px; line-height: 1.3; font-weight: 600; margin: 0 0 var(--spacing-md); color: var(--color-text-primary); }
    h3 { font-size: 18px; line-height: 1.4; font-weight: 500; margin: 0 0 var(--spacing-md); color: var(--color-text-primary); }
    p { margin: 0 0 var(--spacing-md); }

    .top-nav-shell {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: var(--color-bg-secondary);
        border-bottom: 1px solid var(--color-border);
        margin: 0 -1rem var(--spacing-md);
        padding: var(--spacing-sm) var(--spacing-md);
    }
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: var(--spacing-sm) 0;
    }
    .top-nav-left, .nav-logo-container, .nav-button-container {
        display: flex;
        align-items: center;
        gap: var(--spacing-md);
    }
    .logo-wrap { width: 52px; height: 52px; display: inline-flex; align-items: center; justify-content: center; }
    .logo-wrap svg { width: 48px; height: 48px; }
    .vg-mark-wrap { height: 200px; display: flex; align-items: center; justify-content: center; }
    .logo-title {
        font-weight: 600;
        letter-spacing: 0.08em;
        font-size: 16px;
        color: var(--color-text-primary);
    }
    .logo-sub {
        font-size: 13px;
        color: var(--color-text-secondary);
    }

    .hero-band {
        background: linear-gradient(
          180deg,
          #0b0f19 0%,
          #0e1422 100%
        );
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--spacing-xl);
        margin: 0 0 var(--spacing-xl);
        box-shadow: var(--shadow-soft);
    }
    @media (max-width: 800px) {
        .hero-band { padding: var(--spacing-lg); }
    }

    .hero-grid {
        display: flex;
        gap: var(--spacing-xl);
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }

    .hero-chip {
        display: inline-block;
        padding: var(--spacing-xs) var(--spacing-sm);
        border-radius: var(--radius-sm);
        border: 1px solid var(--color-border);
        background: var(--bg-elevated);
        color: var(--color-accent);
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: var(--spacing-sm);
    }

    .hero-title {
        font-size: 32px;
        line-height: 1.2;
        font-weight: 600;
        margin-bottom: var(--spacing-md);
        color: var(--color-text-primary);
    }

    .hero-title span {
        color: var(--color-accent);
    }

    .hero-sub {
        font-size: 15px;
        max-width: 560px;
        color: var(--color-text-secondary);
        margin-bottom: var(--spacing-md);
    }

    .hero-note {
        font-size: 13px;
        color: var(--color-text-secondary);
        max-width: 540px;
    }

    .hero-dna-card {
        min-width: 260px;
        max-width: 320px;
        border-radius: var(--radius-md);
        background: var(--color-bg-secondary);
        border: 1px solid var(--color-border);
        padding: var(--spacing-lg);
        transition: all var(--transition-base);
        box-shadow: var(--shadow-soft);
    }

    .hero-dna-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-soft-hover);
    }

    .hero-dna-title {
        font-size: 18px;
        font-weight: 500;
        margin-bottom: var(--spacing-xs);
        color: var(--color-text-primary);
    }

    .hero-dna-sub {
        font-size: 13px;
        color: var(--color-text-secondary);
        margin-bottom: var(--spacing-sm);
    }
    .hero-dna-card img {
        max-height: 240px;
        object-fit: contain;
        margin-top: var(--spacing-sm);
        width: 100%;
        display: block;
    }
    .mark-card img {
        display: block;
        margin-top: 10px;
        border-radius: 12px;
        max-width: 100%;
    }

    .section-title {
        font-size: 24px;
        line-height: 1.3;
        font-weight: 600;
        margin-bottom: var(--spacing-md);
        color: var(--color-text-primary);
    }

    .section-sub {
        font-size: 15px;
        color: var(--color-text-secondary);
        margin-bottom: var(--spacing-md);
    }
    .home-section { margin: 0 0 var(--spacing-xl); }

    .feature-row {
        display: flex;
        gap: var(--spacing-md);
        flex-wrap: wrap;
        margin-top: var(--spacing-md);
    }

    .feature-card, .how-step, .bio-card, .who-card, .roadmap-card, .newsletter-modal, .section-light {
        flex: 1 1 220px;
        background: var(--color-bg-secondary);
        border-radius: var(--radius-md);
        border: 1px solid var(--color-border);
        padding: var(--spacing-lg);
        box-shadow: var(--shadow-soft);
        color: var(--color-text-primary);
        transition: all var(--transition-base);
    }
    .feature-card:hover, .how-step:hover, .bio-card:hover, .who-card:hover, .roadmap-card:hover, .newsletter-modal:hover, .section-light:hover, .chat-box:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-soft-hover);
    }
    .newsletter-title { font-size: 18px; font-weight: 500; margin-bottom: var(--spacing-xs); color: var(--color-text-primary); }
    .newsletter-sub { font-size: 13px; color: var(--color-text-secondary); margin-bottom: var(--spacing-sm); }
    .small-label, .feature-label, .how-label, .roadmap-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--color-text-secondary);
        font-weight: 500;
        margin-bottom: var(--spacing-xs);
    }
    .feature-title, .roadmap-title, .who-title, .bio-name, .how-step-title {
        font-size: 18px;
        line-height: 1.4;
        font-weight: 500;
        margin-bottom: var(--spacing-xs);
        color: var(--color-text-primary);
    }
    .feature-body, .bio-role, .hero-note {
        font-size: 15px;
        color: var(--color-text-secondary);
    }
    .bio-role { margin-bottom: var(--spacing-sm); }
    .who-row, .how-row { display: flex; gap: var(--spacing-md); flex-wrap: wrap; margin-top: var(--spacing-md); }
    .who-card, .how-step { flex: 1 1 220px; }
    .how-step-number {
        display: inline-block;
        font-size: 13px;
        font-weight: 600;
        color: var(--color-accent);
        margin-bottom: var(--spacing-xs);
    }
    .how-band {
        margin-top: var(--spacing-xl);
        border-radius: var(--radius-lg);
        background: var(--color-bg-secondary);
        border: 1px solid var(--color-border);
        padding: var(--spacing-lg);
        box-shadow: var(--shadow-soft);
    }
    .how-title {
        font-size: 24px;
        line-height: 1.3;
        font-weight: 600;
        margin-bottom: var(--spacing-md);
        color: var(--color-text-primary);
    }
    .pipeline-diagram { margin-top: var(--spacing-lg); border-top: 1px solid var(--color-border); padding-top: var(--spacing-md); }
    .pipeline-flow { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: var(--spacing-sm); }
    .pipe-step { border: 1px solid var(--color-border); border-radius: var(--radius-sm); padding: var(--spacing-sm); font-size: 13px; background: var(--color-bg-primary); }
    .pipe-icon { display: none; }
    @media (max-width: 900px) { .pipeline-flow { grid-template-columns: 1fr 1fr; } }

    .roadmap-card { background: var(--color-bg-secondary); }
    .site-footer {
        margin-top: var(--spacing-xl);
        padding: var(--spacing-md) 0 var(--spacing-sm);
        border-top: 1px solid var(--color-border);
        font-size: 13px;
        color: var(--color-text-secondary);
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: var(--spacing-sm);
    }
    .site-footer a { color: var(--color-accent); text-decoration: none; }
    .site-footer a:hover { text-decoration: underline; }

    .stButton > button {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--color-border) !important;
        background: var(--color-bg-secondary) !important;
        color: var(--color-text-primary) !important;
        transition: all var(--transition-base);
        box-shadow: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-soft-hover) !important;
        border-color: var(--color-accent) !important;
    }
    .stButton > button:disabled {
        background: var(--bg-elevated) !important;
        color: var(--color-text-secondary) !important;
        border-color: var(--color-border) !important;
        cursor: not-allowed !important;
        opacity: 0.75;
        transform: none !important;
        box-shadow: none !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--color-border) !important;
        transition: all var(--transition-fast);
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stSelectbox"] div:focus-within {
        outline: none !important;
        border-color: var(--color-accent) !important;
        box-shadow: 0 0 0 2px var(--color-accent-muted) !important;
    }

    .nav-button-container .stButton > button {
        border: 1px solid transparent !important;
        background: transparent !important;
        color: var(--color-text-primary) !important;
        padding: var(--spacing-xs) var(--spacing-sm) !important;
    }
    .nav-button-container .stButton > button:hover {
        border-color: var(--color-border) !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .scroll-cue { display: none; }
    .page-wrap { animation: none; }
    .evidence-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--color-border);
        font-size: 13px;
        margin-bottom: var(--spacing-xs);
    }
    .evidence-badge.found {
        color: var(--color-success-text);
        background: var(--color-success-bg);
    }
    .evidence-badge.missing {
        color: var(--color-warn-text);
        background: var(--color-warn-bg);
    }
    .skeleton {
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: var(--spacing-lg);
        background: var(--color-bg-secondary);
        color: var(--color-text-secondary);
        margin-bottom: var(--spacing-md);
    }
    .founder-highlights {
        margin-top: var(--spacing-sm);
        font-size: 13px;
        color: var(--color-text-secondary);
    }
    /* White results panel for trait output */
    .results-panel {
      background: #ffffff;
      border: 1px solid rgba(15, 23, 42, 0.10);
      border-radius: 18px;
      padding: 18px 18px 14px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.18);
      max-width: 1100px;
      margin-bottom: var(--spacing-md);
    }

    /* Scroll area INSIDE the panel */
    .results-scroll {
      max-height: 620px;
      overflow-y: auto;
      padding-right: 10px;
      padding-bottom: 6px;
    }

    /* Nice scrollbar */
    .results-scroll::-webkit-scrollbar { width: 10px; }
    .results-scroll::-webkit-scrollbar-thumb {
      background: rgba(15,23,42,0.18);
      border-radius: 999px;
    }
    .results-scroll::-webkit-scrollbar-track {
      background: rgba(15,23,42,0.06);
      border-radius: 999px;
    }

    /* Force readable text inside the white panel */
    .results-panel, .results-panel * {
      color: #111827 !important;
    }

    /* Links inside panel */
    .results-panel a {
      color: #2563eb !important;
      text-decoration: none;
    }
    .results-panel a:hover {
      text-decoration: underline;
    }

    /* Trait card inside panel */
    .trait-card {
      background: #ffffff;
      border: 1px solid rgba(15,23,42,0.10);
      border-radius: 14px;
      padding: 14px 14px 12px;
      margin-bottom: 12px;
      box-shadow: 0 6px 16px rgba(15,23,42,0.08);
    }
    .trait-card, .trait-card * { color: #111827 !important; }
    .trait-title { font-weight: 700; font-size: 1.02rem; margin-bottom: 4px; }
    .trait-meta { color: #6b7280 !important; font-size: 0.86rem; margin-bottom: 8px; }
    .trait-body { font-size: 0.94rem; line-height: 1.45; margin-top: 6px; }
    .trait-sources { margin-top: 10px; font-size: 0.90rem; }
    .trait-sources ul { margin: 6px 0 0 18px; }
    .trait-sources li { margin: 4px 0; }
    .trait-muted { color: #6b7280 !important; }
    .category-summary-box {
      background: #ffffff;
      border: 1px solid rgba(15,23,42,0.10);
      border-radius: 12px;
      padding: 14px 16px;
      margin-bottom: 12px;
      max-height: 180px;
      overflow-y: auto;
    }
    .category-summary-title {
      font-size: 0.95rem;
      font-weight: 700;
      margin-bottom: 6px;
      color: #111827 !important;
    }
    .category-summary-body {
      font-size: 0.9rem;
      line-height: 1.45;
      color: #111827 !important;
      white-space: pre-wrap;
    }

    /* Small “chip” label */
    .chip {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      border: 1px solid rgba(15,23,42,0.12);
      background: rgba(37,99,235,0.08);
      color: #1d4ed8 !important;
      margin-left: 8px;
    }
    .trait-missing {
        font-size: 12px;
        color: #9a3412;
        margin: 4px 0 6px;
    }
    .trait-query {
        font-size: 12px;
        color: #6b7280;
        margin: 6px 0;
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
safe_html("<div class='top-nav-shell'>")
with st.container():
    col_left, col_right = st.columns([1.4, 1.6])

    with col_left:
        logo_svg = load_svg_safe(str(VIVAGENE_LOGO_PATH))
        if logo_svg:
            try:
                safe_html(f"<div class='brand-logo logo-wrap'>{logo_svg}</div>")
            except Exception:
                st.image("genai_logo.png", width=52)
                if not st.session_state.get("_logo_fallback_warned", False):
                    st.warning("Logo SVG rendering fallback active.")
                    st.session_state["_logo_fallback_warned"] = True
        else:
            st.image("genai_logo.png", width=52)
            if not st.session_state.get("_logo_fallback_warned", False):
                st.warning("Logo SVG could not be loaded. Using fallback icon.")
                st.session_state["_logo_fallback_warned"] = True

        safe_html(
            """
            <div class="nav-logo-container">
                <div>
                    <div class="logo-title">VIVAGENE</div>
                    <div class="logo-sub">Evidence-first genomics, clearly explained.</div>
                </div>
            </div>
            """
        )

    with col_right:
        safe_html("<div class='nav-button-container'>")
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

        safe_html("</div>")
safe_html("</div>")

page = st.session_state.active_page

# Wrapper to trigger CSS animation on page change
begin_page_wrap()
              
           
# ---------- Newsletter state ----------
if "report_processing" not in st.session_state:
    st.session_state.report_processing = False
if "evidence_processing" not in st.session_state:
    st.session_state.evidence_processing = False


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
        f"""
        <div class="hero-band">
          <div class="hero-grid">
            <div>
              <div class="hero-chip">Evidence-first genomics</div>
              <div class="hero-title">
                Genomics reporting for <span>clear interpretation.</span>
              </div>
              <div class="hero-sub">
                VivaGene converts raw genotype files into structured trait summaries with explicit confidence, coverage, and evidence links.
              </div>
              <div class="hero-note">
                Current support focuses on Nutrition, Neurobehavior, and Fitness traits. Output is educational and non-diagnostic.
              </div>
            </div>
            <div class="hero-dna-card">
              <div class="hero-dna-title">VivaGene Mark</div>
              <div class="hero-dna-sub">
                Static visual identifier for the report workspace.
              </div>
              {render_vg_helix_svg()}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary CTAs under hero
    cta_col1, cta_col2 = st.columns([0.4, 0.4])
    with cta_col1:
        if st.button("Create your VivaGene report", key="hero_start_report"):
            st.session_state.active_page = "Upload & Report"
    with cta_col2:
        if st.button("Open Lifestyle Q&A", key="hero_chatbot"):
            st.session_state.active_page = "Lifestyle Q&A"

    # Why VivaGene exists
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

    # Why choose VivaGene
    st.markdown('<div class="section-title">Why choose VivaGene?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A structured interface for educational genomics interpretation with evidence boundaries.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-row">
          <div class="feature-card">
            <div class="feature-label">Founder story</div>
            <div class="feature-title">Built as an independent research project</div>
            <div class="feature-body">
              Developed by Anjali Nalla to support careful communication of genomic trait tendencies for education.
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-label">Transparent</div>
            <div class="feature-title">Every trait is traceable</div>
            <div class="feature-body">
              Trait outputs are linked to rsIDs, genes, confidence, and citations so each summary has a visible evidence path.
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-label">Lifestyle focus</div>
            <div class="feature-title">DNA as one piece of the puzzle</div>
            <div class="feature-body">
              Trait output is framed as probabilistic tendencies and does not provide diagnosis or treatment guidance.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # How it works
    st.markdown(
        """
        <div class="how-band">
          <div class="how-label">How it works</div>
          <div class="how-title">From raw file to a clear, citation-aware trait report.</div>
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
          <div class="pipeline-diagram">
            <div class="pipeline-flow">
              <div class="pipe-step"><div class="pipe-icon">1</div><div>Upload genotype</div></div>
              <div class="pipe-step"><div class="pipe-icon">2</div><div>Match curated variants</div></div>
              <div class="pipe-step"><div class="pipe-icon">3</div><div>Bind to evidence</div></div>
              <div class="pipe-step"><div class="pipe-icon">4</div><div>Generate plain-language summary</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Secondary CTA near How it works
    spacer_left, cta_mid, spacer_right = st.columns([0.35, 0.3, 0.35])
    with cta_mid:
        if st.button("Start your report", key="how_start_report"):
            st.session_state.active_page = "Upload & Report"

    # Founder profile / Who this is for / Roadmap
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1.1])

    with col_a:
        st.markdown('<div class="section-title">Founder profile</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="bio-card">
              <div class="bio-name">Anjali Nalla</div>
              <div class="bio-role">Student researcher in computational genomics</div>
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
        <div class="founder-highlights">
          <strong>Founder highlights</strong><br/>
          · Johns Hopkins affiliated research in rare disease biology<br/>
          · Clinical volunteering on a neuro floor and in hospice settings<br/>
          · Internships that combine medicine, psychology, and research<br/>
          · Formal lab safety and benchwork certifications<br/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Safety, privacy, and limitations (Home)
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

    # FAQ (Home - short)
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

        categories_selected = ["Neurobehavior", "Nutrition", "Fitness", "Liver"]
        include_optional_liver = True
        explain_mode = st.radio("Explanation style", ["Patient (simple)", "Doctor (technical)"], horizontal=True)
        st.session_state["explain_mode"] = explain_mode
        max_refresh_traits = st.number_input(
            "Evidence refresh batch size",
            min_value=1,
            max_value=500,
            value=80,
            step=1,
            help="How many traits to query from Europe PMC in one refresh run.",
        )
        refresh_all_evidence = st.button(
            "Processing..." if st.session_state.evidence_processing else "Build/Refresh Evidence Corpus (Europe PMC)",
            disabled=bool(st.session_state.report_processing or st.session_state.evidence_processing),
        )
        if refresh_all_evidence:
            st.session_state.evidence_processing = True
            progress = st.progress(5, text="Initializing Europe PMC refresh...")
            with st.spinner("Building local evidence corpus..."):
                progress.progress(45, text="Querying and writing local corpus...")
                rc, out, err = run_evidence_builder(max_traits=int(max_refresh_traits))
            progress.progress(100, text="Evidence corpus refresh complete.")
            if rc == 0:
                st.success("Evidence corpus refresh completed.")
            else:
                st.warning("Evidence refresh finished with warnings/errors.")
            with st.expander("Evidence build log"):
                if out:
                    st.code(out, language="json")
                if err:
                    st.code(err, language="text")
            st.session_state.evidence_processing = False
            st.rerun()

        generate = st.button(
            "Processing..." if st.session_state.report_processing else "Run analysis",
            disabled=bool(st.session_state.report_processing or st.session_state.evidence_processing),
        )

        if generate:
            if not uploaded:
                st.warning("Please upload a raw genotype .txt file first.")
            else:
                st.session_state.report_processing = True
                skeleton_slot = st.empty()
                skeleton_slot.markdown(
                    "<div class='skeleton'>Preparing report components... parsing upload, matching traits, and checking evidence.</div>",
                    unsafe_allow_html=True,
                )
                file_bytes = uploaded.getvalue()
                ghash = genotype_hash_from_bytes(file_bytes)
                cache_key = (
                    ghash,
                    tuple(categories_selected),
                    bool(include_optional_liver),
                    str(explain_mode),
                    ENGINE_VERSION,
                )

                try:
                    with st.status("Running analysis", expanded=True) as status:
                        status.write("1/5 Parsing genotype")
                        persist_uploaded_file(file_bytes, ghash)
                        parsed_variants = cached_parse_genotype(ghash)

                        status.write("2/5 Computing PRS")
                        cached_report = st.session_state.get("analysis_cache", {})
                        if isinstance(cached_report, dict) and cached_report.get("key") == cache_key:
                            report = normalize_report(cached_report.get("report", {}))
                        else:
                            report = cached_build_base_report(ghash, tuple(categories_selected), include_optional_liver)
                            report = normalize_report(report)
                            st.session_state["analysis_cache"] = {"key": cache_key, "report": report}

                        status.write("3/5 Retrieving evidence")
                        status.write("4/5 Generating explanations")
                        status.write("5/5 Rendering report")
                        status.update(label="Analysis complete", state="complete")

                    report = normalize_report(report)
                    if not isinstance(report, dict):
                        raise ValueError("Report generation failed: expected a dict report object.")
                    report = attach_rag_evidence_to_report(report)

                    # Save for chatbot
                    st.session_state.last_report = report

                    if not report.get("traits"):
                        st.warning("No traits were matched. Check that the file uses the expected rsIDs and genotypes.")
                    else:
                        selected_traits = sort_traits_for_display(report.get("traits", []))
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
                        evidence_found = sum(1 for t in enriched_traits if str(t.get("evidence_status", "")).strip().lower() == "found")
                        evidence_missing = max(0, len(enriched_traits) - evidence_found)
                        st.markdown("### Your VivaGene Trait Summary")
                        st.caption(
                            f"Categories: {', '.join(cats) if cats else '—'} · Traits found: {n_traits} · Coverage: {cov_txt}"
                        )
                        st.caption(
                            f"Matched traits: {len(enriched_traits)} · Evidence found: {evidence_found} · Evidence missing: {evidence_missing} · Coverage: {cov_txt}"
                        )

                        st.markdown("### Evidence-based overview")
                        st.caption("This overview is educational. It does not diagnose disease or replace medical care.")

                        grouped = {"Neurobehavior": [], "Nutrition": [], "Fitness": [], "Liver": []}
                        mode_now = st.session_state.get("explain_mode", "Patient (simple)")
                        mode_key = "doctor" if "Doctor" in mode_now else "patient"
                        fallback_count = 0
                        if not enriched_traits:
                            st.error("No traits were loaded from analysis output. Please verify file format and rerun.")
                        final_cache = st.session_state.get("analysis_final_cache", {})
                        if isinstance(final_cache, dict) and final_cache.get("key") == cache_key:
                            grouped = final_cache.get("grouped", grouped)
                            enriched_traits = final_cache.get("enriched_traits", enriched_traits)
                            fallback_count = int(final_cache.get("fallback_count", 0) or 0)
                        else:
                            enrich_status = st.status("Preparing trait explanations", expanded=False)
                            work_traits = []
                            for trait in enriched_traits:
                                if not isinstance(trait, dict):
                                    continue
                                raw_cat = " ".join(
                                    [
                                        str(trait.get("category", "")),
                                        str(trait.get("track", "")),
                                        str(trait.get("subcategory", "")),
                                        str(trait.get("trait_name", "")),
                                    ]
                                )
                                norm_cat = normalize_category(raw_cat) or "Neurobehavior"
                                coverage = float(trait.get("coverage", 0.0) or 0.0) if isinstance(trait.get("coverage", 0.0), (int, float)) else 0.0
                                has_any_variant = coverage > 0.0
                                trait["has_any_variant"] = has_any_variant
                                trait.setdefault("present_variants", [])
                                trait.setdefault("missing_variants", [])
                                trait["_normalized_category"] = norm_cat
                                work_traits.append(trait)

                            enrich_status.write("a) Parsing genotype")
                            enrich_status.write("b) Computing PRS")
                            enrich_status.write(f"c) Retrieving evidence ({len(work_traits)} traits)")

                            evid_by_id = {}
                            if work_traits:
                                with ThreadPoolExecutor(max_workers=6) as pool:
                                    futures = {
                                        pool.submit(
                                            cached_retrieve_trait_evidence,
                                            {
                                                "trait_id": trait.get("trait_id", ""),
                                                "trait_name": trait.get("trait_name", ""),
                                                "category": trait.get("_normalized_category", ""),
                                                "gene": trait.get("gene", ""),
                                                "rsid": trait.get("rsid", ""),
                                            },
                                        ): str(trait.get("trait_id", ""))
                                        for trait in work_traits
                                    }
                                    for fut in as_completed(futures):
                                        tid = futures[fut]
                                        try:
                                            evid_by_id[tid] = fut.result()
                                        except Exception:
                                            evid_by_id[tid] = {"status": "missing", "queries": [], "snippets": [], "citations": []}

                            enrich_status.write(f"d) Generating explanations ({len(work_traits)} traits)")
                            expl_by_id = {}
                            if work_traits:
                                with ThreadPoolExecutor(max_workers=3) as pool:
                                    futures = {}
                                    for trait in work_traits:
                                        tid = str(trait.get("trait_id", ""))
                                        ev = evid_by_id.get(tid, {"status": "missing", "queries": [], "snippets": [], "citations": []})
                                        snippets_for_mode = ev.get("snippets", []) if isinstance(ev.get("snippets", []), list) else []
                                        futures[pool.submit(cached_generate_trait_explanation, trait, snippets_for_mode, mode_key)] = tid
                                    for fut in as_completed(futures):
                                        tid = futures[fut]
                                        try:
                                            expl_by_id[tid] = fut.result()
                                        except Exception:
                                            expl_by_id[tid] = {
                                                "explanation": "Evidence pending — no explanatory claims shown yet (RAG safety).",
                                                "sources": [],
                                                "status": "missing",
                                                "used_fallback": True,
                                            }

                            for trait in work_traits:
                                tid = str(trait.get("trait_id", ""))
                                evidence = evid_by_id.get(tid, {"status": "missing", "queries": [], "snippets": [], "citations": []})
                                exp = expl_by_id.get(
                                    tid,
                                    {
                                        "explanation": "Evidence pending — no explanatory claims shown yet (RAG safety).",
                                        "sources": [],
                                        "status": "missing",
                                        "used_fallback": True,
                                    },
                                )

                                summary_txt = two_sentence_text(exp.get("explanation", ""))
                                life_txt = ""
                                if not trait.get("has_any_variant", False):
                                    summary_txt = "No variants found in the uploaded file for this trait. General evidence is shown below, but no user-specific claim is made."
                                if exp.get("used_fallback", False):
                                    fallback_count += 1
                                if "Evidence pending" in summary_txt or "could not retrieve enough research" in summary_txt.lower():
                                    life_txt = ""
                                if "Doctor" in mode_now and summary_txt and "Evidence pending" not in summary_txt:
                                    if "[PMID:" not in summary_txt and "[PMCID:" not in summary_txt and "[DOI:" not in summary_txt:
                                        summary_txt = "Evidence pending — no explanatory claims shown yet (RAG safety)."
                                        life_txt = ""

                                trait["_final_summary"] = summary_txt
                                trait["_final_life_impact"] = life_txt
                                trait["_queries_used"] = evidence.get("queries", []) if isinstance(evidence.get("queries", []), list) else []
                                trait["_final_citations"] = exp.get("sources", []) if isinstance(exp.get("sources", []), list) else []
                                trait["citations"] = trait["_final_citations"]
                                trait["sources"] = trait["_final_citations"]
                                trait["_evidence_snippets"] = evidence.get("snippets", []) if isinstance(evidence.get("snippets", []), list) else []
                                trait["evidence_status"] = str(exp.get("status", evidence.get("status", "missing"))).strip()
                                trait["explanation_patient"] = summary_txt
                                trait["explanation_doctor"] = summary_txt
                                grouped.setdefault(trait.get("_normalized_category", "Neurobehavior"), []).append(trait)

                            enrich_status.write("e) Rendering report")
                            enrich_status.update(label="Trait explanations ready", state="complete")
                            st.session_state["analysis_final_cache"] = {
                                "key": cache_key,
                                "grouped": grouped,
                                "enriched_traits": [x for rows in grouped.values() for x in rows if isinstance(x, dict)],
                                "fallback_count": fallback_count,
                            }

                        if fallback_count > 0:
                            st.warning("Some explanations used safe fallback due to an API or evidence retrieval error.")

                        displayed_traits = [x for rows in grouped.values() for x in rows if isinstance(x, dict)]
                        filtered_report = build_filtered_report(report, displayed_traits)
                        st.session_state.last_report = filtered_report

                        with st.spinner("Writing lifestyle explanations with citations..."):
                            for cat_name, cat_traits in grouped.items():
                                if not isinstance(cat_traits, list) or not cat_traits:
                                    continue
                                chunks = [cat_traits[i:i + 20] for i in range(0, len(cat_traits), 20)]
                                merged = {}
                                for chunk in chunks:
                                    payload = []
                                    for t in chunk:
                                        if not isinstance(t, dict):
                                            continue
                                        payload.append(
                                            {
                                                "trait_id": str(t.get("trait_id", "")).strip(),
                                                "trait_name": str(t.get("trait_name", "")).strip(),
                                                "category": cat_name,
                                                "gene": str(t.get("gene", "")).strip(),
                                                "rsid": str(t.get("rsid", "")).strip(),
                                                "genotype": str(t.get("user_genotype", "")).strip(),
                                                "effect_label": str(t.get("effect_label", "")).strip(),
                                                "signal_bucket": str(t.get("effect_level", t.get("bucket", ""))).strip(),
                                                "evidence_strength": str(t.get("evidence_strength", "")).strip(),
                                                "coverage": float(t.get("coverage", 0.0) or 0.0) if isinstance(t.get("coverage", 0.0), (int, float)) else 0.0,
                                                "citations": _normalize_citations_from_trait(t),
                                            }
                                        )
                                    merged.update(cached_generate_category_trait_explainers(cat_name, mode_key, payload))
                                for t in cat_traits:
                                    if not isinstance(t, dict):
                                        continue
                                    tid = str(t.get("trait_id", "")).strip()
                                    row = merged.get(tid, _fallback_trait_lifestyle(t))
                                    t["_lifestyle_one_liner"] = str(row.get("one_liner", "")).strip()
                                    bul = row.get("bullets", [])
                                    if not isinstance(bul, list):
                                        bul = []
                                    t["_lifestyle_bullets"] = [str(x).strip() for x in bul if str(x).strip()][:3]
                                    cits = row.get("citations", [])
                                    if not isinstance(cits, list):
                                        cits = []
                                    t["_lifestyle_citations"] = cits[:3]

                        summary_cache_key = (
                            ghash,
                            mode_key,
                            tuple(categories_selected),
                            bool(include_optional_liver),
                            ENGINE_VERSION,
                        )
                        existing_summary_cache = st.session_state.get("category_summary_cache", {})
                        if isinstance(existing_summary_cache, dict) and existing_summary_cache.get("key") == summary_cache_key:
                            category_summaries = existing_summary_cache.get("summaries", {})
                        else:
                            category_summaries = {}
                            category_order = ["Neurobehavior", "Nutrition", "Fitness"]
                            if grouped.get("Liver"):
                                category_order.append("Liver")
                            for cat_name in category_order:
                                cat_traits = [t for t in grouped.get(cat_name, []) if isinstance(t, dict)]
                                category_summaries[cat_name] = generate_category_summary(cat_name, cat_traits, mode_key)
                            st.session_state["category_summary_cache"] = {
                                "key": summary_cache_key,
                                "summaries": category_summaries,
                            }
                        st.session_state["category_summaries"] = category_summaries

                        with st.container():
                            st.markdown("<div class='results-panel'>", unsafe_allow_html=True)
                            st.markdown("<div class='section-title'>Trait cards</div>", unsafe_allow_html=True)
                            tab_to_cat_order = ["Neurobehavior", "Nutrition", "Fitness"] + (["Liver"] if grouped.get("Liver") else [])
                            tabs = st.tabs(tab_to_cat_order)
                            tab_to_cat = list(zip(tab_to_cat_order, tabs))
                            for cat_name, tab in tab_to_cat:
                                with tab:
                                    st.markdown("<div class='results-scroll'>", unsafe_allow_html=True)
                                    cat_summary = st.session_state.get("category_summaries", {}).get(cat_name, "")
                                    if cat_summary:
                                        if isinstance(cat_summary, dict):
                                            bullets = cat_summary.get("bullets", [])
                                            if not isinstance(bullets, list):
                                                bullets = []
                                            bullets_html = "<ul>" + "".join(f"<li>{_html_escape(str(b))}</li>" for b in bullets[:6]) + "</ul>"
                                            cits = cat_summary.get("citations", [])
                                            cite_line = ""
                                            if isinstance(cits, list) and cits:
                                                parts = []
                                                for c in cits[:6]:
                                                    if not isinstance(c, dict):
                                                        continue
                                                    pmid = str(c.get("pmid", "")).strip()
                                                    title = str(c.get("title", "")).strip() or "Study"
                                                    parts.append(f"{_html_escape(title)}{f' [PMID:{_html_escape(pmid)}]' if pmid else ''}")
                                                if parts:
                                                    cite_line = "<div class='trait-muted'><b>Sources:</b> " + "; ".join(parts) + "</div>"
                                            summary_html = f"<div class='category-summary-body'>{bullets_html}{cite_line}</div>"
                                        else:
                                            summary_html = f"<div class='category-summary-body'>{_html_escape(str(cat_summary))}</div>"
                                        st.markdown(
                                            f"<div class='category-summary-box'><div class='category-summary-title'>Category summary</div>{summary_html}</div>",
                                            unsafe_allow_html=True,
                                        )
                                    cat_traits = grouped.get(cat_name, [])
                                    if not cat_traits:
                                        st.markdown(
                                            "<div class='trait-muted'>No matching variants from your file for this category yet.</div>",
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        for trait in cat_traits:
                                            render_trait_card(trait, "patient" if "Patient" in mode_now else "doctor")
                                    st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                        # Runtime integrity checks
                        rendered_count = sum(len(v) for v in grouped.values())
                        expected_count = len([t for t in enriched_traits if isinstance(t, dict)])
                        if rendered_count != expected_count:
                            st.warning(
                                f"Integrity check: rendered trait cards ({rendered_count}) do not match expected displayed traits ({expected_count})."
                            )
                        bad_doctor_traits = [
                            t.get("trait_id", t.get("trait_name", "Trait"))
                            for rows in grouped.values() for t in rows
                            if "Doctor" in mode_now
                            and "Evidence pending" not in str(t.get("_final_summary", ""))
                            and "[PMID:" not in str(t.get("_final_summary", ""))
                            and "[PMCID:" not in str(t.get("_final_summary", ""))
                            and "[DOI:" not in str(t.get("_final_summary", ""))
                        ]
                        if bad_doctor_traits:
                            st.warning("Citation check: some doctor-mode summaries lacked explicit citations and were replaced with evidence-pending text.")

                        st.markdown("### Detailed trait report")
                        html_report = generate_html_report(filtered_report, ai_summary=None)
                        if html_report is None:
                            html_report = ""
                        if not isinstance(html_report, str):
                            html_report = str(html_report)
                        st.session_state["last_html_report"] = html_report

                        pdf_bytes = generate_pdf_report_bytes(filtered_report, mode_used=mode_now)
                        st.session_state["last_pdf_bytes"] = pdf_bytes
                        st.session_state["last_report_timestamp"] = time.time()
                        st.download_button(
                            label="Download PDF report",
                            data=pdf_bytes,
                            file_name="VivaGene_Report.pdf",
                            mime="application/pdf",
                            key="download_pdf_report",
                        )
                    skeleton_slot.empty()
                    st.session_state.report_processing = False

                except Exception as e:
                    st.session_state.report_processing = False
                    st.error("Something went wrong while generating your report. Please check your file format and try again.")
                    st.caption("If this keeps happening, contact the VivaGene team or check the app logs.")

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

    st.markdown("#### Chat policy")
    st.markdown(
        """
        - Educational and informational support only  
        - No diagnosis, disease prediction, or treatment guidance  
        - If symptoms are discussed, it will recommend professional care  
        - Uses cautious, non-alarmist language  
        - Shows citations when available and refuses unsupported claims  
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
    mode_now = st.session_state.get("explain_mode", "Patient (simple)")
    mode_key = "doctor" if "Doctor" in str(mode_now) else "patient"

    if not ack:
        st.info("Please acknowledge to continue.")

    if report is None:
        st.info("Generate a report on the Upload page first so the chatbot has context about your traits.")
    else:
        # Optionally generate a structured lifestyle overview plan
        if generate_plan_clicked:
            try:
                lifestyle_plan = generate_lifestyle_plan(report, mode=mode_key)
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
                "You may discuss gentle lifestyle ideas related to sleep, focus, caffeine, training, and general wellness. "
                "You must avoid medical advice, diagnosis, or treatment recommendations and never prescribe medications or supplements. "
                "Use careful language like may, might, and could, and encourage the user to talk with a clinician "
                "or genetic counselor for any medical questions. "
                f"Response mode is {mode_key}. "
                "If doctor mode, include concise citation tags like [PMID:xxxx] only when available in context. "
                "If patient mode, avoid dense citation blocks."
            )

            cat_summaries = st.session_state.get("category_summaries", {})
            context_snippet = (
                f"Trait JSON:\n{report}\n\n"
                f"Summary:\n{ai_summary or ''}\n\n"
                f"Category summaries:\n{cat_summaries if isinstance(cat_summaries, dict) else {}}"
            )

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
                    if mode_key == "doctor":
                        src = _collect_report_sources(report, limit=4)
                        if src:
                            reply = reply.rstrip() + "\n\nSuggested reading:\n" + "\n".join(src)
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
        role = st.selectbox(
            "I am primarily a…",
            ["Student", "Educator", "Researcher or clinician", "Developer", "Other"],
        )
        message = st.text_area("Message", height=140)
        submitted = st.form_submit_button("Send message")

        if submitted:
            if not (name.strip() and message.strip()):
                st.warning("Please fill in name and a brief message.")
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

end_page_wrap()
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
