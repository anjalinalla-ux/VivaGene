import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI
from utils.trait_quality import compute_trait_completeness, trait_has_variants
from study_pack_loader import load_all_trait_packs, index_packs_by_id
from genomics_polygenic import normalize_genotype as poly_normalize_genotype, compute_prs_for_trait
from rag_retriever import evidence_quality
from rag_generator import generate_trait_explanation_rag

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
client = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    if OPENROUTER_API_KEY
    else None
)

def normalize_report(report_obj):
    """Normalize report input so downstream code can safely use dict .get().

    Accepts either a full report dict or a list of trait dicts.
    Returns a dict with keys: summary, traits.
    """
    # If a list of trait dicts is passed, wrap it
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

    # If a dict is passed, ensure expected keys and types
    if isinstance(report_obj, dict):
        report_obj.setdefault("summary", {})
        raw_traits = report_obj.get("traits", [])
        traits = [t for t in raw_traits if isinstance(t, dict)] if isinstance(raw_traits, list) else []
        report_obj["traits"] = traits

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

def generate_ai_summary(report):
    """
    Use the OpenAI API to generate a friendly, non-medical summary
    of the person's genetic trait results.
    """

    report_json = json.dumps(report, indent=2)

    system_message = (
        "You are a friendly, supportive genetics educator writing for a teenager or adult "
        "with no formal genetics background. You are given structured genetic trait data in JSON.\n\n"
        "Your job:\n"
        "1. Start with a short 'Big Picture' overview (1–2 short paragraphs) summarizing overall themes.\n"
        "2. Then write a 'Highlights by Category' section. For any categories that appear in the JSON "
        "(e.g., Nutrition, Fitness, Sleep, Neurobehavior, Sensory, Appearance), briefly describe 1–3 key points "
        "in simple language. This should still be in paragraph form, not bullet points.\n"
        "3. End with a 'Remember' section emphasizing that genetics is only one factor and that environment, "
        "lifestyle, mental health, and medical care matter a lot.\n\n"
        "Important rules:\n"
        "- Do NOT give medical advice.\n"
        "- Do NOT diagnose or suggest treatments.\n"
        "- Do NOT mention specific SNP IDs or genotypes; focus on the meaning.\n"
        "- Keep the tone warm, encouraging, and non-alarming.\n"
        "- Write in clear paragraphs, no markdown symbols like ** or bullet points.\n"
    )

    user_message = (
        "Here is the JSON report describing this person's interpreted genetic traits:\n\n"
        f"{report_json}\n\n"
        "Please follow the instructions in the system message and write the summary accordingly."
    )

    if client is None:
        return ""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

TRAIT_DB_PATH = "trait_database.csv"
TRAITS_JSON_PATH = "data/traits.json"
TRAIT_DB_JSON_PATH = "trait_database_model.json"
GENOTYPE_FILE_PATH = "John_doe_genotype.txt"
LAST_MATCH_WARNINGS = []
LAST_DB_WARNINGS = []
EVIDENCE_DIR = Path("evidence_corpus")
EVIDENCE_INDEX_PATH = EVIDENCE_DIR / "index.jsonl"
EVIDENCE_SEARCH_LOG_PATH = EVIDENCE_DIR / "search_log.jsonl"
_EVIDENCE_CACHE = {"sig": "", "rows": []}

TRACK_SUBCATEGORY_MAP = {
    "Nutrition": ("Nutrition", "General"),
    "Fitness": ("Fitness", "General"),
    "Neurobehavior": ("Neurobehavior", "General"),
    "Sleep": ("Neurobehavior", "Sleep"),
    "Sensory": ("Neurobehavior", "Sensory"),
    "Appearance": ("Neurobehavior", "Appearance"),
}
ALLOWED_TRACKS = {"Neurobehavior", "Nutrition", "Fitness", "Liver"}

LIVER_KEYWORDS = ("liver", "nafld", "fatty liver", "steatosis")


def map_category_to_track_subcategory(category: str, trait_name: str = ""):
    cat = (category or "").strip()
    name = (trait_name or "").strip().lower()
    if any(k in name for k in LIVER_KEYWORDS):
        return ("Liver", "Fatty liver")
    track, subcategory = TRACK_SUBCATEGORY_MAP.get(cat, ("Neurobehavior", cat or "General"))
    if track not in ALLOWED_TRACKS:
        track = "Neurobehavior"
    return (track, subcategory)


def generate_text_report(report):
    report = normalize_report(report)
    lines = []
    lines.append("AI-READY GENETIC TRAIT SUMMARY")
    lines.append("=" * 40)
    lines.append(f"Number of traits interpreted: {report['summary']['num_traits_found']}")
    lines.append("Categories: " + ", ".join(report["summary"]["categories"]))
    lines.append("")

    # Group traits by category
    traits_by_cat = {}
    for t in report["traits"]:
        traits_by_cat.setdefault(t["category"], []).append(t)

    for category, traits in traits_by_cat.items():
        lines.append(f"\n## {category}")
        lines.append("-" * (4 + len(category)))

        for t in traits:
            lines.append(f"\nTrait: {t['trait_name']}")
            lines.append(f"Gene: {t['gene']} ({t['rsid']}) — Genotype: {t['user_genotype']}")
            lines.append(f"Effect: {t['effect_label']}  [{t['effect_level']}]")
            lines.append(f"Explanation: {t['explanation']}")
            lines.append(f"Evidence: {t['evidence_strength']}")
            lines.append("")

    return "\n".join(lines)

def load_trait_database(path=TRAIT_DB_PATH, traits_json_path=TRAITS_JSON_PATH):
    """Load trait definitions from traits.json keyed by (rsid, genotype)."""
    lookup = {}
    LAST_DB_WARNINGS.clear()

    try:
        for t in load_traits_catalog(traits_json_path):
            if not isinstance(t, dict):
                continue
            trait_id = (t.get("trait_id") or "").strip()
            title = (t.get("trait_name") or t.get("title") or title_from_trait_id(trait_id)).strip()
            track = (t.get("category") or t.get("track") or "").strip() or "Neurobehavior"
            if track not in ALLOWED_TRACKS:
                track = "Neurobehavior"
            subcategory = (t.get("subcategory") or "").strip() or "General"
            keywords = t.get("keywords", []) or t.get("mechanism_keywords", []) or []
            if not isinstance(keywords, list):
                keywords = []
            variants = t.get("variants", []) or []
            quality = t.get("_quality", compute_db_trait_completeness(t))
            quality_label = quality.get("label") if isinstance(quality, dict) else None

            if not trait_has_variants(t):
                continue
            if quality_label != "Ready":
                continue

            for var in variants:
                if not isinstance(var, dict):
                    continue
                rsid = (var.get("rsid") or "").strip()
                gene = (var.get("gene") or "").strip()
                evidence_strength = var.get("evidence_strength", "Moderate")
                genotypes = var.get("genotype_effects", {}) or var.get("genotypes", {}) or {}
                paper_seeds = var.get("paper_seeds", []) or t.get("paper_seeds", []) or t.get("citation_seeds", []) or []
                if not isinstance(paper_seeds, list):
                    paper_seeds = []

                if not rsid or not isinstance(genotypes, dict):
                    continue

                for gt, gx in genotypes.items():
                    if not isinstance(gx, dict):
                        continue
                    genotype = normalize_genotype(gt)
                    if not genotype or genotype == "--":
                        continue
                    if len(genotype) != 2 or not genotype.isalpha():
                        LAST_DB_WARNINGS.append(
                            f"Skipped unsupported database genotype format '{gt}' for {rsid}."
                        )
                        continue
                    row = {
                        "trait_id": trait_id or f"{track}_{subcategory}_{title}_{rsid}_{genotype}",
                        "trait_name": title,
                        "track": track,
                        "subcategory": subcategory,
                        "category": track,  # compatibility for existing renderers
                        "rsid": rsid,
                        "gene": gene,
                        "genotype": genotype,
                        "effect_label": gx.get("effect_label", ""),
                        "effect_level": gx.get("effect_level", ""),
                        "explanation": gx.get("explanation", ""),
                        "keywords": keywords,
                        "paper_seeds": paper_seeds,
                        "priority": t.get("priority", "Standard"),
                        "flag_level": t.get("flag_level", "Low"),
                        "user_visibility_default": bool(t.get("user_visibility_default", False)),
                        "evidence_strength": evidence_strength,
                        "_quality": quality,
                        "ai_life_impact": "",
                        "ai_red_flag": "",
                        "citations": [],
                    }
                    lookup[(rsid, genotype)] = row
    except FileNotFoundError:
        return {}
    except Exception as e:
        print("Failed to load data/traits.json:", e)

    return lookup


def title_from_trait_id(trait_id: str) -> str:
    return (trait_id or "").replace("_", " ").title()


def normalize_genotype(genotype: str) -> str:
    gt = str(genotype or "").strip().upper()
    if gt == "--":
        return gt
    if len(gt) == 2 and gt.isalpha():
        return "".join(sorted(gt))
    return gt


def reverse_genotype(genotype: str) -> str:
    gt = str(genotype or "").strip().upper()
    if len(gt) == 2:
        return gt[::-1]
    return gt


def compute_db_trait_completeness(trait: dict):
    variants = trait.get("variants", []) if isinstance(trait, dict) else []
    variants = [v for v in variants if isinstance(v, dict)]
    if not variants:
        return {"score": 20, "label": "Placeholder", "missing": ["variants"]}

    complete_count = 0
    has_rsid_gene = False
    has_two_gt = False
    has_exp = False
    has_ev = False

    for v in variants:
        rsid = str(v.get("rsid", "")).strip()
        gene = str(v.get("gene", "")).strip()
        ge = v.get("genotype_effects", {}) or v.get("genotypes", {}) or {}
        ev = str(v.get("evidence_strength", "")).strip()
        this_two = isinstance(ge, dict) and len(ge) >= 2
        this_exp = False
        if isinstance(ge, dict):
            for gx in ge.values():
                if isinstance(gx, dict) and str(gx.get("explanation", "")).strip():
                    this_exp = True
                    break
        if rsid and gene:
            has_rsid_gene = True
        if this_two:
            has_two_gt = True
        if this_exp:
            has_exp = True
        if ev:
            has_ev = True
        if rsid and gene and this_two and this_exp and ev:
            complete_count += 1

    if complete_count == len(variants):
        score = 100
    else:
        score = 20
        score += 20 if has_rsid_gene else 0
        score += 20 if has_two_gt else 0
        score += 20 if has_exp else 0
        score += 20 if has_ev else 0

    if score >= 85:
        label = "Ready"
    elif score >= 50:
        label = "Draft"
    else:
        label = "Placeholder"

    missing = []
    if not has_rsid_gene:
        missing.append("rsid_or_gene")
    if not has_two_gt:
        missing.append("genotype_effects>=2")
    if not has_exp:
        missing.append("explanation")
    if not has_ev:
        missing.append("evidence_strength")
    return {"score": score, "label": label, "missing": missing}


def load_traits_catalog(traits_json_path=TRAITS_JSON_PATH):
    """Load full trait catalog and precompute quality metadata."""
    with open(traits_json_path, encoding="utf-8") as f:
        traits = json.load(f)
    if not isinstance(traits, list):
        return []
    out = []
    for t in traits:
        if not isinstance(t, dict):
            continue
        t2 = dict(t)
        t2["_quality"] = compute_db_trait_completeness(t2)
        out.append(t2)
    return out


def parse_genotype_file(path):
    """
    Parse a 23andMe-style file.
    Returns list of dicts: {rsid, genotype, chromosome, position}
    """
    variants = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expect: rsid  chromosome  position  genotype
            if parts[0].lower() == "rsid":
                # header line
                continue
            if len(parts) < 4:
                continue
            rsid, chrom, pos, genotype = parts[0], parts[1], parts[2], parts[3]
            variants.append(
                {
                    "rsid": rsid,
                    "chromosome": chrom,
                    "position": pos,
                    "genotype": normalize_genotype(genotype),
                }
            )
    return variants


def _build_user_variants_map(variants):
    if not isinstance(variants, list):
        raise ValueError("parse_genotype_file output must be a list of dicts")
    out = {}
    for v in variants:
        if not isinstance(v, dict):
            continue
        rsid = str(v.get("rsid", "")).strip()
        gt = poly_normalize_genotype(v.get("genotype", ""))
        if rsid and gt:
            out[rsid] = gt
    return out


def _summarize_prs_trait(prs_obj):
    if not isinstance(prs_obj, dict):
        raise ValueError("PRS object must be a dict")
    hits = prs_obj.get("variant_hits", [])
    all_rsids = prs_obj.get("all_variant_rsids", []) if isinstance(prs_obj.get("all_variant_rsids", []), list) else []
    present_rsids = []
    for h in hits if isinstance(hits, list) else []:
        if not isinstance(h, dict):
            continue
        r = str(h.get("rsid", "")).strip()
        if r and r not in present_rsids:
            present_rsids.append(r)
    missing_rsids = [r for r in all_rsids if r not in set(present_rsids)]
    dominant_gene = "MULTI"
    dominant_rsid = "MULTI"
    dominant_genotype = ""
    if isinstance(hits, list) and hits:
        try:
            top_hit = max(hits, key=lambda h: abs(float(h.get("weight", 0.0) or 0.0)))
            dominant_gene = str(top_hit.get("gene", "")).strip() or "MULTI"
            dominant_rsid = str(top_hit.get("rsid", "")).strip() or "MULTI"
            dominant_genotype = str(top_hit.get("genotype", "")).strip()
        except Exception:
            dominant_gene = "MULTI"
            dominant_rsid = "MULTI"
            dominant_genotype = ""

    bucket = str(prs_obj.get("bucket", "Insufficient data")).strip()
    confidence = str(prs_obj.get("confidence", "Low")).strip()
    coverage = float(prs_obj.get("coverage", 0.0) or 0.0)
    effect_level = bucket if bucket in {"Low", "Typical", "High"} else "Insufficient data"
    explanation = str(prs_obj.get("explanation", "")).strip()

    priority = "Major" if (bucket == "High" and confidence != "Low" and coverage >= 0.50) else "Standard"
    flag_level = "High" if priority == "Major" else ("Medium" if bucket == "High" else "Low")

    return {
        "trait_id": prs_obj.get("trait_id", ""),
        "trait_name": prs_obj.get("trait_name", ""),
        "track": prs_obj.get("category", ""),
        "subcategory": prs_obj.get("subcategory", "General"),
        "category": prs_obj.get("category", ""),
        "gene": dominant_gene,
        "rsid": dominant_rsid,
        "user_genotype": dominant_genotype or f"{int(prs_obj.get('num_variants_found', 0))}/{int(prs_obj.get('num_variants_total', 0))} loci",
        "effect_label": f"{bucket} polygenic tendency",
        "effect_level": effect_level,
        "explanation": explanation,
        "evidence_strength": prs_obj.get("evidence_top", "Preliminary"),
        "coverage": coverage,
        "present_variants": present_rsids,
        "missing_variants": missing_rsids,
        "has_any_variant": coverage > 0.0,
        "confidence": confidence,
        "evidence_status": prs_obj.get("evidence_status", "missing"),
        "evidence_snippets": prs_obj.get("evidence_snippets", []),
        "search_provenance": prs_obj.get("search_provenance", {}),
        "citations": prs_obj.get("citations", []),
        "warnings": prs_obj.get("warnings", []),
        "trust": prs_obj.get("trust", {}),
        "priority": priority,
        "flag_level": flag_level,
    }


def _safe_jsonl_read(path: Path):
    if not path.exists():
        return []
    rows = []
    try:
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
    except Exception:
        return []
    return rows


def load_local_corpus(index_path: Path = EVIDENCE_INDEX_PATH):
    path = Path(index_path)
    sig = ""
    try:
        if path.exists():
            st = path.stat()
            sig = f"{path.resolve()}:{st.st_mtime_ns}:{st.st_size}"
        else:
            sig = f"{path.resolve()}:missing"
    except Exception:
        sig = str(path)
    if _EVIDENCE_CACHE.get("sig") == sig:
        return _EVIDENCE_CACHE.get("rows", [])
    rows = _safe_jsonl_read(path)
    _EVIDENCE_CACHE["sig"] = sig
    _EVIDENCE_CACHE["rows"] = rows
    return rows


def _latest_search_provenance(trait_id: str, gene: str, rsid: str, log_path: Path = EVIDENCE_SEARCH_LOG_PATH):
    rows = _safe_jsonl_read(log_path)
    tid = str(trait_id or "").strip()
    g = str(gene or "").strip().upper()
    r = str(rsid or "").strip().lower()
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        if tid and str(row.get("trait_id", "")).strip() == tid:
            return row
        if g and str(row.get("gene", "")).strip().upper() == g:
            return row
        if r and str(row.get("rsid", "")).strip().lower() == r:
            return row
    return {}


def retrieve_evidence(trait: dict, k: int = 3):
    rows = load_local_corpus(EVIDENCE_INDEX_PATH)
    if not isinstance(rows, list) or not rows:
        return []

    t = trait if isinstance(trait, dict) else {}
    trait_id = str(t.get("trait_id", "")).strip()
    gene = str(t.get("gene", "")).strip().upper()
    rsid = str(t.get("rsid", "")).strip().lower()
    trait_name = str(t.get("trait_name", "")).strip().lower()
    category = str(t.get("category", "")).strip().lower()
    query_terms = set(
        x
        for x in re.split(r"[^a-z0-9]+", f"{trait_name} {category}")
        if len(x) >= 3
    )

    primary = []
    fallback = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("trait_id", "")).strip()
        rg = str(row.get("gene", "")).strip().upper()
        rr = str(row.get("rsid", "")).strip().lower()
        if trait_id and rid == trait_id:
            primary.append(row)
            continue
        if (gene and rg == gene) or (rsid and rr == rsid):
            fallback.append(row)

    candidates = primary if primary else fallback if fallback else rows

    ranked = []
    for row in candidates:
        score = 0.0
        if trait_id and str(row.get("trait_id", "")).strip() == trait_id:
            score += 4.0
        if gene and str(row.get("gene", "")).strip().upper() == gene:
            score += 2.0
        if rsid and str(row.get("rsid", "")).strip().lower() == rsid:
            score += 2.0
        blob = (
            f"{str(row.get('title', '')).lower()} "
            f"{str(row.get('snippet', row.get('snippet_text', ''))).lower()}"
        )
        overlap = sum(1 for q in query_terms if q and q in blob)
        score += min(1.5, overlap * 0.2)
        ranked.append((score, row))

    ranked.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for _, row in ranked:
        ident = (
            str(row.get("pmid", "")).strip()
            or str(row.get("pmcid", "")).strip()
            or str(row.get("doi", "")).strip()
            or str(row.get("title", "")).strip()
        )
        key = f"{trait_id}|{ident}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= max(1, int(k or 3)):
            break
    return out


def _build_retrieval_query(prs_obj: dict, trait_pack: dict) -> str:
    trait_name = str(prs_obj.get("trait_name", "")).strip()
    category = str(prs_obj.get("category", "")).strip()
    subcategory = str(prs_obj.get("subcategory", "")).strip()
    genes = []
    rsids = []
    for v in trait_pack.get("variants", []) if isinstance(trait_pack.get("variants", []), list) else []:
        if isinstance(v, dict):
            g = str(v.get("gene", "")).strip()
            if g and g not in genes:
                genes.append(g)
            r = str(v.get("rsid", "")).strip()
            if r and r not in rsids:
                rsids.append(r)
    labels = []
    for hit in prs_obj.get("variant_hits", []) if isinstance(prs_obj.get("variant_hits", []), list) else []:
        if isinstance(hit, dict):
            lab = str(hit.get("effect_label", "")).strip()
            if lab:
                labels.append(lab)
    keywords = trait_pack.get("keywords", []) if isinstance(trait_pack.get("keywords", []), list) else []
    terms = " ".join(labels[:5] + [str(k).strip() for k in keywords[:8] if str(k).strip()])
    return f"{' '.join(genes)} {' '.join(rsids)} {trait_name} {category} {subcategory} SNP genotype effect association {terms}".strip()


def _build_citation_objects(snippets, max_items: int = 3):
    out = []
    seen = set()
    for sn in snippets if isinstance(snippets, list) else []:
        if not isinstance(sn, dict):
            continue
        pmid = str(sn.get("pmid", "")).strip()
        pmcid = str(sn.get("pmcid", "")).strip()
        doi = str(sn.get("doi", "")).strip()
        cid = pmid or pmcid or doi or str(sn.get("citation_id", "")).strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(
            {
                "label": len(out) + 1,
                "title": str(sn.get("title", "")).strip() or "Untitled",
                "year": str(sn.get("year", "")).strip(),
                "identifier": f"PMID:{pmid}" if pmid else (f"PMCID:{pmcid}" if pmcid else (f"DOI:{doi}" if doi else cid)),
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "source_url": str(sn.get("url", "")).strip(),
            }
        )
        if len(out) >= max(1, int(max_items or 3)):
            break
    return out


def filter_prs_results_for_display(prs_results, red_flag_only=True):
    if not isinstance(prs_results, list):
        raise ValueError("prs_results must be a list")
    if not red_flag_only:
        return prs_results, ""

    filtered = []
    for r in prs_results:
        if not isinstance(r, dict):
            continue
        bucket = str(r.get("bucket", "")).strip()
        confidence = str(r.get("confidence", "")).strip()
        coverage = float(r.get("coverage", 0.0) or 0.0)
        completeness_status = str((r.get("completeness") or {}).get("status", "")).strip()
        warnings = r.get("warnings", [])
        warnings = warnings if isinstance(warnings, list) else []

        has_insufficient_warning = any("Insufficient data" in str(w) for w in warnings)
        if has_insufficient_warning:
            continue

        strong_high = bucket == "High" and confidence != "Low" and coverage >= 0.50
        complete_high = completeness_status == "Complete" and bucket == "High"
        if strong_high or complete_high:
            filtered.append(r)

    if filtered:
        return filtered, ""

    fallback = sorted(
        [r for r in prs_results if isinstance(r, dict)],
        key=lambda x: float(x.get("coverage", 0.0) or 0.0),
        reverse=True,
    )[:5]
    return fallback, "No high flags detected; showing best-covered traits"


def build_prs_report_from_upload(genotype_path: str, categories_selected: list[str], include_optional_liver: bool, red_flag_only: bool) -> dict:
    variants = parse_genotype_file(genotype_path)
    user_map = _build_user_variants_map(variants)

    categories = [c for c in (categories_selected or []) if isinstance(c, str) and c.strip()]
    if include_optional_liver and "Liver" not in categories:
        categories = categories + ["Liver"]

    packs = load_all_trait_packs(
        base_dir="trait_study_packs",
        categories=categories if categories else None,
        include_optional=include_optional_liver,
    )

    pack_index = index_packs_by_id(packs)
    prs_results = []
    for pack in packs:
        if not isinstance(pack, dict):
            continue
        prs_obj = compute_prs_for_trait(pack, user_map)
        all_rsids = []
        for v in pack.get("variants", []) if isinstance(pack.get("variants", []), list) else []:
            if not isinstance(v, dict):
                continue
            r = str(v.get("rsid", "")).strip()
            if r:
                all_rsids.append(r)
        prs_obj["all_variant_rsids"] = sorted(set(all_rsids))
        prs_results.append(prs_obj)

    display_results = [r for r in prs_results if isinstance(r, dict)]
    fallback_note = ""
    corpus = load_local_corpus(EVIDENCE_INDEX_PATH)
    traits_refused = 0
    traits_with_evidence = 0
    traits_missing_evidence = 0

    for r in display_results:
        if not isinstance(r, dict):
            continue
        trait_id = str(r.get("trait_id", "")).strip()
        trait_pack = pack_index.get(trait_id, {})
        completeness_status = str((r.get("completeness") or {}).get("status", "")).strip()
        r["evidence_status"] = "missing"
        r["evidence_snippets"] = []
        r["citations"] = []
        r["explanation"] = ""
        r["search_provenance"] = {}

        if not trait_pack:
            r.setdefault("warnings", []).append("Insufficient evidence in local corpus")
            r["trust"] = {
                "bucket": r.get("bucket", "Insufficient data"),
                "confidence": r.get("confidence", "Low"),
                "coverage": r.get("coverage", 0.0),
                "evidence_quality": "Low",
                "citations_count": 0,
                "bias_note": "Genetic associations are population-level and may not generalize equally across ancestry groups; this is educational only.",
            }
            traits_refused += 1
            traits_missing_evidence += 1
            r["search_provenance"] = _latest_search_provenance(
                trait_id=trait_id,
                gene=str(r.get("gene", "")).strip(),
                rsid=str(r.get("rsid", "")).strip(),
                log_path=EVIDENCE_SEARCH_LOG_PATH,
            )
            continue

        if (not trait_pack.get("variants")) or completeness_status == "ComingSoon":
            r.setdefault("warnings", []).append("Coming soon: variant mapping and evidence curation in progress.")
            r["trust"] = {
                "bucket": r.get("bucket", "Insufficient data"),
                "confidence": r.get("confidence", "Low"),
                "coverage": r.get("coverage", 0.0),
                "evidence_quality": "Low",
                "citations_count": 0,
                "bias_note": "Genetic associations are population-level and may not generalize equally across ancestry groups; this is educational only.",
            }
            traits_refused += 1
            traits_missing_evidence += 1
            r["search_provenance"] = _latest_search_provenance(
                trait_id=trait_id,
                gene=str(r.get("gene", "")).strip(),
                rsid=str(r.get("rsid", "")).strip(),
                log_path=EVIDENCE_SEARCH_LOG_PATH,
            )
            continue

        query = _build_retrieval_query(r, trait_pack)
        gene_for_query = ""
        rsid_for_query = ""
        for v in trait_pack.get("variants", []) if isinstance(trait_pack.get("variants", []), list) else []:
            if not isinstance(v, dict):
                continue
            if not gene_for_query:
                gene_for_query = str(v.get("gene", "")).strip()
            if not rsid_for_query:
                rsid_for_query = str(v.get("rsid", "")).strip()
        snippets = retrieve_evidence(
            {
                "trait_id": trait_id,
                "trait_name": r.get("trait_name", ""),
                "category": r.get("category", ""),
                "gene": gene_for_query or r.get("gene", ""),
                "rsid": rsid_for_query or r.get("rsid", ""),
                "query": query,
            },
            k=3,
        )
        quality = evidence_quality(snippets)
        r["search_provenance"] = _latest_search_provenance(
            trait_id=trait_id,
            gene=gene_for_query or str(r.get("gene", "")).strip(),
            rsid=rsid_for_query or str(r.get("rsid", "")).strip(),
            log_path=EVIDENCE_SEARCH_LOG_PATH,
        )
        if quality.get("quality") in {"High", "Medium"} and snippets:
            traits_with_evidence += 1
            r["evidence_status"] = "found"
            r["evidence_snippets"] = [s for s in snippets if isinstance(s, dict)][:3]
            citations = _build_citation_objects(snippets, max_items=3)
            gen = generate_trait_explanation_rag(r, snippets, quality)

            if gen.get("unsupported", False):
                r["explanation"] = ""
                r["citations"] = []
                r["evidence_status"] = "missing"
                r.setdefault("warnings", []).append(str(gen.get("refusal_reason", "Insufficient evidence in local corpus")))
                traits_refused += 1
                traits_missing_evidence += 1
            else:
                ex1 = str(gen.get("explanation_1", "")).strip()
                ex2 = str(gen.get("explanation_2", "")).strip()
                r["explanation"] = (ex1 + (" " + ex2 if ex2 else "")).strip()
                r["citations"] = citations
        else:
            r["evidence_status"] = "missing"
            r["evidence_snippets"] = []
            r["citations"] = []
            r["explanation"] = ""
            traits_missing_evidence += 1

        r["trust"] = {
            "bucket": r.get("bucket", "Insufficient data"),
            "confidence": r.get("confidence", "Low"),
            "coverage": r.get("coverage", 0.0),
            "evidence_quality": quality.get("quality", "Low"),
            "citations_count": len(r.get("citations", [])),
            "bias_note": "Genetic associations are population-level and may not generalize equally across ancestry groups; this is educational only.",
        }

    display_traits = [_summarize_prs_trait(r) for r in display_results]

    report = {
        "summary": {
            "num_traits_found": len(display_traits),
            "categories": sorted({t.get("category", "") for t in display_traits if t.get("category")}),
            "engine": "polygenic_v1",
            "red_flag_only": bool(red_flag_only),
            "evidence_found_count": traits_with_evidence,
            "evidence_missing_count": traits_missing_evidence,
        },
        "traits": display_traits,
        "traits_major": [t for t in display_traits if str(t.get("priority", "")).lower() == "major"],
        "traits_standard": [t for t in display_traits if str(t.get("priority", "")).lower() != "major"],
        "prs_debug": {str(r.get("trait_id", "")): r for r in prs_results if isinstance(r, dict) and r.get("trait_id")},
        "trust_summary": {
            "retrieval_corpus_size": len(corpus),
            "traits_with_evidence": traits_with_evidence,
            "traits_refused": traits_refused,
            "traits_missing_evidence": traits_missing_evidence,
        },
        "warnings": [fallback_note] if fallback_note else [],
    }

    avg_cov = 0.0
    if display_results:
        avg_cov = sum(float(r.get("coverage", 0.0) or 0.0) for r in display_results) / len(display_results)
    print(
        f"Engine: polygenic_v1 | Traits displayed: {len(display_traits)} | "
        f"Red-flag-only: {bool(red_flag_only)} | Avg coverage: {avg_cov:.2f}"
    )
    return normalize_report(report)


def match_traits(a, b):
    """
    Match uploaded variants to traits.

    Supports TWO calling styles:
      1) match_traits(trait_lookup_dict, variants_list)   # old CSV lookup style
      2) match_traits(variants_list, traits_list)         # new YAML traits list style

    Returns: matched_traits (list of dicts)
    """
    # --- Detect argument order ---
    variants = None
    trait_lookup = None
    traits_list = None

    # If first arg is a list, determine whether it is variants or trait defs.
    if isinstance(a, list):
        a0 = a[0] if a else {}
        b0 = b[0] if isinstance(b, list) and b else {}
        a_is_dict = isinstance(a0, dict)
        b_is_dict = isinstance(b0, dict)

        a_looks_like_variants = a_is_dict and ("rsid" in a0 and "genotype" in a0)
        a_looks_like_traits = a_is_dict and ("snps" in a0 or "trait_name" in a0 or "title" in a0 or "variants" in a0)
        b_looks_like_variants = b_is_dict and ("rsid" in b0 and "genotype" in b0)

        if a_looks_like_variants:
            variants = a
            if isinstance(b, dict):
                trait_lookup = b
            elif isinstance(b, list):
                traits_list = b
            else:
                raise TypeError("Second argument must be a dict (lookup) or list (traits).")
        elif a_looks_like_traits and isinstance(b, list) and b_looks_like_variants:
            # Support call style: match_traits(traits_list, variants)
            traits_list = a
            variants = b
        else:
            # Preserve old style fallback while avoiding .get() on a list later.
            trait_lookup = a if isinstance(a, dict) else None
            variants = b
    else:
        # old style: a is lookup dict, b is variants
        trait_lookup = a
        variants = b

    # --- Build rsid -> genotype map from uploaded variants ---
    LAST_MATCH_WARNINGS.clear()
    rs_to_gt = {}
    for row in variants if isinstance(variants, list) else []:
        if not isinstance(row, dict):
            continue
        rsid = str(row.get("rsid", "")).strip()
        raw_gt = str(row.get("genotype", "")).strip().upper()
        gt = normalize_genotype(raw_gt)
        if not rsid:
            continue
        if not raw_gt or raw_gt == "--" or gt == "--":
            LAST_MATCH_WARNINGS.append(f"Skipped {rsid} due to missing genotype.")
            continue
        if len(gt) != 2 or not gt.isalpha():
            LAST_MATCH_WARNINGS.append(
                f"Skipped {rsid} due to unsupported genotype format '{raw_gt}'."
            )
            continue
        rs_to_gt[rsid] = {"raw": raw_gt, "canonical": gt}

    uploaded_rsids = set(rs_to_gt.keys())

    matched_traits = []

    # =========================================================
    # NEW YAML STYLE: traits_list contains 'snps' list per trait
    # =========================================================
    if traits_list is not None:
        for t in traits_list:
            if not isinstance(t, dict):
                continue
            if not trait_has_variants(t):
                continue
            q = t.get("_quality", {})
            if not isinstance(q, dict) or q.get("label") != "Ready":
                continue
            snps = t.get("snps", []) or []
            # Only show traits that are present in the upload (your requirement)
            present = [rs for rs in snps if rs in uploaded_rsids]
            if not present:
                continue

            # Choose one representative SNP to display
            rsid = present[0]
            trait_obj = {
                "trait_id": t.get("trait_id", t.get("id", "")),
                "trait_name": t.get("trait_name", t.get("title", "")),
                "track": t.get("track", t.get("category", "")),
                "subcategory": t.get("subcategory", "General"),
                "category": t.get("category", ""),
                "rsid": rsid,
                "gene": "",  # not available in YAML yet
                "user_genotype": (rs_to_gt.get(rsid, {}) or {}).get("canonical", ""),
                "effect_label": "",  # not available in YAML yet
                "effect_level": "",  # not available in YAML yet
                "explanation": t.get("explanation", ""),
                "why_it_matters": t.get("why_it_matters", ""),
                "recommendations": t.get("recommendations", []),
                "what_it_does_not_mean": t.get("what_it_does_not_mean", ""),
                "evidence_strength": t.get("evidence_strength", "Limited"),
                "priority": t.get("priority", "Standard"),
                "flag_level": t.get("flag_level", "Low"),
                "user_visibility_default": bool(t.get("user_visibility_default", False)),
                "_quality": q,
                "ai_life_impact": "",
                "ai_red_flag": "",
                "citations": [],
            }
            matched_traits.append(trait_obj)

        return matched_traits

    # =========================================================
    # OLD CSV STYLE: trait_lookup keyed by (rsid, genotype)
    # =========================================================
    if trait_lookup is None:
        raise ValueError("No trait lookup provided.")

    for rsid, gt_info in rs_to_gt.items():
        raw_gt = (gt_info or {}).get("raw", "")
        canonical_gt = (gt_info or {}).get("canonical", "")
        candidates = []
        for cand in (raw_gt, reverse_genotype(raw_gt), canonical_gt):
            c = str(cand or "").strip().upper()
            if c and c not in candidates:
                candidates.append(c)

        row = None
        matched_gt = canonical_gt
        for cand in candidates:
            key = (rsid, normalize_genotype(cand))
            row = trait_lookup.get(key)
            if row:
                matched_gt = normalize_genotype(cand)
                break
        if not row:
            continue

        trait_obj = {
            "trait_id": row.get("trait_id", ""),
            "trait_name": row.get("trait_name", ""),
            "track": row.get("track", row.get("category", "")),
            "subcategory": row.get("subcategory", "General"),
            "category": row.get("track", row.get("category", "")),
            "rsid": row.get("rsid", rsid),
            "gene": row.get("gene", ""),
            "user_genotype": matched_gt,
            "effect_label": row.get("effect_label", ""),
            "effect_level": row.get("effect_level", ""),
            "explanation": row.get("explanation", ""),
            "evidence_strength": row.get("evidence_strength", "Limited"),
            "paper_seeds": row.get("paper_seeds", []),
            "keywords": row.get("keywords", []),
            "priority": row.get("priority", "Standard"),
            "flag_level": row.get("flag_level", "Low"),
            "user_visibility_default": bool(row.get("user_visibility_default", False)),
            "_quality": row.get("_quality", {}),
            "ai_life_impact": "",
            "ai_red_flag": "",
            "citations": [],
        }
        matched_traits.append(trait_obj)

    return matched_traits

def build_report_object(matched_traits):
    if not isinstance(matched_traits, list):
        matched_traits = []
    matched_traits = [t for t in matched_traits if isinstance(t, dict)]
    """
    Build a JSON-like object summarizing everything.
    This is what you'd send into an AI model prompt.
    """
    warnings = []
    if len(matched_traits) == 0:
        warnings.append("No traits matched your uploaded file.")
    if LAST_DB_WARNINGS:
        warnings.append("Some trait rules were skipped due to unsupported genotype encoding in the database.")
    if LAST_MATCH_WARNINGS:
        warnings.append("Some variants were skipped due to missing or unsupported genotype format.")
    if len(matched_traits) > 0 and all(int((t.get("_quality") or {}).get("score", 0)) < 100 for t in matched_traits):
        warnings.append("Only traits with incomplete data matched.")
    if len(matched_traits) > 0 and any(not str(t.get("explanation", "")).strip() for t in matched_traits):
        warnings.append("Coming soon: some matched traits do not yet have full explanations.")

    report = {
        "summary": {
            "num_traits_found": len(matched_traits),
            "categories": sorted({t.get("track", t.get("category", "")) for t in matched_traits if t.get("track", t.get("category", ""))}),
            "total_matched_ready": len(matched_traits),
            "major_matched_count": sum(1 for t in matched_traits if str(t.get("priority", "")).strip().lower() == "major"),
            "standard_matched_count": sum(1 for t in matched_traits if str(t.get("priority", "")).strip().lower() != "major"),
            "traits_ready_count": sum(1 for t in matched_traits if isinstance(t.get("_quality"), dict) and t.get("_quality", {}).get("label") == "Ready"),
            "traits_draft_count": sum(1 for t in matched_traits if isinstance(t.get("_quality"), dict) and t.get("_quality", {}).get("label") == "Draft"),
            "traits_placeholder_count": sum(1 for t in matched_traits if isinstance(t.get("_quality"), dict) and t.get("_quality", {}).get("label") == "Placeholder"),
        },
        "traits": matched_traits,
        "traits_major": [t for t in matched_traits if str(t.get("priority", "")).strip().lower() == "major"],
        "traits_standard": [t for t in matched_traits if str(t.get("priority", "")).strip().lower() != "major"],
        "polygenic": {},
        "evidence": {},
        "trust": {"coverage": 0.0, "confidence": "Low", "citations": []},
        "warnings": warnings,
    }
    return normalize_report(report)

def effect_level_to_percent(effect_level: str) -> int:
    """
    Map effect_level strings to a rough percentage for the visual bar.
    This is just for visualization, not a true quantitative score.
    """
    level = effect_level.upper()

    # Very rough mapping based on keywords
    if "VERY_HIGH" in level or "HIGH" in level and "LOW" not in level:
        return 85
    if "POWER" in level or "HIGH_RESPONSE" in level:
        return 80
    if "LOW" in level and "TOLERANCE" in level:
        return 25
    if "LOWER" in level or "REDUCED" in level:
        return 35
    if "INTERMEDIATE" in level or "MEDIUM" in level or "MIXED" in level:
        return 55
    if "TYPICAL" in level:
        return 50
    if "ENDURANCE" in level:
        return 60
    if "LIGHT" in level:
        return 65
    if "DARK" in level:
        return 45

    # Fallback
    return 50

def generate_html_report(report, ai_summary=None):
    report = normalize_report(report)
    # Simple icon per category
    category_icons = {
        "Nutrition": "🥦",
        "Fitness": "🏃‍♀️",
        "Sleep": "😴",
        "Neurobehavior": "🧠",
        "Sensory": "👁️",
        "Appearance": "🌈",
    }

    html_parts = []

    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Genetic Trait Report</title>
    <style>
        body {
            font-family: -apple-system, system-ui, -webkit-system-font, sans-serif;
            margin: 0;
            padding: 24px 32px 40px;
            background: #f5f7fb;
            color: #111827;
        }
        h1, h2, h3 {
            color: #111827;
            margin-top: 0;
        }
        h1 {
            font-size: 26px;
            margin-bottom: 4px;
        }
        h2 {
            font-size: 20px;
            margin-top: 22px;
        }
        p {
            margin: 4px 0 8px;
        }
        .header-sub {
            font-size: 0.9em;
            color: #4b5563;
        }
        .page {
            max-width: 900px;
            margin: 0 auto;
            padding: 22px 26px 30px;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 12px 25px rgba(15, 23, 42, 0.12);
        }
        .toc {
            margin: 16px 0 18px;
            padding: 10px 12px;
            background: #f3f4ff;
            border-radius: 12px;
            border: 1px solid #e5e7ff;
            font-size: 0.9em;
        }
        .toc-title {
            font-weight: 600;
            margin-bottom: 6px;
            color: #4338ca;
        }
        .toc-links a {
            margin-right: 10px;
            text-decoration: none;
            color: #4338ca;
            font-weight: 500;
            cursor: pointer;
        }
        .toc-links a:hover {
            text-decoration: underline;
        }
        .category {
            margin-top: 26px;
        }
        .category h2 {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .category-icon {
            font-size: 1.3em;
        }
        .trait-grid {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 6px;
        }
        .trait-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 10px 14px 12px;
            background: #f9fafb;
        }
        .trait-title {
            font-weight: 600;
            font-size: 1.0em;
            margin-bottom: 2px;
        }
        .meta {
            font-size: 0.82em;
            color: #6b7280;
        }
        .effect {
            margin-top: 6px;
            font-size: 0.9em;
        }
        .effect-label {
            font-weight: 500;
        }
        .effect-tag {
            display: inline-block;
            font-size: 0.78em;
            padding: 1px 8px;
            border-radius: 999px;
            background: #eef2ff;
            color: #4338ca;
            margin-left: 6px;
        }
        .bar-outer {
            width: 100%;
            background: #e5e7eb;
            border-radius: 999px;
            height: 8px;
            margin-top: 6px;
            overflow: hidden;
        }
        .bar-inner {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #6366f1, #22c55e);
        }
        .explanation {
            margin-top: 8px;
            font-size: 0.9em;
        }
        .evidence {
            font-size: 0.78em;
            color: #6b7280;
            margin-top: 4px;
        }
        .disclaimer {
            font-size: 0.8em;
            color: #6b7280;
            margin-top: 26px;
            border-top: 1px solid #e5e7eb;
            padding-top: 10px;
        }
        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76em;
            color: #6b7280;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .overview-box {
            margin-top: 10px;
            padding: 10px 12px;
            border-radius: 12px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            font-size: 0.95em;
        }
    </style>
</head>
<body>
<div class="page">
""")

    html_parts.append("<h1>Genetic Trait Summary</h1>")
    html_parts.append(
        f"<div class='header-sub'>Traits interpreted: "
        f"<strong>{report['summary']['num_traits_found']}</strong> "
        f"&nbsp;·&nbsp; Categories: {', '.join(report['summary']['categories'])}</div>"
    )

    categories = report["summary"]["categories"]
    if categories:
        html_parts.append("<div class='toc'>")
        html_parts.append("<div class='toc-title'>Jump to a section</div>")
        html_parts.append("<div class='toc-links'>")
        for cat in categories:
            icon = category_icons.get(cat, "🧬")
            safe_id = f"cat-{cat.replace(' ', '-')}"
            html_parts.append(
                f"<a href='#{safe_id}'>{icon} {cat}</a>"
            )
        html_parts.append("</div></div>")

    if ai_summary:
        ai_html = ai_summary.replace("\n", "<br>")
        html_parts.append("<div class='section-label'>Personalized overview</div>")
        html_parts.append("<div class='overview-box'>")
        html_parts.append(f"{ai_html}")
        html_parts.append("</div>")

    traits_by_cat = {}
    for t in report["traits"]:
        traits_by_cat.setdefault(t["category"], []).append(t)

    for category, traits in traits_by_cat.items():
        safe_id = f"cat-{category.replace(' ', '-')}"
        icon = category_icons.get(category, "🧬")

        html_parts.append(f'<div class="category" id="{safe_id}">')
        html_parts.append(
            f'<h2><span class="category-icon">{icon}</span>{category}</h2>'
        )
        html_parts.append('<div class="trait-grid">')

        for t in traits:
            from math import floor
            # crude mapping: use effect_level length as proxy if you have no numeric score
            percent = 50
            try:
                level = str(t["effect_level"]).upper()
                if "HIGH" in level and "LOW" not in level:
                    percent = 80
                elif "LOW" in level:
                    percent = 30
                elif "MEDIUM" in level or "INTERMEDIATE" in level or "TYPICAL" in level:
                    percent = 55
            except Exception:
                percent = 50

            html_parts.append('<div class="trait-card">')
            html_parts.append(f'<div class="trait-title">{t["trait_name"]}</div>')
            html_parts.append(
                f'<div class="meta">Gene: {t["gene"]} ({t["rsid"]}) · Genotype: {t["user_genotype"]}</div>'
            )
            if "coverage" in t or "confidence" in t:
                cov = t.get("coverage", None)
                cov_txt = f"{float(cov) * 100:.0f}%" if isinstance(cov, (int, float)) else "n/a"
                conf_txt = t.get("confidence", "n/a")
                cits = t.get("citations", []) if isinstance(t.get("citations", []), list) else []
                html_parts.append(
                    f'<div class="meta">Coverage: {cov_txt} · Confidence: {conf_txt} · Citations: {len(cits)}</div>'
                )
            trust = t.get("trust", {}) if isinstance(t.get("trust"), dict) else {}
            if trust:
                t_bucket = trust.get("bucket", t.get("effect_level", "n/a"))
                t_conf = trust.get("confidence", t.get("confidence", "n/a"))
                t_cov = trust.get("coverage", t.get("coverage", None))
                t_cov_txt = f"{float(t_cov) * 100:.0f}%" if isinstance(t_cov, (int, float)) else "n/a"
                t_eq = trust.get("evidence_quality", "Low")
                t_cits = t.get("citations", []) if isinstance(t.get("citations", []), list) else []
                t_bias = trust.get("bias_note", "")
                html_parts.append('<div class="meta"><strong>Trust Panel</strong></div>')
                html_parts.append(f'<div class="meta">Polygenic bucket: {t_bucket}</div>')
                html_parts.append(f'<div class="meta">Confidence: {t_conf}</div>')
                html_parts.append(f'<div class="meta">Coverage: {t_cov_txt}</div>')
                html_parts.append(f'<div class="meta">Evidence quality: {t_eq}</div>')
                html_parts.append(
                    f'<div class="meta">Citations: {", ".join(t_cits) if t_cits else "None yet"}</div>'
                )
                if t_bias:
                    html_parts.append(f'<div class="meta">Bias note: {t_bias}</div>')
            warns = t.get("warnings", []) if isinstance(t.get("warnings", []), list) else []
            if warns:
                html_parts.append(f'<div class="meta">Warnings: {"; ".join(str(w) for w in warns[:2])}</div>')
            html_parts.append(
                '<div class="effect">'
                f'<span class="effect-label">Effect:</span> {t["effect_label"]} '
                f'<span class="effect-tag">{t["effect_level"]}</span>'
                '</div>'
            )
            html_parts.append('<div class="bar-outer">')
            html_parts.append(f'<div class="bar-inner" style="width: {percent}%;"></div>')
            html_parts.append('</div>')
            html_parts.append(f'<div class="explanation">{t["explanation"]}</div>')
            html_parts.append(
                f'<div class="evidence"><strong>Evidence level:</strong> {t["evidence_strength"]}</div>'
            )
            html_parts.append("</div>")

        html_parts.append("</div>")
        html_parts.append("</div>")

    html_parts.append("""
<div class="disclaimer">
    <strong>Important:</strong> This report is for educational and informational purposes only.
    It does not provide medical advice, diagnosis, or treatment. Genetics is one factor among many including
    environment, sleep, stress, and medical history. For any health related questions, talk with a licensed
    healthcare provider or genetic counselor.
</div>
</div>
</body>
</html>
""")

    return "\n".join(html_parts)


def main():
    trait_lookup = load_trait_database(TRAIT_DB_PATH)
    variants = parse_genotype_file(GENOTYPE_FILE_PATH)
    matched_traits = match_traits(trait_lookup, variants)
    report = build_report_object(matched_traits)

    # JSON output
    print("Matched traits:")
    print(json.dumps(report, indent=2))

    # AI summary
    try:
        ai_summary = generate_ai_summary(report)
        print("\n\nAI Summary:\n")
        print(ai_summary)
    except Exception as e:
        print("AI generation failed:", e)
        ai_summary = None

    # Human-readable text report
    text_report = generate_text_report(report)
    print("\n\nHuman-readable report:\n")
    print(text_report)

    # Save text version
    with open("genetic_report.txt", "w", encoding="utf-8") as f:
        f.write(text_report)

    # Save HTML version (with AI overview if available)
    html_report = generate_html_report(report, ai_summary=ai_summary)
    with open("genetic_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)

    print("\nSaved genetic_report.txt and genetic_report.html")


if __name__ == "__main__":
    main()
