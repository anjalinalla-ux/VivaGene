from __future__ import annotations

from typing import Any

from trait_completeness import compute_trait_completeness
from rag_retrieval import trait_has_variants, trait_has_evidence


EVIDENCE_RANK = {"Preliminary": 0, "Moderate": 1, "High": 2}


def normalize_genotype(gt: str) -> str:
    g = str(gt or "").strip().upper()
    if not g or g in {"--", "00"}:
        return ""
    if len(g) == 1:
        return g
    if len(g) == 2:
        return "".join(sorted(g))
    return ""


def count_effect_alleles(genotype: str, effect_allele: str) -> int | None:
    g = normalize_genotype(genotype)
    if g == "":
        return None
    e = str(effect_allele or "").strip().upper()
    if not e:
        return None
    if len(g) == 1:
        return 1 if g == e else 0
    return sum(1 for ch in g if ch == e)


def compute_evidence_top(trait_pack: dict) -> str:
    strengths = []
    variants = trait_pack.get("variants", []) if isinstance(trait_pack, dict) else []
    for v in variants if isinstance(variants, list) else []:
        if not isinstance(v, dict):
            continue
        s = str(v.get("evidence_strength", "")).strip().title()
        if s not in EVIDENCE_RANK:
            s = "Preliminary"
        strengths.append(s)
    if not strengths:
        return "Preliminary"
    # Max-confidence strategy for prototype.
    return max(strengths, key=lambda x: EVIDENCE_RANK.get(x, 0))


def _map_confidence_from_coverage(coverage: float) -> str:
    if coverage >= 0.80:
        return "High"
    if 0.50 <= coverage < 0.80:
        return "Medium"
    return "Low"


def _downgrade_confidence(confidence: str, cap: str) -> str:
    order = ["Low", "Medium", "High"]
    return order[min(order.index(confidence), order.index(cap))]


def _bucket_from_prs(coverage: float, prs_norm: float | None) -> str:
    if coverage < 0.30:
        return "Insufficient data"
    if prs_norm is None:
        return "Insufficient data"
    if prs_norm <= 0.33:
        return "Low"
    if prs_norm < 0.66:
        return "Typical"
    return "High"


def compute_prs_for_trait(trait_pack: dict, user_variants_map: dict) -> dict[str, Any]:
    if not isinstance(trait_pack, dict):
        raise ValueError("trait_pack must be a dict")
    if not isinstance(user_variants_map, dict):
        raise ValueError("user_variants_map must be a dict")

    trait_id = str(trait_pack.get("trait_id", "")).strip()
    trait_name = str(trait_pack.get("trait_name", trait_id)).strip()
    category = str(trait_pack.get("category", "")).strip()
    subcategory = str(trait_pack.get("subcategory", "General")).strip()
    variants = trait_pack.get("variants", [])
    if not isinstance(variants, list):
        variants = []

    completeness = compute_trait_completeness(trait_pack)
    evidence_top = compute_evidence_top(trait_pack)

    hits = []
    citations = set()
    prs_raw = 0.0
    denom = 0.0

    num_total = 0
    num_found = 0

    for v in variants:
        if not isinstance(v, dict):
            continue
        rsid = str(v.get("rsid", "")).strip()
        gene = str(v.get("gene", "")).strip()
        if not rsid:
            continue
        num_total += 1

        gt = normalize_genotype(user_variants_map.get(rsid, ""))
        if not gt:
            continue

        effect_allele = str(v.get("effect_allele", "")).strip().upper()
        effect_count = count_effect_alleles(gt, effect_allele)
        if effect_count is None:
            continue

        num_found += 1
        weight = float(v.get("weight", 0.0) or 0.0)
        prs_raw += weight * effect_count
        denom += abs(weight)

        gmap = v.get("genotype_map", {}) if isinstance(v.get("genotype_map", {}), dict) else {}
        gx = gmap.get(gt, {}) if isinstance(gmap.get(gt, {}), dict) else {}

        hit_citations = [str(c).strip() for c in (v.get("citations", []) if isinstance(v.get("citations", []), list) else []) if str(c).strip()]
        for c in hit_citations:
            citations.add(c)

        hit = {
            "rsid": rsid,
            "gene": gene,
            "genotype": gt,
            "effect_allele": effect_allele,
            "effect_count": effect_count,
            "weight": weight,
            "evidence_strength": str(v.get("evidence_strength", "Preliminary")).title(),
            "citations": hit_citations,
            "effect_label": str(gx.get("effect_label", "Prototype polygenic component")).strip(),
            "effect_level": str(gx.get("effect_level", "Typical")).strip(),
        }
        hits.append(hit)

    coverage = (num_found / num_total) if num_total > 0 else 0.0
    prs_norm = (prs_raw / denom) if denom > 0 else None
    bucket = _bucket_from_prs(coverage, prs_norm)

    confidence = _map_confidence_from_coverage(coverage)
    if evidence_top == "Preliminary":
        confidence = _downgrade_confidence(confidence, "Low")
    if completeness.get("status") == "ComingSoon":
        confidence = _downgrade_confidence(confidence, "Low")

    warnings = []
    if not trait_has_variants(trait_pack):
        warnings.append("No variants curated yet")
    if not trait_has_evidence(trait_pack):
        warnings.append("Evidence not curated yet")
    if completeness.get("status") != "Complete":
        warnings.append("Trait pack is still being curated (coming soon/partial)")
    if bucket == "Insufficient data":
        warnings.append("Insufficient data")
    warnings.append("Prototype calibration: bucket thresholds are preliminary.")
    warnings.append("Educational only; genetics is one factor among many.")

    return {
        "trait_id": trait_id,
        "trait_name": trait_name,
        "category": category,
        "subcategory": subcategory,
        "num_variants_total": num_total,
        "num_variants_found": num_found,
        "coverage": float(coverage),
        "prs_raw": float(prs_raw),
        "prs_norm": float(prs_norm) if prs_norm is not None else None,
        "bucket": bucket,
        "confidence": confidence,
        "evidence_top": evidence_top,
        "completeness": completeness,
        "variant_hits": hits,
        "citations": sorted(citations),
        "warnings": warnings,
    }
