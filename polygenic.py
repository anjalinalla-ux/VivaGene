from typing import Any


def _allele_dosage(genotype: str, effect_allele: str) -> int:
    gt = str(genotype or "").strip().upper()
    ea = str(effect_allele or "").strip().upper()
    if len(gt) != 2 or not ea:
        return 0
    return sum(1 for a in gt if a == ea)


def compute_prs(trait_pack: dict, user_variants: list[dict]) -> dict[str, Any]:
    """Compute a simple additive PRS scaffold with coverage reporting."""
    variants = trait_pack.get("variants", []) if isinstance(trait_pack, dict) else []
    if not isinstance(variants, list):
        variants = []

    by_rsid = {}
    for row in user_variants or []:
        if not isinstance(row, dict):
            continue
        rsid = str(row.get("rsid", "")).strip()
        genotype = str(row.get("genotype", "")).strip().upper()
        if not rsid or not genotype or genotype == "--":
            continue
        by_rsid[rsid] = genotype

    total = len([v for v in variants if isinstance(v, dict) and str(v.get("rsid", "")).strip()])
    found = 0
    score = 0.0

    for v in variants:
        if not isinstance(v, dict):
            continue
        rsid = str(v.get("rsid", "")).strip()
        if not rsid:
            continue
        effect_allele = str(v.get("effect_allele", "")).strip().upper()
        weight = float(v.get("weight", 0.0) or 0.0)
        genotype = by_rsid.get(rsid)
        if not genotype:
            continue
        found += 1
        dosage = _allele_dosage(genotype, effect_allele)
        score += dosage * weight

    coverage = (found / total) if total > 0 else 0.0
    z_like = score if found > 0 else None
    confidence = "Low" if coverage < 0.2 else "Medium"
    warning = "Low variant coverage for PRS interpretation." if coverage < 0.2 else ""

    return {
        "prs_score": float(score),
        "z_like": z_like,
        "coverage": float(round(coverage, 4)),
        "found": int(found),
        "total": int(total),
        "confidence": confidence,
        "warning": warning,
    }
