from typing import Any


def _expected_genotype_keys(effect_allele: str, other_allele: str) -> set[str]:
    ea = str(effect_allele or "").strip().upper()
    oa = str(other_allele or "").strip().upper()
    bases = {"A", "C", "G", "T"}
    if ea in bases and oa in bases and ea and oa:
        if ea == oa:
            return {ea + ea}
        a, b = sorted([ea, oa])
        return {a + a, a + b, b + b}
    # Graceful subset mode (e.g., non-standard/X/Y contexts)
    return set()


def compute_trait_completeness(trait_pack: dict[str, Any]) -> dict[str, Any]:
    score = 0
    missing: list[str] = []

    variants = trait_pack.get("variants", []) if isinstance(trait_pack, dict) else []
    if isinstance(variants, list) and len(variants) >= 1:
        score += 25
    else:
        missing.append("variants")
        variants = []

    variants_struct_ok = True
    if variants:
        for v in variants:
            if not isinstance(v, dict):
                variants_struct_ok = False
                break
            rsid = str(v.get("rsid", "")).strip()
            gene = str(v.get("gene", "")).strip()
            gmap = v.get("genotype_map", {})
            if not rsid or not gene or not isinstance(gmap, dict) or not gmap:
                variants_struct_ok = False
                break

            expected = _expected_genotype_keys(v.get("effect_allele", ""), v.get("other_allele", ""))
            keys = {str(k).strip().upper() for k in gmap.keys()}
            if expected:
                if not expected.issubset(keys):
                    variants_struct_ok = False
                    break
            else:
                # graceful subset allowance for non-standard allele contexts
                if len(keys) < 1:
                    variants_struct_ok = False
                    break

    if variants and variants_struct_ok:
        score += 25
    else:
        missing.append("variant_structure_or_genotype_map")

    evidence = trait_pack.get("evidence", []) if isinstance(trait_pack, dict) else []
    evidence_ok = False
    if isinstance(evidence, list) and len(evidence) >= 1:
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            citation_id = str(ev.get("citation_id", "")).strip()
            quote = str(ev.get("quote", "")).strip()
            title = str(ev.get("title", "")).strip()
            year = ev.get("year", "")
            if citation_id and quote and str(year).strip() and title:
                evidence_ok = True
                break

    if evidence_ok:
        score += 25
    else:
        missing.append("evidence")

    limitations = trait_pack.get("limitations", []) if isinstance(trait_pack, dict) else []
    if isinstance(limitations, list) and len(limitations) >= 1:
        score += 15
    else:
        missing.append("limitations")

    metadata = trait_pack.get("metadata", {}) if isinstance(trait_pack, dict) else {}
    curation_status = str(metadata.get("curation_status", "")).strip()
    last_updated = str(metadata.get("last_updated", "")).strip()
    if curation_status == "Complete" and last_updated:
        score += 10
    else:
        missing.append("metadata_complete")

    if score < 60:
        status = "ComingSoon"
    elif score < 85:
        status = "Partial"
    else:
        status = "Complete"

    return {"score": int(score), "missing": missing, "status": status}
