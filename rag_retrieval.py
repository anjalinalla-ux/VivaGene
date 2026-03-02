import json
from pathlib import Path
from typing import Any

from trait_completeness import compute_trait_completeness


def load_study_packs(dir_path: str) -> list[dict]:
    """Load local study packs recursively from a directory. Never performs network calls."""
    base = Path(dir_path)
    if not base.exists() or not base.is_dir():
        return []

    packs: list[dict] = []
    for file_path in sorted(base.rglob("*.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data.setdefault("_pack_path", str(file_path))
            data["_completeness"] = compute_trait_completeness(data)
            packs.append(data)
    return packs


def trait_has_variants(trait_pack: dict) -> bool:
    variants = trait_pack.get("variants", []) if isinstance(trait_pack, dict) else []
    return isinstance(variants, list) and len(variants) >= 1


def trait_has_evidence(trait_pack: dict) -> bool:
    evidence = trait_pack.get("evidence", []) if isinstance(trait_pack, dict) else []
    if not isinstance(evidence, list) or len(evidence) == 0:
        return False
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("citation_id", "")).strip() and str(ev.get("quote", "")).strip():
            return True
    return False


def coming_soon_message(trait_pack: dict) -> str:
    comp = trait_pack.get("_completeness") if isinstance(trait_pack, dict) else None
    if not isinstance(comp, dict):
        comp = compute_trait_completeness(trait_pack if isinstance(trait_pack, dict) else {})
    if comp.get("status") != "Complete":
        return "This trait is in progress. The structure is ready, but variant/evidence curation is not complete yet."
    return ""


def retrieve_evidence(trait_id: str, study_packs: list[dict], k: int = 3) -> dict[str, Any]:
    """Return local evidence passages and citation fields for a trait id."""
    safe_k = max(1, int(k or 3))
    target = str(trait_id or "").strip()

    pack = None
    for candidate in study_packs or []:
        if isinstance(candidate, dict) and str(candidate.get("trait_id", "")).strip() == target:
            pack = candidate
            break

    if not isinstance(pack, dict):
        return {
            "trait_id": target,
            "passages": [],
            "coverage_notes": "No local study pack found for this trait.",
            "warnings": ["pack_missing"],
        }

    passages = []
    for ev in pack.get("evidence", []) or []:
        if not isinstance(ev, dict):
            continue
        passages.append(
            {
                "quote": str(ev.get("quote", "")).strip(),
                "citation_id": str(ev.get("citation_id", "")).strip(),
                "url": str(ev.get("url", "")).strip(),
                "title": str(ev.get("title", "")).strip(),
                "year": ev.get("year", ""),
            }
        )
        if len(passages) >= safe_k:
            break

    warnings = []
    if not trait_has_variants(pack):
        warnings.append("no_variants")
    if not trait_has_evidence(pack):
        warnings.append("no_evidence")

    if passages:
        coverage_notes = "Local study pack evidence retrieved."
    else:
        coverage_notes = "Study pack found but no evidence passages were available."

    msg = coming_soon_message(pack)
    if msg:
        warnings.append("coming_soon")

    return {
        "trait_id": target,
        "passages": passages,
        "coverage_notes": coverage_notes,
        "warnings": warnings,
        "coming_soon_message": msg,
    }
