from __future__ import annotations

from pathlib import Path

from rag_retrieval import load_study_packs


def load_all_trait_packs(base_dir: str = "trait_study_packs", categories: list[str] | None = None, include_optional: bool = True) -> list[dict]:
    packs = load_study_packs(base_dir)
    if not isinstance(packs, list):
        return []

    allowed = set(categories or [])
    filtered = []
    for p in packs:
        if not isinstance(p, dict):
            continue
        cat = str(p.get("category", "")).strip()
        if allowed and cat not in allowed:
            continue
        if not include_optional and bool(p.get("is_optional", False)):
            continue
        filtered.append(p)
    return filtered


def index_packs_by_id(packs: list[dict]) -> dict[str, dict]:
    out = {}
    for p in packs if isinstance(packs, list) else []:
        if not isinstance(p, dict):
            continue
        tid = str(p.get("trait_id", "")).strip()
        if tid:
            out[tid] = p
    return out
