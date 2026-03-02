from __future__ import annotations

import json
from pathlib import Path

from study_pack_loader import load_all_trait_packs


def _choose_example_variants(packs: list[dict], n_traits: int = 12) -> list[tuple[str, str]]:
    picked = []
    seen = set()
    for p in packs:
        if not isinstance(p, dict):
            continue
        for v in p.get("variants", []) if isinstance(p.get("variants", []), list) else []:
            if not isinstance(v, dict):
                continue
            rsid = str(v.get("rsid", "")).strip()
            if not rsid or rsid in seen or rsid.startswith("rsTBD"):
                continue
            ea = str(v.get("effect_allele", "A")).strip().upper() or "A"
            oa = str(v.get("other_allele", "G")).strip().upper() or "G"
            gt = "".join(sorted(ea + oa))
            picked.append((rsid, gt))
            seen.add(rsid)
            break
        if len(picked) >= n_traits:
            break
    return picked


def _write_user_file(path: Path, variant_pairs: list[tuple[str, str]]) -> None:
    lines = ["rsid\tchromosome\tposition\tgenotype"]
    for i, (rsid, gt) in enumerate(variant_pairs, start=1):
        lines.append(f"{rsid}\t1\t{100000 + i}\t{gt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_validation_users(base_dir: str = "validation/data") -> list[dict]:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    packs = load_all_trait_packs("trait_study_packs", categories=["Neurobehavior", "Nutrition", "Fitness"], include_optional=False)
    examples = _choose_example_variants(packs, n_traits=15)
    if len(examples) < 6:
        examples = examples + [(f"rsEX{i}", "AG") for i in range(1, 7 - len(examples))]

    user_a = examples[:10]
    user_b = examples[3:13]
    user_c = examples[::2][:8]

    users = [
        ("A", user_a, "contains neurobehavior + nutrition + fitness hits"),
        ("B", user_b, "shifted overlap profile"),
        ("C", user_c, "sparser profile for robustness testing"),
    ]

    out = []
    for uid, variants, notes in users:
        path = root / f"user_{uid}.txt"
        _write_user_file(path, variants)
        out.append({"user_id": uid, "path": str(path), "notes": notes})
    return out


def load_validation_users() -> list[dict]:
    return ensure_validation_users()


def load_trait_panel_subset() -> list[str]:
    packs = load_all_trait_packs("trait_study_packs")
    trait_ids = []
    for p in packs:
        if not isinstance(p, dict):
            continue
        tid = str(p.get("trait_id", "")).strip()
        variants = p.get("variants", [])
        if tid and isinstance(variants, list) and len(variants) >= 1:
            trait_ids.append(tid)
    return trait_ids
