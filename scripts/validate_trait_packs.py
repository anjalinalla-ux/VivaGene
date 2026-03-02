#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))
from trait_completeness import compute_trait_completeness


REQUIRED_TOP = {
    "trait_id",
    "category",
    "subcategory",
    "trait_name",
    "description_public",
    "is_optional",
    "variants",
    "evidence",
    "limitations",
    "safety",
    "metadata",
}

ALLOWED_CATEGORIES = {"Neurobehavior", "Nutrition", "Fitness", "Liver"}


def validate_pack(pack: dict[str, Any], file_path: Path) -> list[str]:
    errs = []
    missing_top = REQUIRED_TOP - set(pack.keys())
    if missing_top:
        errs.append(f"{file_path}: missing keys {sorted(missing_top)}")

    cat = str(pack.get("category", "")).strip()
    if cat not in ALLOWED_CATEGORIES:
        errs.append(f"{file_path}: invalid category '{cat}'")

    variants = pack.get("variants", [])
    if not isinstance(variants, list) or len(variants) < 1:
        errs.append(f"{file_path}: variants must be a non-empty list")
    else:
        for i, v in enumerate(variants):
            if not isinstance(v, dict):
                errs.append(f"{file_path}: variant[{i}] must be object")
                continue
            for k in ["rsid", "gene", "effect_allele", "genotype_map", "evidence_strength"]:
                if k not in v:
                    errs.append(f"{file_path}: variant[{i}] missing '{k}'")
            gmap = v.get("genotype_map", {})
            if not isinstance(gmap, dict) or len(gmap) < 1:
                errs.append(f"{file_path}: variant[{i}] genotype_map empty")

    evidence = pack.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) < 1:
        errs.append(f"{file_path}: evidence must be a non-empty list")
    else:
        for j, ev in enumerate(evidence):
            if not isinstance(ev, dict):
                errs.append(f"{file_path}: evidence[{j}] must be object")
                continue
            for k in ["citation_id", "citation_type", "title", "year", "quote"]:
                if not str(ev.get(k, "")).strip():
                    errs.append(f"{file_path}: evidence[{j}] missing '{k}'")

    comp = compute_trait_completeness(pack)
    if not isinstance(comp, dict) or "score" not in comp:
        errs.append(f"{file_path}: completeness computation failed")

    return errs


def validate_dir(base: Path) -> tuple[int, list[str]]:
    errors = []
    files = list(base.rglob("*.json")) if base.exists() else []
    if not files:
        errors.append(f"No pack files found in {base}")
        return 1, errors

    for fp in sorted(files):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{fp}: invalid JSON ({e})")
            continue
        if not isinstance(data, dict):
            errors.append(f"{fp}: root must be object")
            continue
        errors.extend(validate_pack(data, fp))

    return (1 if errors else 0), errors


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    base = project_root / "trait_study_packs"
    code, errs = validate_dir(base)

    if errs:
        print("Validation warnings/errors:")
        for e in errs:
            print(f"- {e}")
    else:
        print("Validation OK: all trait study packs passed checks.")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
