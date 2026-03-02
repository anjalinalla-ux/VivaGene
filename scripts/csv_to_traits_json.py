#!/usr/bin/env python3
import csv
import json
from pathlib import Path


CATEGORY_TO_TRACK_SUBCATEGORY = {
    "Nutrition": ("Nutrition", "General"),
    "Fitness": ("Fitness", "General"),
    "Neurobehavior": ("Neurobehavior", "General"),
    "Sleep": ("Neurobehavior", "Sleep"),
    "Sensory": ("Neurobehavior", "Sensory"),
    "Appearance": ("Neurobehavior", "Appearance"),
}

LIVER_KEYWORDS = ("liver", "nafld", "fatty liver", "steatosis")


def normalize_evidence_strength(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"high", "strong"}:
        return "High"
    if v in {"moderate", "medium"}:
        return "Moderate"
    if v in {"low", "weak", "limited"}:
        return "Low"
    return "Moderate"


def map_track_subcategory(category: str, trait_name: str):
    name = (trait_name or "").strip().lower()
    if any(k in name for k in LIVER_KEYWORDS):
        return ("Liver", "Fatty liver")
    return CATEGORY_TO_TRACK_SUBCATEGORY.get((category or "").strip(), ("Neurobehavior", "General"))


def migrate(csv_path: Path, output_path: Path):
    # Group by logical trait and keep nested variants/genotypes.
    traits = {}
    for row in csv.DictReader(csv_path.open(newline="", encoding="utf-8")):
        trait_id = (row.get("trait_id") or "").strip()
        title = (row.get("trait_name") or "").strip()
        category = (row.get("category") or "").strip()
        rsid = (row.get("rsid") or "").strip()
        gene = (row.get("gene") or "").strip()
        genotype = (row.get("genotype") or "").strip().upper()
        if not trait_id or not title or not rsid or not genotype:
            continue

        track, subcategory = map_track_subcategory(category, title)
        trait_key = (trait_id, track, subcategory, title)
        trait_obj = traits.setdefault(
            trait_key,
            {
                "trait_id": trait_id,
                "track": track,
                "subcategory": subcategory,
                "title": title,
                "trait_name": title,
                "priority": "Major" if normalize_evidence_strength(row.get("evidence_strength", "")) in {"High", "Moderate"} else "Standard",
                "flag_level": "Medium" if normalize_evidence_strength(row.get("evidence_strength", "")) in {"High", "Moderate"} else "Low",
                "user_visibility_default": normalize_evidence_strength(row.get("evidence_strength", "")) in {"High", "Moderate"},
                "user_question": "",
                "what_it_means": "",
                "why_it_matters": "",
                "user_summary": "",
                "limitations": "",
                "tags": [],
                "variants": {},
                "research_query_hint": "",
                "citation_seeds": [],
                "mechanism_keywords": [],
            },
        )

        variant_key = (rsid, gene)
        if variant_key not in trait_obj["variants"]:
            trait_obj["variants"][variant_key] = {
                "rsid": rsid,
                "gene": gene,
                "genotypes": {},
                "evidence_strength": normalize_evidence_strength(row.get("evidence_strength", "")),
                "sources": [],
            }

        trait_obj["variants"][variant_key]["genotypes"][genotype] = {
            "effect_label": row.get("effect_label", ""),
            "effect_level": row.get("effect_level", ""),
            "explanation": row.get("explanation", ""),
        }

    out = []
    for trait in traits.values():
        variants = list(trait["variants"].values())
        trait["variants"] = variants
        out.append(trait)

    out.sort(key=lambda t: (t.get("track", ""), t.get("subcategory", ""), t.get("title", ""), t.get("trait_id", "")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} traits to {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    migrate(
        csv_path=project_root / "trait_database.csv",
        output_path=project_root / "data" / "traits.json",
    )
