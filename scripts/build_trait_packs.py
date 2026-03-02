#!/usr/bin/env python3
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))
from trait_completeness import compute_trait_completeness


ALLOWED_CATEGORIES = {"Neurobehavior", "Nutrition", "Fitness", "Liver"}
TARGET_COUNTS = {
    "Neurobehavior": 25,
    "Nutrition": 25,
    "Fitness": 20,
    "Liver": 20,
}

CATEGORY_DIR = {
    "Neurobehavior": "neurobehavior",
    "Nutrition": "nutrition",
    "Fitness": "fitness",
    "Liver": "liver_optional",
}

SUBCATEGORY_HINTS = {
    "sleep": "Sleep",
    "stress": "Stress",
    "focus": "Cognition",
    "memory": "Cognition",
    "caffeine": "Caffeine",
    "lactose": "Digestion",
    "fat": "Lipids",
    "endurance": "Performance",
    "recovery": "Recovery",
    "muscle": "Performance",
    "liver": "Liver fat & metabolism",
}

LIVER_PLACEHOLDER_NAMES = [
    "Liver fat accumulation tendency",
    "Hepatic lipid transport tendency",
    "Liver triglyceride handling tendency",
    "Hepatic insulin response tendency",
    "Liver oxidative stress tendency",
    "Inflammatory liver signaling tendency",
    "Bile acid transport tendency",
    "Hepatic detox capacity tendency",
    "Mitochondrial liver energy tendency",
    "Liver fibrosis marker tendency",
    "Hepatic inflammation sensitivity",
    "Liver antioxidant pathway tendency",
    "Hepatic fatty acid oxidation tendency",
    "Liver carbohydrate flux tendency",
    "Bile synthesis tendency",
    "Liver nutrient sensing tendency",
    "Hepatic stress response tendency",
    "Liver repair signaling tendency",
    "Hepatic cytokine balance tendency",
    "Liver metabolic flexibility tendency",
]


def slugify(text: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    return re.sub(r"_+", "_", t).strip("_")


def normalize_category(raw: str) -> str:
    x = (raw or "").strip()
    if x in ALLOWED_CATEGORIES:
        return x
    # Existing repo categories mapped into the fixed ontology.
    if x in {"Sleep", "Sensory", "Appearance"}:
        return "Neurobehavior"
    return "Neurobehavior"


def infer_subcategory(trait_name: str, category: str) -> str:
    name = (trait_name or "").lower()
    for k, v in SUBCATEGORY_HINTS.items():
        if k in name:
            return v
    if category == "Neurobehavior":
        return "Cognition"
    if category == "Nutrition":
        return "General nutrition"
    if category == "Fitness":
        return "Performance"
    return "Liver fat & metabolism"


def infer_alleles(genotypes: list[str]) -> tuple[str, str]:
    alleles = []
    for g in genotypes:
        gg = (g or "").strip().upper()
        if len(gg) != 2:
            continue
        for a in gg:
            if a in {"A", "C", "G", "T"} and a not in alleles:
                alleles.append(a)
    if not alleles:
        return "A", "G"
    if len(alleles) == 1:
        return alleles[0], alleles[0]
    return alleles[0], alleles[1]


def default_genotype_map(a1: str, a2: str) -> dict[str, dict[str, str]]:
    if a1 == a2:
        key = a1 + a1
        return {
            key: {
                "effect_level": "Typical",
                "effect_label": "Typical tendency",
                "notes": "This genotype is currently treated as a neutral baseline in the scaffold.",
            }
        }

    aa = a1 + a1
    ab = "".join(sorted(a1 + a2))
    bb = a2 + a2
    return {
        aa: {
            "effect_level": "Higher",
            "effect_label": "Slightly higher tendency",
            "notes": "This estimate may vary by population context and study design.",
        },
        ab: {
            "effect_level": "Typical",
            "effect_label": "Typical tendency",
            "notes": "This estimate may vary by population context and study design.",
        },
        bb: {
            "effect_level": "Lower",
            "effect_label": "Slightly lower tendency",
            "notes": "This estimate may vary by population context and study design.",
        },
    }


def make_variant(rows: list[dict[str, str]], trait_name: str) -> dict[str, Any]:
    rsid = (rows[0].get("rsid") or "").strip() or "rsTBD_1"
    gene = (rows[0].get("gene") or "").strip() or "GENE_TBD"
    strength_raw = (rows[0].get("evidence_strength") or "").strip().lower()
    if strength_raw in {"high", "strong"}:
        evidence_strength = "High"
    elif strength_raw in {"moderate", "medium"}:
        evidence_strength = "Moderate"
    else:
        evidence_strength = "Preliminary"

    genotypes = [str(r.get("genotype", "")).strip().upper() for r in rows]
    a1, a2 = infer_alleles(genotypes)

    genotype_map = {}
    for r in rows:
        gt = str(r.get("genotype", "")).strip().upper()
        if len(gt) != 2:
            continue
        effect_label = str(r.get("effect_label", "")).strip() or "Typical tendency"
        effect_level = str(r.get("effect_level", "")).strip().replace("_", " ").title() or "Typical"
        notes = str(r.get("explanation", "")).strip() or "This trait may vary across individuals and contexts."
        genotype_map[gt] = {
            "effect_level": effect_level,
            "effect_label": effect_label,
            "notes": notes,
        }

    if not genotype_map:
        genotype_map = default_genotype_map(a1, a2)

    citation_id = "PMID:TBD"
    return {
        "rsid": rsid,
        "gene": gene,
        "effect_allele": a1,
        "other_allele": a2,
        "weight": 0.01,
        "unit": "beta",
        "genotype_map": genotype_map,
        "evidence_strength": evidence_strength,
        "citations": [citation_id],
    }


def make_evidence(trait_name: str, category: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    explanation = str(rows[0].get("explanation", "")).strip() if rows else ""
    quote = explanation or (
        f"Current evidence placeholders suggest this {category.lower()} trait may show small tendency differences in the general population."
    )
    return [
        {
            "citation_id": "PMID:TBD",
            "citation_type": "PMID",
            "title": f"Placeholder evidence for {trait_name}",
            "year": 2020,
            "journal": "TBD Journal",
            "url": "https://example.org/tbd",
            "quote": quote[:260],
            "claim_tags": [slugify(trait_name), "general_population", "lifestyle_tendency"],
        }
    ]


def make_pack_from_group(trait_name: str, category: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    subcategory = infer_subcategory(trait_name, category)
    trait_id = (rows[0].get("trait_id") or "").strip() if rows else ""
    if not trait_id:
        trait_id = f"{category[:5].upper()}_{slugify(trait_name).upper()}_PRS_V1"

    by_variant = defaultdict(list)
    for r in rows:
        key = ((r.get("rsid") or "").strip(), (r.get("gene") or "").strip())
        by_variant[key].append(r)

    variants = []
    for _, grp in by_variant.items():
        variants.append(make_variant(grp, trait_name))

    pack = {
        "trait_id": trait_id,
        "category": category,
        "subcategory": subcategory,
        "trait_name": trait_name,
        "description_public": f"A polygenic estimate related to {trait_name.lower()} variation in the general population.",
        "is_optional": category == "Liver",
        "variants": variants,
        "evidence": make_evidence(trait_name, category, rows),
        "limitations": [
            "Not deterministic",
            "Population-specific effects possible",
        ],
        "safety": ["Educational only", "Not medical advice"],
        "metadata": {
            "curation_status": "Complete" if variants and not any(v["rsid"].startswith("rsTBD") for v in variants) else "Partial",
            "last_updated": str(date.today()),
            "source_notes": "Autogenerated from trait_database.csv scaffold.",
        },
    }
    pack["completeness"] = compute_trait_completeness(pack)
    return pack


def make_placeholder_pack(category: str, trait_name: str, index: int) -> dict[str, Any]:
    tid_prefix = {
        "Neurobehavior": "NEURO",
        "Nutrition": "NUTR",
        "Fitness": "FIT",
        "Liver": "LIVER",
    }[category]
    trait_id = f"{tid_prefix}_{slugify(trait_name).upper()}_{index:03d}_PRS_V1"
    pack = {
        "trait_id": trait_id,
        "category": category,
        "subcategory": infer_subcategory(trait_name, category),
        "trait_name": trait_name,
        "description_public": "This trait pack is scaffolded and will be expanded with curated variants and citations.",
        "is_optional": category == "Liver",
        "variants": [
            {
                "rsid": "rsTBD_1",
                "gene": "GENE_TBD",
                "effect_allele": "A",
                "other_allele": "G",
                "weight": 0.0,
                "unit": "beta",
                "genotype_map": {
                    "AA": {"effect_level": "Unknown", "effect_label": "Coming soon", "notes": "Variant-level interpretation is in progress."},
                    "AG": {"effect_level": "Unknown", "effect_label": "Coming soon", "notes": "Variant-level interpretation is in progress."},
                    "GG": {"effect_level": "Unknown", "effect_label": "Coming soon", "notes": "Variant-level interpretation is in progress."},
                },
                "evidence_strength": "Preliminary",
                "citations": ["PMID:TBD"],
            }
        ],
        "evidence": [
            {
                "citation_id": "PMID:TBD",
                "citation_type": "PMID",
                "title": f"Placeholder evidence for {trait_name}",
                "year": 2020,
                "journal": "TBD Journal",
                "url": "https://example.org/tbd",
                "quote": "Evidence retrieval pending for this scaffold trait.",
                "claim_tags": [slugify(trait_name), "placeholder", "general_population"],
            }
        ],
        "limitations": ["Not deterministic", "Population-specific effects possible"],
        "safety": ["Educational only", "Not medical advice"],
        "metadata": {
            "curation_status": "Placeholder",
            "last_updated": str(date.today()),
            "source_notes": "Placeholder created to complete ontology coverage.",
        },
    }
    pack["completeness"] = compute_trait_completeness(pack)
    return pack


def write_pack(out_root: Path, pack: dict[str, Any]) -> Path:
    category = pack["category"]
    subdir = CATEGORY_DIR[category]
    folder = out_root / subdir
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{slugify(pack['trait_name'])}_prs_v1.json"
    p = folder / filename
    p.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def build_packs(csv_path: Path, out_root: Path) -> dict[str, Any]:
    # Clear old json packs only.
    if out_root.exists():
        for f in out_root.rglob("*.json"):
            f.unlink()

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not isinstance(r, dict):
                continue
            rows.append(r)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        trait_name = (r.get("trait_name") or "").strip() or "Trait TBD"
        category = normalize_category(r.get("category") or "")
        grouped[(category, trait_name)].append(r)

    packs_by_category: dict[str, list[dict[str, Any]]] = {c: [] for c in ALLOWED_CATEGORIES}
    for (category, trait_name), grp in grouped.items():
        pack = make_pack_from_group(trait_name, category, grp)
        packs_by_category[category].append(pack)

    # Guarantee exact target sizes.
    for cat, target in TARGET_COUNTS.items():
        existing_names = {p["trait_name"] for p in packs_by_category[cat]}
        idx = 1
        if cat == "Liver":
            source_names = LIVER_PLACEHOLDER_NAMES
        else:
            source_names = [f"{cat} trait scaffold {i}" for i in range(1, target + 50)]

        while len(packs_by_category[cat]) < target:
            if idx - 1 < len(source_names):
                name = source_names[idx - 1]
            else:
                name = f"{cat} trait scaffold {idx}"
            if name in existing_names:
                idx += 1
                continue
            pack = make_placeholder_pack(cat, name, idx)
            packs_by_category[cat].append(pack)
            existing_names.add(name)
            idx += 1

        if len(packs_by_category[cat]) > target:
            # Keep curated-looking packs first.
            packs_by_category[cat].sort(
                key=lambda p: (p.get("metadata", {}).get("curation_status") == "Placeholder", p.get("trait_name", ""))
            )
            packs_by_category[cat] = packs_by_category[cat][:target]

    # Write files + summary
    summary = {}
    for cat in ["Neurobehavior", "Nutrition", "Fitness", "Liver"]:
        packs = sorted(packs_by_category[cat], key=lambda p: p.get("trait_name", ""))
        counts = {"Complete": 0, "Partial": 0, "ComingSoon": 0}
        for p in packs:
            comp = compute_trait_completeness(p)
            p["completeness"] = comp
            status = comp.get("status", "ComingSoon")
            counts[status] = counts.get(status, 0) + 1
            write_pack(out_root, p)

        summary[cat] = {
            "total": len(packs),
            "Complete": counts.get("Complete", 0),
            "Partial": counts.get("Partial", 0),
            "ComingSoon": counts.get("ComingSoon", 0),
        }

    return summary


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "trait_database.csv"
    out_root = project_root / "trait_study_packs"

    if not csv_path.exists():
        print(f"[ERROR] Missing CSV: {csv_path}")
        return 1

    summary = build_packs(csv_path, out_root)

    print("Trait pack build summary")
    for cat in ["Neurobehavior", "Nutrition", "Fitness", "Liver"]:
        s = summary.get(cat, {})
        print(
            f"- {cat}: total={s.get('total',0)} complete={s.get('Complete',0)} "
            f"partial={s.get('Partial',0)} coming_soon={s.get('ComingSoon',0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
