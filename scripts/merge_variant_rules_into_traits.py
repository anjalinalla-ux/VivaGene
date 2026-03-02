#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ALLOWED_CATEGORIES = {"Neurobehavior", "Nutrition", "Fitness", "Liver"}
ALLOWED_GENOTYPES = {"AA", "AC", "AG", "AT", "CC", "CG", "CT", "GG", "GT", "TT", "--"}
CSV_HEADER = [
    "trait_id",
    "category",
    "rsid",
    "gene",
    "genotype",
    "effect_label",
    "effect_level",
    "evidence_strength",
    "priority",
    "flag_level",
    "explanation",
    "paper_seeds",
    "keywords",
]


def normalize_genotype(gt: str) -> str:
    x = (gt or "").strip().upper()
    if x == "--":
        return x
    if len(x) == 2 and x.isalpha():
        return "".join(sorted(x))
    return x


def parse_list_field(raw: str):
    text = (raw or "").strip()
    if not text:
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def title_from_trait_id(trait_id: str) -> str:
    return (trait_id or "").replace("_", " ").title()


def compute_trait_completeness(trait: dict):
    variants = trait.get("variants", [])
    if not isinstance(variants, list) or not variants:
        return {"score": 20, "label": "Placeholder", "missing": ["variants"]}

    variant_complete = 0
    has_rsid_gene = False
    has_2plus_genotypes = False
    has_explanation = False
    has_evidence = False
    missing = set()

    for v in variants:
        if not isinstance(v, dict):
            continue
        rsid = str(v.get("rsid", "")).strip()
        gene = str(v.get("gene", "")).strip()
        ge = v.get("genotype_effects", {})
        ev = str(v.get("evidence_strength", "")).strip()

        if rsid and gene:
            has_rsid_gene = True
        else:
            missing.add("rsid_or_gene")
        if ev:
            has_evidence = True
        else:
            missing.add("evidence_strength")

        if isinstance(ge, dict) and len(ge) >= 2:
            has_2plus_genotypes = True
        else:
            missing.add("genotype_effects")

        ge_explanations_ok = False
        if isinstance(ge, dict):
            for gx in ge.values():
                if isinstance(gx, dict) and str(gx.get("explanation", "")).strip():
                    ge_explanations_ok = True
                    has_explanation = True
                    break
        if not ge_explanations_ok:
            missing.add("explanation")

        if rsid and gene and ev and isinstance(ge, dict) and len(ge) >= 2 and ge_explanations_ok:
            variant_complete += 1

    if variant_complete == len([v for v in variants if isinstance(v, dict)]) and variant_complete > 0:
        score = 100
    else:
        score = 0
        if variants:
            score += 20
        if has_rsid_gene:
            score += 20
        if has_2plus_genotypes:
            score += 20
        if has_explanation:
            score += 20
        if has_evidence:
            score += 20

    if score >= 85:
        label = "Ready"
    elif score >= 50:
        label = "Draft"
    else:
        label = "Placeholder"

    return {"score": score, "label": label, "missing": sorted(missing)}


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "variant_rules.csv"
    json_path = project_root / "data" / "traits.json"

    warnings = []
    rows_skipped = 0
    traits_updated = 0
    variants_added = 0
    genotypes_added = 0

    if not csv_path.exists():
        print(f"[ERROR] Missing {csv_path}")
        return 1

    grouped = {}
    by_variant = defaultdict(set)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_HEADER:
            print("[ERROR] CSV header mismatch.")
            print("Expected:", ",".join(CSV_HEADER))
            print("Found   :", ",".join(reader.fieldnames or []))
            return 1

        for i, row in enumerate(reader, start=2):
            try:
                trait_id = str(row.get("trait_id", "")).strip()
                category = str(row.get("category", "")).strip()
                rsid = str(row.get("rsid", "")).strip()
                gene = str(row.get("gene", "")).strip()
                genotype = normalize_genotype(row.get("genotype", ""))
                effect_label = str(row.get("effect_label", "")).strip()
                effect_level = str(row.get("effect_level", "")).strip()
                evidence_strength = str(row.get("evidence_strength", "")).strip() or "Emerging"
                priority = str(row.get("priority", "")).strip() or "Standard"
                flag_level = str(row.get("flag_level", "")).strip() or "Low"
                explanation = str(row.get("explanation", "")).strip()
                paper_seeds = parse_list_field(row.get("paper_seeds", ""))
                keywords = parse_list_field(row.get("keywords", ""))

                if not trait_id:
                    warnings.append(f"Row {i}: missing trait_id")
                    rows_skipped += 1
                    continue
                if category not in ALLOWED_CATEGORIES:
                    warnings.append(f"Row {i}: invalid category '{category}' for {trait_id}")
                    rows_skipped += 1
                    continue
                if genotype not in ALLOWED_GENOTYPES:
                    warnings.append(f"Row {i}: invalid genotype '{genotype}' for {trait_id}")
                    rows_skipped += 1
                    continue
                if not rsid.startswith("rs"):
                    warnings.append(f"Row {i}: invalid rsid '{rsid}' for {trait_id}")

                trait = grouped.get(trait_id)
                if trait is None:
                    trait = {
                        "trait_id": trait_id,
                        "trait_name": title_from_trait_id(trait_id),
                        "category": category,
                        "priority": "Major" if priority == "Major" else "Standard",
                        "flag_level": flag_level if flag_level in {"High", "Medium", "Low"} else "Low",
                        "keywords": keywords,
                        "variants": [],
                    }
                    grouped[trait_id] = trait
                    traits_updated += 1
                else:
                    # Keep most visible config if inconsistent.
                    if priority == "Major":
                        trait["priority"] = "Major"
                    if trait.get("flag_level") != "High" and flag_level == "High":
                        trait["flag_level"] = "High"
                    if keywords:
                        merged_kw = list(dict.fromkeys((trait.get("keywords", []) or []) + keywords))
                        trait["keywords"] = merged_kw

                key = (trait_id, rsid, gene)
                variant = None
                for v in trait["variants"]:
                    if v.get("rsid") == rsid and v.get("gene") == gene:
                        variant = v
                        break
                if variant is None:
                    variant = {
                        "rsid": rsid,
                        "gene": gene,
                        "evidence_strength": evidence_strength,
                        "paper_seeds": paper_seeds,
                        "genotype_effects": {},
                    }
                    trait["variants"].append(variant)
                    variants_added += 1
                else:
                    if evidence_strength:
                        variant["evidence_strength"] = evidence_strength
                    if paper_seeds:
                        variant["paper_seeds"] = list(dict.fromkeys((variant.get("paper_seeds") or []) + paper_seeds))

                if genotype not in variant["genotype_effects"]:
                    genotypes_added += 1
                variant["genotype_effects"][genotype] = {
                    "effect_label": effect_label,
                    "effect_level": effect_level,
                    "explanation": explanation,
                }
                by_variant[key].add(genotype)
            except Exception as e:
                warnings.append(f"Row {i}: {e}")
                rows_skipped += 1

    traits = list(grouped.values())
    traits.sort(key=lambda t: (t.get("category", ""), t.get("trait_id", "")))

    # Defaults + quality metadata + visibility.
    for t in traits:
        if t.get("priority") == "Major":
            t["user_visibility_default"] = True
        else:
            t["user_visibility_default"] = False
        q = compute_trait_completeness(t)
        t["_quality"] = q

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(traits, f, ensure_ascii=False, indent=2)

    print("Merge complete.")
    print(f"traits_updated: {traits_updated}")
    print(f"variants_added: {variants_added}")
    print(f"genotypes_added: {genotypes_added}")
    print(f"rows_skipped: {rows_skipped}")
    print(f"warnings: {len(warnings)}")
    for w in warnings:
        print(f"WARN: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
