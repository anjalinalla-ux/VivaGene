#!/usr/bin/env python3
import json
from pathlib import Path


TRACK_SPEC = {
    "Neurobehavior": {
        "prefix": "NB",
        "count": 35,
        "subcategories": [
            "Sleep & circadian",
            "Focus & attention",
            "Stress & reactivity",
            "Mood & affect",
            "Learning & memory",
            "Sensory sensitivity",
        ],
        "subcategory_counts": [6, 6, 6, 6, 6, 5],
        "codes": {
            "Sleep & circadian": "SLEEP",
            "Focus & attention": "FOCUS",
            "Stress & reactivity": "STRESS",
            "Mood & affect": "MOOD",
            "Learning & memory": "LEARN",
            "Sensory sensitivity": "SENS",
        },
    },
    "Nutrition": {
        "prefix": "NU",
        "count": 35,
        "subcategories": [
            "Caffeine & stimulants",
            "Carbs & glucose handling",
            "Lipids & cholesterol traits",
            "Appetite & satiety",
            "Micronutrients",
            "Digestion & food response",
        ],
        "subcategory_counts": [6, 6, 6, 6, 6, 5],
        "codes": {
            "Caffeine & stimulants": "CAFF",
            "Carbs & glucose handling": "CARB",
            "Lipids & cholesterol traits": "LIPID",
            "Appetite & satiety": "APP",
            "Micronutrients": "MICRO",
            "Digestion & food response": "DIGEST",
        },
    },
    "Fitness": {
        "prefix": "FIT",
        "count": 28,
        "subcategories": [
            "Training response & adaptation",
            "Recovery & soreness",
            "Injury & connective tissue",
            "Cardio & performance",
        ],
        "subcategory_counts": [7, 7, 7, 7],
        "codes": {
            "Training response & adaptation": "ADAPT",
            "Recovery & soreness": "RECOV",
            "Injury & connective tissue": "INJ",
            "Cardio & performance": "CARD",
        },
    },
    "Liver": {
        "prefix": "LIV",
        "count": 20,
        "subcategories": [
            "Liver fat & metabolism",
            "Inflammation & oxidative stress",
            "Bile & detox pathways",
        ],
        "subcategory_counts": [7, 7, 6],
        "codes": {
            "Liver fat & metabolism": "FAT",
            "Inflammation & oxidative stress": "INFL",
            "Bile & detox pathways": "DETOX",
        },
    },
}


def build_trait_name(track: str, subcategory: str, idx: int) -> str:
    return f"{track} placeholder trait {idx:03d} - {subcategory}"


def generate_templates():
    items = []
    for track, spec in TRACK_SPEC.items():
        prefix = spec["prefix"]
        subcats = spec["subcategories"]
        counts = spec["subcategory_counts"]
        codes = spec["codes"]

        global_idx = 1
        for subcat, n in zip(subcats, counts):
            code = codes[subcat]
            for local_idx in range(1, n + 1):
                trait_id = f"{prefix}_{code}_{local_idx:03d}"
                trait_name = build_trait_name(track, subcat, global_idx)
                is_major = local_idx <= 2
                items.append(
                    {
                        "trait_id": trait_id,
                        "track": track,
                        "subcategory": subcat,
                        "trait_name": trait_name,
                        "priority": "Major" if is_major else "Standard",
                        "flag_level": "High" if local_idx == 1 else ("Medium" if local_idx == 2 else "Low"),
                        "user_visibility_default": bool(is_major),
                        "user_question": "",
                        "what_it_means": "",
                        "why_it_matters": "",
                        "limitations": "",
                        "tags": [],
                        "variants": [],
                        "research_query_hint": "",
                        "citation_seeds": [],
                        "mechanism_keywords": [],
                    }
                )
                global_idx += 1

        if len([x for x in items if x["track"] == track]) != spec["count"]:
            raise ValueError(f"Track {track} count mismatch.")

    return items


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "traits.json"
    traits = generate_templates()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traits, indent=2), encoding="utf-8")
    print(f"Wrote {len(traits)} trait templates to {out_path}")
