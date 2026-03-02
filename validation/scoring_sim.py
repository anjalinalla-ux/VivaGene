from __future__ import annotations

import random
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from genomics_interpreter import parse_genotype_file
from genomics_polygenic import compute_prs_for_trait, normalize_genotype
from study_pack_loader import load_all_trait_packs


def _variants_to_map(variants: list[dict]) -> dict[str, str]:
    out = {}
    for v in variants if isinstance(variants, list) else []:
        if not isinstance(v, dict):
            continue
        rsid = str(v.get("rsid", "")).strip()
        gt = normalize_genotype(v.get("genotype", ""))
        if rsid and gt:
            out[rsid] = gt
    return out


def simulate_missing_variants(genotype_variants: list[dict], missing_rate: float, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for row in genotype_variants if isinstance(genotype_variants, list) else []:
        if rng.random() < float(missing_rate):
            continue
        out.append(row)
    return out


def recompute_scores_and_drift(user_path: str, missing_rate: float, categories: list[str] | None = None) -> list[dict]:
    variants = parse_genotype_file(user_path)
    full_map = _variants_to_map(variants)
    missing = simulate_missing_variants(variants, missing_rate=missing_rate, seed=123)
    miss_map = _variants_to_map(missing)

    packs = load_all_trait_packs("trait_study_packs", categories=categories, include_optional=False)

    rows = []
    for p in packs:
        full = compute_prs_for_trait(p, full_map)
        miss = compute_prs_for_trait(p, miss_map)
        drift = abs(float(miss.get("prs_raw", 0.0) or 0.0) - float(full.get("prs_raw", 0.0) or 0.0))
        bucket_change = int(str(miss.get("bucket", "")) != str(full.get("bucket", "")))
        rows.append(
            {
                "trait_id": full.get("trait_id", ""),
                "category": full.get("category", ""),
                "missing_rate": float(missing_rate),
                "drift": float(drift),
                "bucket_change": bucket_change,
            }
        )
    return rows


def repeat_generation_stability(user_id: str, trait_outputs: dict[str, list[str]]) -> list[dict]:
    """trait_outputs maps trait_id -> list of repeated output strings."""
    rows = []
    for trait_id, runs in trait_outputs.items():
        runs = [str(r or "").strip() for r in runs if str(r or "").strip()]
        if len(runs) < 2:
            rows.append(
                {
                    "user_id": user_id,
                    "trait_id": trait_id,
                    "mean_similarity": 1.0,
                }
            )
            continue

        tfidf = TfidfVectorizer(stop_words="english")
        mat = tfidf.fit_transform(runs)
        sims = []
        for i, j in combinations(range(len(runs)), 2):
            sim = cosine_similarity(mat[i], mat[j])[0][0]
            sims.append(float(sim))
        mean_sim = sum(sims) / len(sims) if sims else 1.0
        rows.append({"user_id": user_id, "trait_id": trait_id, "mean_similarity": mean_sim})
    return rows
