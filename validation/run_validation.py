from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

from validation.datasets import load_validation_users
from validation.prompts import unconstrained_prompt
from validation.metrics import hallucination_rate
from validation.scoring_sim import recompute_scores_and_drift, repeat_generation_stability
from validation.baselines import benchmark_agreement

from study_pack_loader import load_all_trait_packs, index_packs_by_id
from genomics_interpreter import parse_genotype_file
from genomics_polygenic import compute_prs_for_trait, normalize_genotype
from rag_retriever import load_corpus, retrieve_evidence, evidence_quality
from rag_generator import generate_trait_explanation_rag

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_CLIENT = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    if (OpenAI is not None and OPENROUTER_API_KEY)
    else None
)


def _call_unconstrained(trait_summary: dict) -> str:
    system, user = unconstrained_prompt(trait_summary)
    if OPENROUTER_CLIENT is None:
        return (
            f"This trait is associated with a {trait_summary.get('bucket','Typical')} tendency in this profile. "
            "The observed pattern may vary across people and contexts."
        )
    try:
        resp = OPENROUTER_CLIENT.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return (
            f"This trait is associated with a {trait_summary.get('bucket','Typical')} tendency in this profile. "
            "The observed pattern may vary across people and contexts."
        )


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


def _build_trait_results(user_path: str):
    variants = parse_genotype_file(user_path)
    user_map = _variants_to_map(variants)
    packs = load_all_trait_packs("trait_study_packs", categories=["Neurobehavior", "Nutrition", "Fitness"], include_optional=False)
    pack_index = index_packs_by_id(packs)
    prs = [compute_prs_for_trait(p, user_map) for p in packs]
    return prs, pack_index


def run() -> None:
    out_dir = Path("validation_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    users = load_validation_users()
    corpus = load_corpus("evidence_corpus")

    halluc_rows = []
    stability_rows = []
    robustness_rows = []
    benchmark_rows = []

    for user in users:
        user_id = user["user_id"]
        prs_results, pack_index = _build_trait_results(user["path"])

        unconstrained_runs = {}
        rag_runs = {}

        for prs_obj in prs_results:
            trait_id = prs_obj.get("trait_id", "")
            pack = pack_index.get(trait_id, {})
            query = f"{prs_obj.get('trait_name','')} {prs_obj.get('category','')}"
            snippets = retrieve_evidence(query, pack, k=6)
            quality = evidence_quality(snippets)

            # Unconstrained condition
            txt_u = _call_unconstrained(prs_obj)
            m_u = hallucination_rate(txt_u, snippets)
            halluc_rows.append(
                {
                    "user_id": user_id,
                    "trait_id": trait_id,
                    "condition": "unconstrained",
                    **m_u,
                    "refusal": 0,
                }
            )

            # RAG evidence-only condition
            rag_out = generate_trait_explanation_rag(prs_obj, snippets, quality)
            txt_r = (str(rag_out.get("explanation_1", "")) + " " + str(rag_out.get("explanation_2", ""))).strip()
            m_r = hallucination_rate(txt_r, snippets)
            halluc_rows.append(
                {
                    "user_id": user_id,
                    "trait_id": trait_id,
                    "condition": "rag",
                    **m_r,
                    "refusal": int(bool(rag_out.get("unsupported", False))),
                }
            )

            unconstrained_runs[trait_id] = []
            for _ in range(5):
                unconstrained_runs[trait_id].append(_call_unconstrained(prs_obj))

            rag_runs[trait_id] = []
            for _ in range(3):
                r = generate_trait_explanation_rag(prs_obj, snippets, quality)
                rag_runs[trait_id].append((str(r.get("explanation_1", "")) + " " + str(r.get("explanation_2", ""))).strip())

            b = benchmark_agreement(prs_obj)
            benchmark_rows.append({"user_id": user_id, **b})

        for row in repeat_generation_stability(user_id, unconstrained_runs):
            stability_rows.append({**row, "condition": "unconstrained"})
        for row in repeat_generation_stability(user_id, rag_runs):
            stability_rows.append({**row, "condition": "rag"})

        for mr in (0.1, 0.3, 0.5):
            robustness_rows.extend(recompute_scores_and_drift(user["path"], missing_rate=mr, categories=["Neurobehavior", "Nutrition", "Fitness"]))

    # Write CSVs
    def write_csv(path: Path, rows: list[dict]):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        cols = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write_csv(out_dir / "hallucination.csv", halluc_rows)
    write_csv(out_dir / "stability.csv", stability_rows)
    write_csv(out_dir / "robustness.csv", robustness_rows)
    write_csv(out_dir / "benchmark.csv", benchmark_rows)

    # Aggregate metrics
    def avg(rows, cond, key):
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows if r.get("condition") == cond]
        return (sum(vals) / len(vals)) if vals else 0.0

    hall_u = avg(halluc_rows, "unconstrained", "unsupported_rate")
    hall_r = avg(halluc_rows, "rag", "unsupported_rate")
    stab_u = avg(stability_rows, "unconstrained", "mean_similarity")
    stab_r = avg(stability_rows, "rag", "mean_similarity")

    rb03 = [r for r in robustness_rows if abs(float(r.get("missing_rate", 0.0)) - 0.3) < 1e-9]
    rb03_change = (sum(int(r.get("bucket_change", 0)) for r in rb03) / len(rb03)) if rb03 else 0.0

    comparable = [r for r in benchmark_rows if int(r.get("comparable", 0)) == 1]
    agree = (sum(int(r.get("agrees", 0)) for r in comparable) / len(comparable)) if comparable else 0.0

    summary = {
        "hallucination_rate_unconstrained": hall_u,
        "hallucination_rate_rag": hall_r,
        "stability_unconstrained": stab_u,
        "stability_rag": stab_r,
        "robustness_bucket_change@0.3": rb03_change,
        "benchmark_agreement": agree,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    plt.figure(figsize=(6, 4))
    plt.bar(["Unconstrained", "RAG"], [hall_u, hall_r])
    plt.ylabel("Unsupported claim rate")
    plt.title("Hallucination reduction")
    plt.tight_layout()
    plt.savefig(out_dir / "hallucination_reduction.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Unconstrained", "RAG"], [stab_u, stab_r])
    plt.ylabel("Mean pairwise similarity")
    plt.title("Interpretive stability")
    plt.tight_layout()
    plt.savefig(out_dir / "stability.png")
    plt.close()

    # robustness by category
    by_cat = {}
    for r in robustness_rows:
        cat = str(r.get("category", "Unknown"))
        mr = float(r.get("missing_rate", 0.0) or 0.0)
        by_cat.setdefault(cat, {}).setdefault(mr, []).append(int(r.get("bucket_change", 0)))

    plt.figure(figsize=(7, 4))
    for cat, series in by_cat.items():
        xs = sorted(series.keys())
        ys = [sum(series[x]) / len(series[x]) if series[x] else 0.0 for x in xs]
        plt.plot(xs, ys, marker="o", label=cat)
    plt.xlabel("Missing rate")
    plt.ylabel("Bucket change %")
    plt.title("Robustness to missing variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "robustness.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Agreement"], [agree])
    plt.ylim(0, 1)
    plt.ylabel("Agreement rate")
    plt.title("Benchmark agreement")
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_agreement.png")
    plt.close()

    print("Validation run complete. Outputs saved to validation_outputs/")


if __name__ == "__main__":
    run()
