from __future__ import annotations


def _evidence_rank(value: str) -> int:
    v = str(value or "").strip().title()
    if v == "High":
        return 3
    if v == "Moderate":
        return 2
    return 1


def _label_direction(label: str) -> str:
    text = str(label or "").lower()
    if any(k in text for k in ["higher", "increase", "more", "elevated"]):
        return "high"
    if any(k in text for k in ["lower", "decrease", "less", "reduced"]):
        return "low"
    if any(k in text for k in ["typical", "average", "normal"]):
        return "typical"
    return "unknown"


def prs_bucket_direction(bucket: str) -> str:
    b = str(bucket or "").strip().lower()
    if b == "high":
        return "high"
    if b == "low":
        return "low"
    if b == "typical":
        return "typical"
    return "unknown"


def baseline_single_snp_label(prs_obj: dict) -> str:
    hits = prs_obj.get("variant_hits", []) if isinstance(prs_obj, dict) else []
    if not isinstance(hits, list) or not hits:
        return "unknown"

    best = None
    best_rank = -1
    best_weight = -1.0
    for h in hits:
        if not isinstance(h, dict):
            continue
        rank = _evidence_rank(h.get("evidence_strength", "Preliminary"))
        wt = abs(float(h.get("weight", 0.0) or 0.0))
        if rank > best_rank or (rank == best_rank and wt > best_weight):
            best_rank = rank
            best_weight = wt
            best = h

    if not isinstance(best, dict):
        return "unknown"
    return str(best.get("effect_label", "")).strip() or "unknown"


def benchmark_agreement(prs_obj: dict) -> dict:
    baseline_label = baseline_single_snp_label(prs_obj)
    baseline_dir = _label_direction(baseline_label)
    prs_dir = prs_bucket_direction(prs_obj.get("bucket", "unknown"))

    comparable = baseline_dir in {"high", "low", "typical"} and prs_dir in {"high", "low", "typical"}
    agrees = int(comparable and baseline_dir == prs_dir)

    return {
        "trait_id": prs_obj.get("trait_id", ""),
        "comparable": int(comparable),
        "baseline_label": baseline_label,
        "prs_bucket": prs_obj.get("bucket", ""),
        "agrees": agrees,
    }
