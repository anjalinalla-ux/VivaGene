def compute_trait_completeness(trait: dict) -> dict:
    """Compute deterministic trait completeness score and status."""
    t = trait if isinstance(trait, dict) else {}
    score = 0
    missing = []

    trait_name = (t.get("trait_name") or "").strip()
    if trait_name:
        score += 20
    else:
        missing.append("trait_name")

    user_question = (t.get("user_question") or "").strip()
    if user_question:
        score += 10
    else:
        missing.append("user_question")

    what_it_means = (t.get("what_it_means") or "").strip()
    if what_it_means:
        score += 15
    else:
        missing.append("what_it_means")

    why_it_matters = (t.get("why_it_matters") or "").strip()
    if why_it_matters:
        score += 10
    else:
        missing.append("why_it_matters")

    limitations = (t.get("limitations") or "").strip()
    if limitations:
        score += 10
    else:
        missing.append("limitations")

    variants = t.get("variants")
    if isinstance(variants, list) and len(variants) >= 1:
        score += 15
    else:
        missing.append("variants")

    tags = t.get("tags")
    if isinstance(tags, list) and len(tags) >= 2:
        score += 10
    else:
        missing.append("tags")

    research_query_hint = (t.get("research_query_hint") or "").strip()
    citation_seeds = t.get("citation_seeds")
    mechanism_keywords = t.get("mechanism_keywords")
    has_research_signal = (
        bool(research_query_hint)
        or (isinstance(citation_seeds, list) and len(citation_seeds) >= 1)
        or (isinstance(mechanism_keywords, list) and len(mechanism_keywords) >= 1)
    )
    if has_research_signal:
        score += 10
    else:
        missing.append("research_or_citations_or_mechanisms")

    score = max(0, min(100, score))
    if score < 40:
        label = "Placeholder"
    elif score < 75:
        label = "Draft"
    else:
        label = "Ready"

    return {
        "score": score,
        "label": label,
        "missing": missing,
    }


def trait_has_variants(trait: dict) -> bool:
    """True if trait has at least one minimally usable variant definition."""
    t = trait if isinstance(trait, dict) else {}
    variants = t.get("variants")
    if not isinstance(variants, list) or len(variants) < 1:
        return False

    for v in variants:
        if not isinstance(v, dict):
            continue
        rsid = (v.get("rsid") or "").strip()
        gene = (v.get("gene") or "").strip()
        if not rsid or not gene:
            continue

        # Full variant definitions typically include genotype_effects or genotypes.
        genotype_effects = v.get("genotype_effects")
        genotypes = v.get("genotypes")
        if isinstance(genotype_effects, dict) and len(genotype_effects) >= 1:
            return True
        if isinstance(genotypes, dict) and len(genotypes) >= 1:
            return True

        # Placeholder variant minimum (rsid + gene).
        return True

    return False
