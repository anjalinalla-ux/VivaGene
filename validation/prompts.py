from __future__ import annotations

import json


def unconstrained_prompt(trait_summary: dict) -> tuple[str, str]:
    system = (
        "You are a careful educational genetics explainer. "
        "Write 1-2 short sentences in cautious language for general audiences. "
        "No medical advice. No diagnosis. No lifestyle advice."
    )
    user = (
        "Trait summary JSON:\n"
        f"{json.dumps(trait_summary, indent=2)}\n\n"
        "Return 1-2 short sentences explaining what this may mean and why it may matter."
    )
    return system, user


def rag_prompt(trait_summary: dict, evidence_block: str) -> tuple[str, str]:
    system = (
        "You are an evidence-bound scientific explainer. "
        "You may ONLY use the provided snippets. If a claim is not supported, do not include it. "
        "Output JSON only."
    )
    user = (
        "Trait summary JSON:\n"
        f"{json.dumps(trait_summary, indent=2)}\n\n"
        "Evidence snippets:\n"
        f"{evidence_block}\n\n"
        "Return JSON with explanation_1/explanation_2 and [n] citations in each sentence."
    )
    return system, user
