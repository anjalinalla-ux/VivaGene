import re
from typing import Any


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "has", "have", "had", "can", "may", "might", "could", "into", "than", "then",
    "your", "you", "their", "them", "about", "over", "under", "trait", "traits",
}


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    return {t for t in tokens if len(t) >= 4 and t not in STOPWORDS}


def enforce_evidence_only(draft_text: str, evidence_passages: list[dict]) -> tuple[bool, list[str]]:
    """Heuristic scaffold: flag sentences with weak keyword overlap with evidence."""
    draft = str(draft_text or "").strip()
    if not draft:
        return True, []

    evidence_tokens = set()
    for p in evidence_passages or []:
        if isinstance(p, dict):
            evidence_tokens |= _tokenize(p.get("quote", ""))

    if not evidence_tokens:
        return False, ["No evidence passages available for claim validation."]

    unsupported = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft) if s.strip()]
    for s in sentences:
        s_tokens = _tokenize(s)
        if not s_tokens:
            continue
        overlap = len(s_tokens & evidence_tokens)
        ratio = overlap / max(1, len(s_tokens))
        if ratio < 0.2:
            unsupported.append(s)

    return len(unsupported) == 0, unsupported
