# Minimal-Change Upgrade Plan

## UI constraints (unchanged)
- Keep all existing Streamlit pages and layout:
  - Home, Upload & Report, Lifestyle Chatbot, Trait Explorer, Trait Science, About, Contact
- Keep existing CSS/theme/nav untouched
- Keep current upload/report/chat/explorer features intact

## Phase 1: Scaffold (this step)
1. Add local study-pack corpus folder (`trait_study_packs/`) with placeholder JSON packs
2. Add local-only retrieval module (`rag_retrieval.py`)
3. Add PRS scaffold module (`polygenic.py`)
4. Add evidence gating scaffold (`evidence_guard.py`)
5. Wire minimal pipeline enrichment after existing SNP matching:
   - `report["polygenic"]`
   - `report["evidence"]`
   - `report["trust"]`
6. Add warnings + strict try/except so report page never blanks
7. Add pytest tests for loading, retrieval, PRS math

## Phase 2: Polygenic implementation
1. Replace placeholder study-pack weights with curated GWAS weights
2. Add allele harmonization (strand checks, reference allele normalization)
3. Add score normalization strategy (population-specific baselines)
4. Expose confidence/coverage thresholds in config

## Phase 3: RAG + citation gating
1. Expand local study pack evidence passages per trait
2. Add retrieval ranking (claim-tag match + lexical overlap)
3. Gate generator output with evidence guard before rendering
4. Require citation IDs for each claim block

## Phase 4: Validation & quality
1. Add schema validation for study packs
2. Add regression tests for report contract (`polygenic/evidence/trust` keys)
3. Add deterministic fixtures for known genotype → known PRS snapshots
4. Add error telemetry/logging hooks

## New modules and purpose
- `trait_study_packs/*.json`: local curated trait study corpus
- `rag_retrieval.py`: local retrieval of passages/citations (no web)
- `polygenic.py`: additive PRS scaffold with coverage/confidence
- `evidence_guard.py`: heuristic evidence-only contract checker

## Backward compatibility strategy
- Existing SNP matching remains the primary path
- New PRS/RAG fields are additive in `report` object only
- If packs missing or malformed, app shows warning and falls back to current single-SNP behavior
- Existing UI renders unchanged; later prompts can consume new report keys
