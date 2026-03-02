# Genomics Project Architecture Audit

## Current file tree summary
- `app.py`: Streamlit app (page routing, upload/report flow, report rendering, chatbot, explorer)
- `genomics_interpreter.py`: core parser + matching + report object + html/text report builders
- `data/traits.json`: primary trait DB (variant rules and effect text)
- `data/variant_rules.csv`: authoring CSV for variant rules merged into `traits.json`
- `scripts/merge_variant_rules_into_traits.py`: merge/validate generator for `traits.json`
- `rag/europe_pmc.py`: external evidence search utilities (currently network-based)
- `utils/trait_quality.py`: completeness/quality helpers

## Current flow diagram (text)
1. Upload page receives genotype file (`st.file_uploader` in `app.py`)
2. File saved to local temp (`uploaded_genome.txt`)
3. Parse variants (`parse_genotype_file` in `genomics_interpreter.py`)
4. Load trait lookup from `data/traits.json` (`load_trait_database`)
5. Match rsid+genotype (`match_traits`)
6. Build report object (`build_report_object`)
7. (Current) Enrich with evidence calls and AI trait annotations (`app.py`)
8. Render cards in Streamlit + technical text + HTML report (`generate_html_report`)
9. Save report in `st.session_state.last_report`
10. Chatbot page reads `last_report` / `last_ai_summary` as context for responses

## Where traits are defined
- Source-of-truth for runtime: `data/traits.json`
- Authoring source: `data/variant_rules.csv`
- Converter/merger: `scripts/merge_variant_rules_into_traits.py`

## Where the AI prompt is built
- Report generation helper(s): `generate_with_fallback_model` and trait/card prompt builders in `app.py`
- Legacy summary helper also exists in `genomics_interpreter.py` (`generate_ai_summary`)

## Where HTML report is generated
- `generate_html_report` in `genomics_interpreter.py`
- Embedded via `st.components.v1.html(...)` in Upload & Report flow in `app.py`

## Where chatbot reads context
- Lifestyle chatbot section in `app.py`
- Reads from `st.session_state.last_report` and `st.session_state.last_ai_summary`

## Current failure points and guards
- Missing or invalid upload file
  - Guard: `st.warning` + early stop
- Malformed genotype lines
  - Guard: parser skips short/invalid lines
- Missing `data/traits.json`
  - Guard: `load_trait_database` returns empty dict
- Unsupported genotype formats (`--`, one-letter, malformed)
  - Guard: skip + warnings in report
- AI provider not configured/unavailable
  - Guard: local fallbacks in app helper
- Evidence retrieval errors
  - Guard: try/except with fallback text and empty citations
- Potential UI blanking from exceptions
  - Guard now should include top-level report try/except + traceback expander (added in scaffold step)

## Integration points for polygenic + local RAG scaffold
- After `matched_traits` and `report` creation in Upload & Report flow in `app.py`
- Add local study-pack load, PRS computation, and local evidence retrieval
- Attach to report keys:
  - `report["polygenic"]`
  - `report["evidence"]`
  - `report["trust"]`
