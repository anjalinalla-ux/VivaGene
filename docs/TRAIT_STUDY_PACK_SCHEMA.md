# Trait Study Pack JSON Schema

This schema is designed for local, curated, expandable trait packs used by PRS + RAG scaffolding.
It is backward compatible with the earlier minimal scaffold and adds structured fields for completeness scoring.

## Top-level object
Required fields:
- `trait_id` (string): unique stable ID, e.g. `NEURO_SLEEP_DURATION_PRS_V1`
- `category` (string): one of `Neurobehavior`, `Nutrition`, `Fitness`, `Liver`
- `subcategory` (string): human-friendly subgroup label
- `trait_name` (string): public-facing trait title
- `description_public` (string): 1-2 sentence friendly summary
- `is_optional` (boolean): use `true` for optional tracks (e.g. liver optional)
- `variants` (array of variant objects): at least one element expected
- `evidence` (array of evidence objects): at least one element expected
- `limitations` (array[string]): interpretation caveats
- `safety` (array[string]): safety/disclaimer strings
- `metadata` (object): curation metadata

Optional/additive fields:
- `completeness` (object): computed score snapshot (`score`, `missing`, `status`)
- `_pack_path` (string): runtime helper path (in-memory only)
- `_completeness` (object): runtime helper copy of completeness

## Variant object
Required fields:
- `rsid` (string): `rs...` or placeholder `rsTBD_*`
- `gene` (string)
- `effect_allele` (string): allele symbol
- `other_allele` (string): counterpart allele (or same if monomorphic placeholder)
- `weight` (number): additive weight scaffold (beta-like)
- `unit` (string): usually `beta`
- `genotype_map` (object): genotype key -> interpretation object
- `evidence_strength` (string): `High` | `Moderate` | `Preliminary`
- `citations` (array[string]): citation IDs, e.g. `PMID:12345678`, `DOI:10...`, `PMID:TBD`

`genotype_map` item format:
- key: genotype string (e.g. `AA`, `AG`, `GG`)
- value object fields:
  - `effect_level` (string)
  - `effect_label` (string)
  - `notes` (string, 1 sentence)

## Evidence object
Required fields:
- `citation_id` (string): e.g. `PMID:12345678`, `DOI:...`, or placeholder `PMID:TBD`
- `citation_type` (string): `PMID` | `DOI` | `URL`
- `title` (string)
- `year` (number or numeric string)
- `journal` (string)
- `url` (string)
- `quote` (string): short supported passage (1-2 sentences)
- `claim_tags` (array[string])

## Metadata object
Required fields:
- `curation_status` (string): `Complete` | `Partial` | `Placeholder`
- `last_updated` (string): `YYYY-MM-DD`
- `source_notes` (string)

## Example: completed trait (high completeness)
```json
{
  "trait_id": "NUTR_LACTOSE_PRS_V1",
  "category": "Nutrition",
  "subcategory": "Digestion",
  "trait_name": "Lactose tolerance tendency",
  "description_public": "A polygenic estimate related to typical adult lactose digestion variation.",
  "is_optional": false,
  "variants": [
    {
      "rsid": "rs4988235",
      "gene": "LCT",
      "effect_allele": "T",
      "other_allele": "C",
      "weight": 0.0123,
      "unit": "beta",
      "genotype_map": {
        "TT": {"effect_level": "Higher", "effect_label": "Higher lactose tolerance tendency", "notes": "This tendency may vary by ancestry and environment."},
        "CT": {"effect_level": "Typical", "effect_label": "Typical lactose tolerance tendency", "notes": "This tendency may vary by ancestry and environment."},
        "CC": {"effect_level": "Lower", "effect_label": "Lower lactose tolerance tendency", "notes": "This tendency may vary by ancestry and environment."}
      },
      "evidence_strength": "Moderate",
      "citations": ["PMID:12345678"]
    }
  ],
  "evidence": [
    {
      "citation_id": "PMID:12345678",
      "citation_type": "PMID",
      "title": "Genetic variation near LCT and lactose persistence",
      "year": 2020,
      "journal": "Nutrigenetics Journal",
      "url": "https://example.org/lct",
      "quote": "Population studies report that common variants near LCT may contribute to lactose persistence variation.",
      "claim_tags": ["lactose", "digestion", "general_population"]
    }
  ],
  "limitations": ["Not deterministic", "Population-specific effects possible"],
  "safety": ["Educational only", "Not medical advice"],
  "metadata": {
    "curation_status": "Complete",
    "last_updated": "2026-02-15",
    "source_notes": "Curated from local CSV and scaffold templates."
  },
  "completeness": {"score": 100, "missing": [], "status": "Complete"}
}
```

## Example: partially completed trait (coming soon)
```json
{
  "trait_id": "NEURO_STRESS_REACTIVITY_PRS_V1",
  "category": "Neurobehavior",
  "subcategory": "Stress",
  "trait_name": "Stress reactivity tendency",
  "description_public": "This trait pack is scaffolded and will be expanded with curated variants and citations.",
  "is_optional": false,
  "variants": [
    {
      "rsid": "rsTBD_1",
      "gene": "GENE_TBD",
      "effect_allele": "A",
      "other_allele": "G",
      "weight": 0.0,
      "unit": "beta",
      "genotype_map": {
        "AA": {"effect_level": "Unknown", "effect_label": "Coming soon", "notes": "Variant-level interpretation is in progress."}
      },
      "evidence_strength": "Preliminary",
      "citations": ["PMID:TBD"]
    }
  ],
  "evidence": [
    {
      "citation_id": "PMID:TBD",
      "citation_type": "PMID",
      "title": "Placeholder evidence for stress reactivity tendency",
      "year": 2020,
      "journal": "TBD Journal",
      "url": "https://example.org/tbd",
      "quote": "Evidence retrieval pending for this scaffold trait.",
      "claim_tags": ["stress", "placeholder", "general_population"]
    }
  ],
  "limitations": ["Not deterministic"],
  "safety": ["Educational only", "Not medical advice"],
  "metadata": {
    "curation_status": "Placeholder",
    "last_updated": "2026-02-15",
    "source_notes": "Placeholder created to complete ontology coverage."
  },
  "completeness": {"score": 65, "missing": ["metadata_complete"], "status": "Partial"}
}
```
