import csv
import re
from pathlib import Path
from collections import defaultdict, Counter

from genomics_interpreter import parse_genotype_file, match_traits

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "2026-03-02T12-14_export.csv"
PATIENT_PATH = ROOT / "patient_A.txt"

VALID_GTS = {a + b for a in "ACGT" for b in "ACGT"}


def split_rsids(value: str):
    if value is None:
        return []
    return [p.strip() for p in re.split(r"[;,]+", str(value)) if p.strip()]


def clean_gt(value: str):
    if value is None:
        return None
    gt = "".join(ch for ch in str(value).upper() if ch in "ACGT")
    if len(gt) == 2 and gt in VALID_GTS:
        return gt
    return None


def gt_from_effect_allele(value: str):
    a = (value or "").strip().upper()
    if a in {"A", "C", "G", "T"}:
        partner = "G" if a != "G" else "A"
        return a + partner
    return None


def build_lookup_and_counts():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = {c.lower().strip(): c for c in (reader.fieldnames or [])}

        def pick(*names):
            for n in names:
                if n in fields:
                    return fields[n]
            return None

        rsid_col = pick("rsid", "snp", "variant_id", "rs")
        gt_col = pick("genotype", "risk_genotype", "effect_genotype")
        allele_col = pick("effect_allele", "risk_allele")

        # pick dominant genotype per rsid (same rule as generator)
        wanted = defaultdict(Counter)
        rows = []
        for row in reader:
            rsids = [r for r in split_rsids(row.get(rsid_col) if rsid_col else "") if r.lower().startswith("rs")]
            if not rsids:
                continue
            rows.append((rsids, row))
            gt = clean_gt(row.get(gt_col) if gt_col else None)
            if not gt:
                gt = gt_from_effect_allele(row.get(allele_col) if allele_col else None)
            if not gt:
                gt = "AG"
            for rsid in rsids:
                wanted[rsid][gt] += 1

        chosen = {rsid: cnt.most_common(1)[0][0] for rsid, cnt in wanted.items()}

        # build match_traits-compatible lookup dict keyed by (rsid, genotype)
        lookup = {}
        total_rows_with_rsid = 0
        for rsids, row in rows:
            total_rows_with_rsid += 1
            for rsid in rsids:
                gt_key = chosen.get(rsid, "AG")
                lookup[(rsid, gt_key)] = {
                    "rsid": rsid,
                    "genotype": gt_key,
                    "trait_name": row.get("trait_name", row.get("title", "")),
                    "gene": row.get("gene", ""),
                    "category": row.get("category", ""),
                    "effect_label": row.get("effect_label", ""),
                    "effect_level": row.get("effect_level", ""),
                    "explanation": row.get("explanation", ""),
                    "evidence_strength": row.get("evidence_strength", ""),
                }

    return lookup, total_rows_with_rsid, len(chosen)


def main():
    lookup, total_rows, total_unique_rsids = build_lookup_and_counts()
    variants = parse_genotype_file(str(PATIENT_PATH))
    matched = match_traits(lookup, variants)

    print(f"Matched {len(matched)} / {total_rows} traits")
    if total_rows:
        print(f"Coverage: {len(matched) / total_rows * 100:.2f}%")
    print(f"Unique rsids in CSV: {total_unique_rsids}")
    print(f"Maximum possible with one genotype per rsid in this parser: {total_unique_rsids}")


if __name__ == "__main__":
    main()
