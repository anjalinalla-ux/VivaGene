#!/usr/bin/env python3
import json
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    packs_dir = project_root / "trait_study_packs"
    out_path = project_root / "evidence_corpus" / "snippets.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    packs = []
    for fp in sorted(packs_dir.rglob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            packs.append(data)

    lines = []
    for pack in packs:
        trait_id = str(pack.get("trait_id", "")).strip()
        if not trait_id:
            continue
        title = str(pack.get("trait_name", trait_id)).strip()
        genes = []
        for v in pack.get("variants", []) if isinstance(pack.get("variants", []), list) else []:
            if isinstance(v, dict):
                g = str(v.get("gene", "")).strip()
                if g and g not in genes:
                    genes.append(g)

        for i in (1, 2):
            obj = {
                "citation_id": f"CURATED:{trait_id}:{i}",
                "title": f"Curated placeholder for {title}",
                "year": 2020,
                "source": "CuratedLocal",
                "url": "",
                "genes": genes,
                "trait_ids": [trait_id],
                "text": "Curated evidence placeholder – replace with real snippet.",
            }
            lines.append(json.dumps(obj, ensure_ascii=False))

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote {len(lines)} snippets to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
