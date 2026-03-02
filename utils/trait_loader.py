import yaml
from pathlib import Path

def load_traits_yaml(path="data/traits.yaml"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML trait file: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("traits.yaml must be a dict: Category -> list of traits")

    # Quick validation
    required = {"id","title","snps","explanation","why_it_matters","recommendations","what_it_does_not_mean","evidence_strength"}
    for cat, traits in data.items():
        if not isinstance(traits, list):
            raise ValueError(f"Category '{cat}' must contain a list of traits")
        for t in traits:
            missing = required - set(t.keys())
            if missing:
                raise ValueError(f"Trait '{t.get('title','(no title)')}' missing fields: {sorted(missing)}")
            if not isinstance(t["snps"], list) or len(t["snps"]) > 5:
                raise ValueError(f"Trait '{t['title']}' has invalid snps list (must be <=5)")
            if not isinstance(t["recommendations"], list):
                raise ValueError(f"Trait '{t['title']}' recommendations must be a list")

    return data