from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_evidence import build_trait_queries


def test_query_builder_returns_multiple_queries():
    trait = {
        "trait_id": "NB_SLEEP_001",
        "trait_name": "Sleep duration tendency",
        "category": "Neurobehavior",
        "gene": "ADORA2A",
        "rsid": "rs5751876",
    }
    queries = build_trait_queries(trait)
    assert isinstance(queries, list)
    assert len(queries) >= 2
    assert any("ADORA2A" in q for q in queries)
