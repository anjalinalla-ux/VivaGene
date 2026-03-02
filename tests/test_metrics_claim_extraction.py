from validation.metrics import extract_atomic_claims


def test_extract_atomic_claims_count():
    text = (
        "This trait suggests a higher tendency in this profile. "
        "Results may vary by context. "
        "No citation here"
    )
    claims = extract_atomic_claims(text)
    assert len(claims) == 2
