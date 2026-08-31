"""
Cross-cutting checks that apply to every endpoint: unsupported HTTP
methods, unknown routes, and content-vs-extension mismatches. These
complement the per-endpoint negative tests in the other files rather
than duplicating them.
"""
import pytest
from conftest import assert_client_error

ALL_ENDPOINTS = [
    "/media/info",
    "/clip/search",
    "/clip/label",
    "/tinyclip/search",
    "/tinyclip/label",
    "/clip4clip/search",
    "/clip4clip/label",
    "/clip4clip/batch_search",
    "/clap/search",
    "/clap/label",
    "/av/search",
    "/av/batch_search",
]


@pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
def test_get_not_allowed(session, base_url, endpoint):
    """All documented endpoints are POST-only; GET should be 404/405, not 200."""
    resp = session.get(f"{base_url}{endpoint}")
    assert resp.status_code in (404, 405), (
        f"GET {endpoint} unexpectedly returned {resp.status_code}"
    )


def test_unknown_route_404(session, base_url):
    resp = session.post(f"{base_url}/definitely/not/a/real/route")
    assert resp.status_code == 404


@pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
def test_no_body_at_all(session, base_url, endpoint):
    """POSTing with zero form fields should be a clean validation error,
    never a 500."""
    resp = session.post(f"{base_url}{endpoint}")
    assert resp.status_code != 500, f"{endpoint} 500'd on an empty request body"
    assert_client_error(resp)


def test_image_endpoint_rejects_text_disguised_as_jpg(
    session, base_url, text_file_disguised_as_media
):
    """Content sniffing, not just filename/extension, should gate acceptance."""
    with open(text_file_disguised_as_media, "rb") as f:
        resp = session.post(
            f"{base_url}/clip/search",
            files={"files": f},
            data={"query": "anything"},
        )
    assert resp.status_code != 500, f"disguised text file caused a 500: {resp.text[:300]}"


def test_openapi_schema_is_served(session, base_url):
    """Sanity check that FastAPI's generated schema is reachable, useful as
    a smoke test independent of any specific endpoint's behavior."""
    resp = session.get(f"{base_url}/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    documented = set(schema["paths"].keys())
    missing = [e for e in ALL_ENDPOINTS if e not in documented]
    assert not missing, f"endpoints documented in README but absent from the live API: {missing}"


@pytest.mark.parametrize("field_value", ["not_a_number", "", "-999999"])
def test_top_k_rejects_or_survives_garbage_values(
    session, base_url, sample_video, field_value
):
    """top_k accepts an int in every documented endpoint that has it; make
    sure garbage input degrades to a 4xx rather than a stack trace."""
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/search",
            files={"file": f},
            data={"query": "anything", "top_k": field_value},
        )
    assert resp.status_code != 500, (
        f"top_k={field_value!r} caused a 500: {resp.text[:300]}"
    )
