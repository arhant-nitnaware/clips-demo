"""
Tests for POST /clip/search, /clip/label, /tinyclip/search, /tinyclip/label.

CLIP and TinyCLIP share an identical documented contract, so the same
test bodies are parametrized across both prefixes instead of duplicated.
"""
import pytest
from conftest import assert_ok, assert_client_error, assert_time_taken, assert_score_in_unit_range

PREFIXES = ["clip", "tinyclip"]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_search_single_image(session, base_url, sample_image, prefix):
    with open(sample_image, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/search",
            files={"files": f},
            data={"query": "a red sports car"},
        )
    assert_ok(resp)
    body = resp.json()
    assert body["query"] == "a red sports car"
    assert_time_taken(body)
    assert isinstance(body["results"], list) and len(body["results"]) == 1
    result = body["results"][0]
    assert "image_path" in result
    assert_score_in_unit_range(result["score"])


@pytest.mark.parametrize("prefix", PREFIXES)
def test_search_multiple_images_ranked(session, base_url, sample_image, sample_image_2, prefix):
    with open(sample_image, "rb") as f1, open(sample_image_2, "rb") as f2:
        resp = session.post(
            f"{base_url}/{prefix}/search",
            files=[("files", f1), ("files", f2)],
            data={"query": "a colorful test pattern"},
        )
    assert_ok(resp)
    results = resp.json()["results"]
    assert len(results) == 2
    scores = [r["score"] for r in results]
    # Results should be ranked, i.e. sorted by score descending.
    assert scores == sorted(scores, reverse=True), f"results not ranked: {scores}"


@pytest.mark.parametrize("prefix", PREFIXES)
def test_search_missing_query(session, base_url, sample_image, prefix):
    with open(sample_image, "rb") as f:
        resp = session.post(f"{base_url}/{prefix}/search", files={"files": f})
    assert_client_error(resp)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_search_missing_files(session, base_url, prefix):
    resp = session.post(f"{base_url}/{prefix}/search", data={"query": "anything"})
    assert_client_error(resp)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_search_empty_query_string(session, base_url, sample_image, prefix):
    with open(sample_image, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/search",
            files={"files": f},
            data={"query": ""},
        )
    # An empty query is either rejected outright or degrades gracefully;
    # either way it must not 500.
    assert resp.status_code != 500


@pytest.mark.parametrize("prefix", PREFIXES)
def test_search_non_image_file(session, base_url, sample_audio, prefix):
    with open(sample_audio, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/search",
            files={"files": f},
            data={"query": "a red sports car"},
        )
    assert resp.status_code != 500
    assert_client_error(resp) if resp.status_code != 200 else None


@pytest.mark.parametrize("prefix", PREFIXES)
def test_search_corrupt_image(session, base_url, corrupt_file, prefix):
    with open(corrupt_file, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/search",
            files={"files": f},
            data={"query": "anything"},
        )
    assert resp.status_code != 500, f"corrupt image caused a 500: {resp.text[:300]}"


@pytest.mark.parametrize("prefix", PREFIXES)
def test_label_single_image(session, base_url, sample_image, prefix):
    with open(sample_image, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/label",
            files={"file": f},
            data={"labels_str": "nature,car,city,dog"},
        )
    assert_ok(resp)
    body = resp.json()
    assert_time_taken(body)
    results = body["results"]
    assert len(results) == 4
    labels_seen = {r["label"] for r in results}
    assert labels_seen == {"nature", "car", "city", "dog"}
    for r in results:
        assert_score_in_unit_range(r["score"], low=-0.0001, high=1.0001)
    # zero-shot classification scores should be sorted, highest confidence first
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), f"label results not ranked: {scores}"


@pytest.mark.parametrize("prefix", PREFIXES)
def test_label_scores_sum_to_one(session, base_url, sample_image, prefix):
    """Zero-shot label scores are typically a softmax distribution."""
    with open(sample_image, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/label",
            files={"file": f},
            data={"labels_str": "cat,dog,tree"},
        )
    assert_ok(resp)
    total = sum(r["score"] for r in resp.json()["results"])
    assert abs(total - 1.0) < 0.02, f"label scores don't sum to ~1: {total}"


@pytest.mark.parametrize("prefix", PREFIXES)
def test_label_single_label(session, base_url, sample_image, prefix):
    with open(sample_image, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/label",
            files={"file": f},
            data={"labels_str": "car"},
        )
    assert_ok(resp)
    results = resp.json()["results"]
    assert len(results) == 1
    assert abs(results[0]["score"] - 1.0) < 0.02


@pytest.mark.parametrize("prefix", PREFIXES)
def test_label_messy_labels_str_is_trimmed(session, base_url, sample_image, prefix):
    """Extra whitespace/commas in labels_str shouldn't produce empty labels."""
    with open(sample_image, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/label",
            files={"file": f},
            data={"labels_str": " nature ,  car ,,city"},
        )
    assert resp.status_code != 500
    if resp.status_code == 200:
        labels_seen = {r["label"].strip() for r in resp.json()["results"]}
        assert "" not in labels_seen, f"empty label leaked through: {labels_seen}"


@pytest.mark.parametrize("prefix", PREFIXES)
def test_label_missing_labels_str(session, base_url, sample_image, prefix):
    with open(sample_image, "rb") as f:
        resp = session.post(f"{base_url}/{prefix}/label", files={"file": f})
    assert_client_error(resp)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_label_empty_labels_str(session, base_url, sample_image, prefix):
    with open(sample_image, "rb") as f:
        resp = session.post(
            f"{base_url}/{prefix}/label",
            files={"file": f},
            data={"labels_str": ""},
        )
    assert_client_error(resp)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_label_missing_file(session, base_url, prefix):
    resp = session.post(f"{base_url}/{prefix}/label", data={"labels_str": "a,b"})
    assert_client_error(resp)
