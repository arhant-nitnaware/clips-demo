"""Tests for CLIP4Clip video-frame endpoints."""
from conftest import assert_ok, assert_client_error, assert_time_taken, assert_score_in_unit_range


# --------------------------------------------------------------------------- #
# /clip4clip/search
# --------------------------------------------------------------------------- #

def test_search_basic(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/search",
            files={"file": f},
            data={"query": "a yellow race car", "max_frames": 12, "top_k": 4},
        )
    assert_ok(resp)
    body = resp.json()
    assert body["query"] == "a yellow race car"
    assert_time_taken(body)
    results = body["results"]
    assert 0 < len(results) <= 4
    for r in results:
        assert isinstance(r["frame_index"], int) and r["frame_index"] >= 0
        assert_score_in_unit_range(r["score"])
        assert r["image"].startswith("data:image/"), "frame image is not a data URI"


def test_search_results_ranked(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/search",
            files={"file": f},
            data={"query": "a moving pattern", "max_frames": 10, "top_k": 5},
        )
    assert_ok(resp)
    scores = [r["score"] for r in resp.json()["results"]]
    assert scores == sorted(scores, reverse=True), f"frames not ranked: {scores}"


def test_search_top_k_larger_than_max_frames(session, base_url, sample_video):
    """top_k exceeding the number of extracted frames should clamp, not error."""
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/search",
            files={"file": f},
            data={"query": "anything", "max_frames": 4, "top_k": 50},
        )
    assert_ok(resp)
    assert len(resp.json()["results"]) <= 4


def test_search_max_frames_one(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/search",
            files={"file": f},
            data={"query": "anything", "max_frames": 1, "top_k": 4},
        )
    assert_ok(resp)
    assert len(resp.json()["results"]) == 1


def test_search_missing_query(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(f"{base_url}/clip4clip/search", files={"file": f})
    assert_client_error(resp)


def test_search_missing_file(session, base_url):
    resp = session.post(f"{base_url}/clip4clip/search", data={"query": "anything"})
    assert_client_error(resp)


def test_search_corrupt_video(session, base_url, corrupt_file):
    with open(corrupt_file, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/search",
            files={"file": f},
            data={"query": "anything"},
        )
    assert resp.status_code != 500, f"corrupt video caused a 500: {resp.text[:300]}"


def test_search_negative_top_k(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/search",
            files={"file": f},
            data={"query": "anything", "top_k": -1},
        )
    assert resp.status_code != 500
    assert_client_error(resp) if resp.status_code != 200 else None


# --------------------------------------------------------------------------- #
# /clip4clip/label
# --------------------------------------------------------------------------- #

def test_label_basic(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/label",
            files={"file": f},
            data={"labels_str": "sports,cooking,news", "max_frames": 12},
        )
    assert_ok(resp)
    body = resp.json()
    assert_time_taken(body)
    results = body["results"]
    labels_seen = {r["label"] for r in results}
    assert labels_seen == {"sports", "cooking", "news"}
    for r in results:
        assert_score_in_unit_range(r["score"], low=-0.0001, high=1.0001)


def test_label_missing_labels_str(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(f"{base_url}/clip4clip/label", files={"file": f})
    assert_client_error(resp)


def test_label_missing_file(session, base_url):
    resp = session.post(f"{base_url}/clip4clip/label", data={"labels_str": "a,b"})
    assert_client_error(resp)


# --------------------------------------------------------------------------- #
# /clip4clip/batch_search
# --------------------------------------------------------------------------- #

def test_batch_search_basic(session, base_url, sample_video, sample_video_2):
    with open(sample_video, "rb") as f1, open(sample_video_2, "rb") as f2:
        resp = session.post(
            f"{base_url}/clip4clip/batch_search",
            files=[("files", f1), ("files", f2)],
            data={"query": "a yellow race car", "max_frames": 8, "top_k": 4},
        )
    assert_ok(resp)
    body = resp.json()
    assert_time_taken(body)
    results = body["results"]
    assert 0 < len(results) <= 4
    for r in results:
        assert_score_in_unit_range(r["score"])
        # Batch results should be traceable to a source video in some form.
        assert any(k in r for k in ("video_name", "video_path", "source")), (
            f"batch result missing a source-video identifier: {r}"
        )


def test_batch_search_results_ranked_globally(session, base_url, sample_video, sample_video_2):
    with open(sample_video, "rb") as f1, open(sample_video_2, "rb") as f2:
        resp = session.post(
            f"{base_url}/clip4clip/batch_search",
            files=[("files", f1), ("files", f2)],
            data={"query": "a colorful pattern", "max_frames": 8, "top_k": 6},
        )
    assert_ok(resp)
    scores = [r["score"] for r in resp.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_batch_search_single_file_behaves_like_search(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/batch_search",
            files=[("files", f)],
            data={"query": "anything", "max_frames": 6, "top_k": 3},
        )
    assert_ok(resp)
    assert len(resp.json()["results"]) <= 3


def test_batch_search_missing_files(session, base_url):
    resp = session.post(
        f"{base_url}/clip4clip/batch_search", data={"query": "anything"}
    )
    assert_client_error(resp)


def test_batch_search_missing_query(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/clip4clip/batch_search", files=[("files", f)]
        )
    assert_client_error(resp)
