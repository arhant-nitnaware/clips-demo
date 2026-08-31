"""Tests for CLAP audio endpoints."""
from conftest import assert_ok, assert_client_error, assert_time_taken, assert_score_in_unit_range


# --------------------------------------------------------------------------- #
# /clap/search
# --------------------------------------------------------------------------- #

def test_search_basic(session, base_url, sample_audio):
    with open(sample_audio, "rb") as f:
        resp = session.post(
            f"{base_url}/clap/search",
            files={"file": f},
            data={"query": "a continuous tone", "segment_seconds": 3.0, "top_k": 4},
        )
    assert_ok(resp)
    body = resp.json()
    assert body["query"] == "a continuous tone"
    assert_time_taken(body)
    results = body["results"]
    assert 0 < len(results) <= 4
    for r in results:
        assert r["end_time"] > r["start_time"] >= 0
        assert_score_in_unit_range(r["score"])
        assert r["audio"].startswith("data:audio/"), "segment audio is not a data URI"


def test_search_results_ranked(session, base_url, sample_audio):
    with open(sample_audio, "rb") as f:
        resp = session.post(
            f"{base_url}/clap/search",
            files={"file": f},
            data={"query": "birds chirping", "segment_seconds": 2.0, "top_k": 5},
        )
    assert_ok(resp)
    scores = [r["score"] for r in resp.json()["results"]]
    assert scores == sorted(scores, reverse=True), f"segments not ranked: {scores}"


def test_search_segment_seconds_changes_segment_count(session, base_url, sample_audio):
    with open(sample_audio, "rb") as f:
        resp_fine = session.post(
            f"{base_url}/clap/search",
            files={"file": f},
            data={"query": "tone", "segment_seconds": 1.0, "top_k": 50},
        )
    assert_ok(resp_fine)
    with open(sample_audio, "rb") as f:
        resp_coarse = session.post(
            f"{base_url}/clap/search",
            files={"file": f},
            data={"query": "tone", "segment_seconds": 6.0, "top_k": 50},
        )
    assert_ok(resp_coarse)
    fine_n = len(resp_fine.json()["results"])
    coarse_n = len(resp_coarse.json()["results"])
    assert fine_n >= coarse_n, (
        f"finer segmentation ({fine_n} segs) should yield >= segments than "
        f"coarser ({coarse_n} segs)"
    )


def test_search_mp3_input(session, base_url, sample_audio_mp3):
    with open(sample_audio_mp3, "rb") as f:
        resp = session.post(
            f"{base_url}/clap/search",
            files={"file": f},
            data={"query": "a tone", "segment_seconds": 3.0},
        )
    assert_ok(resp)


def test_search_missing_query(session, base_url, sample_audio):
    with open(sample_audio, "rb") as f:
        resp = session.post(f"{base_url}/clap/search", files={"file": f})
    assert_client_error(resp)


def test_search_missing_file(session, base_url):
    resp = session.post(f"{base_url}/clap/search", data={"query": "birds chirping"})
    assert_client_error(resp)


def test_search_corrupt_audio(session, base_url, corrupt_file):
    with open(corrupt_file, "rb") as f:
        resp = session.post(
            f"{base_url}/clap/search",
            files={"file": f},
            data={"query": "anything"},
        )
    assert resp.status_code != 500, f"corrupt audio caused a 500: {resp.text[:300]}"


def test_search_segment_longer_than_clip(session, base_url, sample_audio):
    """segment_seconds larger than the whole clip should degrade gracefully
    to a single segment, not error."""
    with open(sample_audio, "rb") as f:
        resp = session.post(
            f"{base_url}/clap/search",
            files={"file": f},
            data={"query": "anything", "segment_seconds": 999},
        )
    assert resp.status_code != 500
    if resp.status_code == 200:
        assert len(resp.json()["results"]) >= 1


# --------------------------------------------------------------------------- #
# /clap/label
# --------------------------------------------------------------------------- #

def test_label_basic(session, base_url, sample_audio):
    with open(sample_audio, "rb") as f:
        resp = session.post(
            f"{base_url}/clap/label",
            files={"file": f},
            data={"labels_str": "birds,applause,coughing"},
        )
    assert_ok(resp)
    body = resp.json()
    assert_time_taken(body)
    results = body["results"]
    labels_seen = {r["label"] for r in results}
    assert labels_seen == {"birds", "applause", "coughing"}
    for r in results:
        assert_score_in_unit_range(r["score"], low=-0.0001, high=1.0001)


def test_label_missing_labels_str(session, base_url, sample_audio):
    with open(sample_audio, "rb") as f:
        resp = session.post(f"{base_url}/clap/label", files={"file": f})
    assert_client_error(resp)


def test_label_missing_file(session, base_url):
    resp = session.post(f"{base_url}/clap/label", data={"labels_str": "a,b"})
    assert_client_error(resp)
