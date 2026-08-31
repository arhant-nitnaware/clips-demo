"""Tests for the fused Audio+Video endpoints."""
from conftest import assert_ok, assert_client_error, assert_time_taken, assert_score_in_unit_range


# --------------------------------------------------------------------------- #
# /av/search
# --------------------------------------------------------------------------- #

def test_search_basic(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/av/search",
            files={"file": f},
            data={
                "query": "helicopter blades roaring",
                "visual_weight": 0.5,
                "audio_weight": 0.5,
                "segment_seconds": 5.0,
                "max_frames": 8,
                "top_k": 4,
            },
        )
    assert_ok(resp)
    body = resp.json()
    assert body["query"] == "helicopter blades roaring"
    assert_time_taken(body)
    results = body["results"]
    assert 0 < len(results) <= 4
    for r in results:
        assert r["end_time"] > r["start_time"] >= 0
        assert_score_in_unit_range(r["visual_score"])
        assert_score_in_unit_range(r["audio_score"])
        assert_score_in_unit_range(r["fused_score"])
        assert r["image"].startswith("data:image/")
        assert "video_name" in r


def test_search_fused_score_matches_weighted_sum(session, base_url, sample_video):
    """fused_score should track the documented visual_weight/audio_weight
    formula (fused = w_v * visual + w_a * audio), within tolerance for any
    internal normalization."""
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/av/search",
            files={"file": f},
            data={
                "query": "a tone",
                "visual_weight": 0.5,
                "audio_weight": 0.5,
                "segment_seconds": 5.0,
                "max_frames": 8,
                "top_k": 4,
            },
        )
    assert_ok(resp)
    for r in resp.json()["results"]:
        expected = 0.5 * r["visual_score"] + 0.5 * r["audio_score"]
        assert abs(r["fused_score"] - expected) < 0.15, (
            f"fused_score {r['fused_score']} far from weighted sum {expected} "
            f"(visual={r['visual_score']}, audio={r['audio_score']})"
        )


def test_search_results_ranked_by_fused_score(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/av/search",
            files={"file": f},
            data={
                "query": "a tone",
                "visual_weight": 0.5,
                "audio_weight": 0.5,
                "segment_seconds": 5.0,
                "max_frames": 8,
                "top_k": 5,
            },
        )
    assert_ok(resp)
    fused = [r["fused_score"] for r in resp.json()["results"]]
    assert fused == sorted(fused, reverse=True), f"results not ranked by fused_score: {fused}"


def test_search_weight_skew_favors_visual(session, base_url, sample_video):
    """Pushing visual_weight to the extreme should make fused_score track
    visual_score closely, and vice versa for audio."""
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/av/search",
            files={"file": f},
            data={
                "query": "a tone",
                "visual_weight": 1.0,
                "audio_weight": 0.0,
                "segment_seconds": 5.0,
                "max_frames": 8,
                "top_k": 4,
            },
        )
    assert_ok(resp)
    for r in resp.json()["results"]:
        assert abs(r["fused_score"] - r["visual_score"]) < 0.15


def test_search_weights_not_summing_to_one(session, base_url, sample_video):
    """Weights that don't sum to 1 (e.g. 0.3 / 0.7) shouldn't crash the
    server, whether it normalizes them or takes them literally."""
    with open(sample_video, "rb") as f:
        resp = session.post(
            f"{base_url}/av/search",
            files={"file": f},
            data={
                "query": "a tone",
                "visual_weight": 0.3,
                "audio_weight": 0.9,
                "segment_seconds": 5.0,
                "max_frames": 8,
                "top_k": 4,
            },
        )
    assert resp.status_code != 500


def test_search_video_without_audio_track(session, base_url, silent_video):
    """A video with no audio stream should error cleanly or degrade to a
    visual-only score, never 500."""
    with open(silent_video, "rb") as f:
        resp = session.post(
            f"{base_url}/av/search",
            files={"file": f},
            data={
                "query": "anything",
                "visual_weight": 0.5,
                "audio_weight": 0.5,
                "segment_seconds": 3.0,
                "max_frames": 6,
                "top_k": 4,
            },
        )
    assert resp.status_code != 500, f"silent video caused a 500: {resp.text[:300]}"


def test_search_missing_query(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(f"{base_url}/av/search", files={"file": f})
    assert_client_error(resp)


def test_search_missing_file(session, base_url):
    resp = session.post(f"{base_url}/av/search", data={"query": "anything"})
    assert_client_error(resp)


def test_search_corrupt_video(session, base_url, corrupt_file):
    with open(corrupt_file, "rb") as f:
        resp = session.post(
            f"{base_url}/av/search",
            files={"file": f},
            data={"query": "anything"},
        )
    assert resp.status_code != 500, f"corrupt video caused a 500: {resp.text[:300]}"


# --------------------------------------------------------------------------- #
# /av/batch_search
# --------------------------------------------------------------------------- #

def test_batch_search_basic(session, base_url, sample_video, sample_video_2):
    with open(sample_video, "rb") as f1, open(sample_video_2, "rb") as f2:
        resp = session.post(
            f"{base_url}/av/batch_search",
            files=[("files", f1), ("files", f2)],
            data={
                "query": "helicopter blades roaring",
                "visual_weight": 0.5,
                "audio_weight": 0.5,
                "segment_seconds": 5.0,
                "max_frames": 8,
                "top_k": 4,
            },
        )
    assert_ok(resp)
    body = resp.json()
    assert_time_taken(body)
    results = body["results"]
    assert 0 < len(results) <= 4
    video_names_seen = {r["video_name"] for r in results}
    assert video_names_seen, "no video_name identifiers found in batch results"


def test_batch_search_ranked_globally_across_videos(session, base_url, sample_video, sample_video_2):
    with open(sample_video, "rb") as f1, open(sample_video_2, "rb") as f2:
        resp = session.post(
            f"{base_url}/av/batch_search",
            files=[("files", f1), ("files", f2)],
            data={
                "query": "a tone",
                "visual_weight": 0.5,
                "audio_weight": 0.5,
                "segment_seconds": 5.0,
                "max_frames": 8,
                "top_k": 6,
            },
        )
    assert_ok(resp)
    fused = [r["fused_score"] for r in resp.json()["results"]]
    assert fused == sorted(fused, reverse=True)


def test_batch_search_missing_files(session, base_url):
    resp = session.post(f"{base_url}/av/batch_search", data={"query": "anything"})
    assert_client_error(resp)


def test_batch_search_missing_query(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(f"{base_url}/av/batch_search", files=[("files", f)])
    assert_client_error(resp)
