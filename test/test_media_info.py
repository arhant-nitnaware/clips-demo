"""Tests for POST /media/info"""
import pytest
from conftest import assert_ok, assert_client_error

ENDPOINT = "/media/info"


def test_media_info_video(session, base_url, sample_video):
    with open(sample_video, "rb") as f:
        resp = session.post(f"{base_url}{ENDPOINT}", files={"file": f})
    assert_ok(resp)
    body = resp.json()
    for key in ("duration", "fps", "frame_count", "resolution", "sample_rate", "file_size_mb"):
        assert key in body, f"missing '{key}' in {body}"
    assert body["duration"] > 0
    assert body["fps"] > 0
    assert body["frame_count"] > 0
    assert isinstance(body["resolution"], str) and "x" in body["resolution"]
    assert body["file_size_mb"] > 0


def test_media_info_audio(session, base_url, sample_audio):
    with open(sample_audio, "rb") as f:
        resp = session.post(f"{base_url}{ENDPOINT}", files={"file": f})
    assert_ok(resp)
    body = resp.json()
    assert body["duration"] > 0
    assert body["sample_rate"] > 0
    assert body["file_size_mb"] > 0
    # audio-only input: video-specific fields should be absent, null, or 0 --
    # any of those is acceptable, a nonsensical positive value is not.
    if body.get("fps") not in (None, 0):
        pytest.fail(f"unexpected fps for audio-only file: {body.get('fps')}")


def test_media_info_image(session, base_url, sample_image):
    with open(sample_image, "rb") as f:
        resp = session.post(f"{base_url}{ENDPOINT}", files={"file": f})
    # An image has no duration/framerate in the usual sense; accept either
    # a 200 with sensible defaults or a clean 4xx rejecting the modality.
    assert resp.status_code in (200, 400, 422), resp.text[:300]
    if resp.status_code == 200:
        body = resp.json()
        assert body["file_size_mb"] > 0


def test_media_info_via_path_param(session, base_url, sample_video):
    """README documents both an uploaded `file` and a server-local
    `video_path` / `image_path` input mode."""
    resp = session.post(f"{base_url}{ENDPOINT}", data={"video_path": sample_video})
    assert resp.status_code in (200, 404, 422), resp.text[:300]


def test_media_info_missing_file(session, base_url):
    resp = session.post(f"{base_url}{ENDPOINT}")
    assert_client_error(resp)


def test_media_info_corrupt_file(session, base_url, corrupt_file):
    with open(corrupt_file, "rb") as f:
        resp = session.post(f"{base_url}{ENDPOINT}", files={"file": f})
    # Garbage bytes must not crash the server.
    assert resp.status_code != 500, f"corrupt input caused a 500: {resp.text[:300]}"
    assert_client_error(resp) if resp.status_code != 200 else None


def test_media_info_empty_file(session, base_url, empty_file):
    with open(empty_file, "rb") as f:
        resp = session.post(f"{base_url}{ENDPOINT}", files={"file": f})
    assert resp.status_code != 500, f"empty input caused a 500: {resp.text[:300]}"


def test_media_info_nonexistent_server_path(session, base_url):
    resp = session.post(f"{base_url}{ENDPOINT}", data={"video_path": "/no/such/file.mp4"})
    assert_client_error(resp)
