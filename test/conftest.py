"""
Shared fixtures for the Multimodal AI Inference API test suite.

The suite treats the API as a black box: it drives a *running* instance
(`python api.py`) over HTTP, the same way the documented `curl` examples do.

Configuration
-------------
API_BASE_URL   Base URL of the running FastAPI server. Defaults to
                http://localhost:8000.
TEST_ASSETS_DIR Optional path to a folder containing real sample media
                (e.g. the repo's `dataset/` and `demo_videos/` folders).
                When set, real files are used instead of the synthetic
                ffmpeg-generated fixtures below, which is recommended for
                a final pre-release run since it exercises the real model
                weights against real content instead of pure noise.

If the server isn't reachable, or ffmpeg is unavailable and no
TEST_ASSETS_DIR is supplied, tests are skipped rather than failed --
a skipped suite is a signal to go start the server / install ffmpeg,
not a red build.
"""
import os
import shutil
import subprocess
import wave
import struct
import math

import pytest
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
TEST_ASSETS_DIR = os.environ.get("TEST_ASSETS_DIR")
REQUEST_TIMEOUT = float(os.environ.get("API_TEST_TIMEOUT", "60"))

FFMPEG = shutil.which("ffmpeg")


# --------------------------------------------------------------------------- #
# Server availability
# --------------------------------------------------------------------------- #

def _server_is_up() -> bool:
    for path in ("/docs", "/openapi.json", "/"):
        try:
            r = requests.get(f"{API_BASE_URL}{path}", timeout=3)
            if r.status_code < 500:
                return True
        except requests.RequestException:
            continue
    return False


@pytest.fixture(scope="session", autouse=True)
def _require_server():
    if not _server_is_up():
        pytest.skip(
            f"API server not reachable at {API_BASE_URL}. "
            f"Start it with `python api.py` or set API_BASE_URL.",
            allow_module_level=True,
        )


@pytest.fixture(scope="session")
def base_url():
    return API_BASE_URL


@pytest.fixture(scope="session")
def session():
    s = requests.Session()
    yield s
    s.close()


# --------------------------------------------------------------------------- #
# Synthetic media generation (fallback when TEST_ASSETS_DIR isn't provided)
# --------------------------------------------------------------------------- #

def _run_ffmpeg(args):
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    if TEST_ASSETS_DIR:
        return TEST_ASSETS_DIR
    return str(tmp_path_factory.mktemp("synthetic_media"))


@pytest.fixture(scope="session")
def sample_image(media_dir):
    """A small, valid JPEG."""
    if TEST_ASSETS_DIR:
        p = os.path.join(TEST_ASSETS_DIR, "images", "car1.jpg")
        if os.path.exists(p):
            return p
    if not FFMPEG:
        pytest.skip("ffmpeg not available to synthesize a test image")
    path = os.path.join(media_dir, "sample_image.jpg")
    _run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=1",
        "-frames:v", "1", path,
    ])
    return path


@pytest.fixture(scope="session")
def sample_image_2(media_dir):
    """A second, visually distinct JPEG -- for batch/ranking tests."""
    if not FFMPEG:
        pytest.skip("ffmpeg not available to synthesize a test image")
    path = os.path.join(media_dir, "sample_image_2.jpg")
    _run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=blue:size=320x240",
        "-frames:v", "1", path,
    ])
    return path


@pytest.fixture(scope="session")
def sample_audio(media_dir):
    """A short sine-wave WAV. Most CLAP/librosa-style backends decode WAV
    fine even though the README examples use mp3."""
    if TEST_ASSETS_DIR:
        p = os.path.join(TEST_ASSETS_DIR, "audio", "birds-5.mp3")
        if os.path.exists(p):
            return p
    if not FFMPEG:
        pytest.skip("ffmpeg not available to synthesize test audio")
    path = os.path.join(media_dir, "sample_audio.wav")
    _run_ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-ar", "16000", "-ac", "1", path,
    ])
    return path


@pytest.fixture(scope="session")
def sample_audio_mp3(media_dir):
    """Same tone, encoded as mp3, to match the documented content type."""
    if not FFMPEG:
        pytest.skip("ffmpeg not available to synthesize test audio")
    path = os.path.join(media_dir, "sample_audio.mp3")
    _run_ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-ar", "44100", "-ac", "1", path,
    ])
    return path


@pytest.fixture(scope="session")
def sample_video(media_dir):
    """A short MP4 with both a visual pattern and an audible tone, so it
    works for CLIP4Clip-only and AV-fusion endpoints alike."""
    if TEST_ASSETS_DIR:
        p = os.path.join(TEST_ASSETS_DIR, "..", "demo_videos", "sample.mp4")
        if os.path.exists(p):
            return p
    if not FFMPEG:
        pytest.skip("ffmpeg not available to synthesize a test video")
    path = os.path.join(media_dir, "sample_video.mp4")
    _run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=15",
        "-f", "lavfi", "-i", "sine=frequency=523:duration=15",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture(scope="session")
def sample_video_2(media_dir):
    """A second, distinct video for batch_search tests."""
    if not FFMPEG:
        pytest.skip("ffmpeg not available to synthesize a test video")
    path = os.path.join(media_dir, "sample_video_2.mp4")
    _run_ffmpeg([
        "-f", "lavfi", "-i", "smptebars=size=320x240:rate=10:duration=15",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=15",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture(scope="session")
def silent_video(media_dir):
    """A video with a video stream but NO audio stream, for exercising
    /av/search's handling of audio-less input."""
    if not FFMPEG:
        pytest.skip("ffmpeg not available to synthesize a test video")
    path = os.path.join(media_dir, "silent_video.mp4")
    _run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", path,
    ])
    return path


@pytest.fixture(scope="session")
def corrupt_file(media_dir):
    """A file with a media-like extension but garbage bytes -- for
    negative/error-handling tests."""
    path = os.path.join(media_dir, "corrupt.jpg")
    with open(path, "wb") as f:
        f.write(os.urandom(256))
    return path


@pytest.fixture(scope="session")
def empty_file(media_dir):
    path = os.path.join(media_dir, "empty.mp4")
    open(path, "wb").close()
    return path


@pytest.fixture(scope="session")
def text_file_disguised_as_media(media_dir):
    """Plain text saved with a .jpg extension -- tests that the API
    validates content, not just the filename."""
    path = os.path.join(media_dir, "not_really_an_image.jpg")
    with open(path, "w") as f:
        f.write("this is definitely not image data\n" * 20)
    return path


# --------------------------------------------------------------------------- #
# Shared assertion helpers
# --------------------------------------------------------------------------- #

def assert_ok(resp, expect_status=200):
    assert resp.status_code == expect_status, (
        f"expected {expect_status}, got {resp.status_code}: {resp.text[:500]}"
    )


def assert_client_error(resp):
    """Malformed/missing input should yield a 4xx, never a raw 500."""
    assert 400 <= resp.status_code < 500, (
        f"expected a 4xx client error, got {resp.status_code}: {resp.text[:500]}"
    )


def assert_time_taken(body):
    assert "time_taken" in body, f"missing time_taken in {body}"
    assert isinstance(body["time_taken"], (int, float))
    assert body["time_taken"] >= 0


def assert_score_in_unit_range(score, low=-1.0001, high=1.0001):
    assert isinstance(score, (int, float)), f"score is not numeric: {score!r}"
    assert low <= score <= high, f"score {score} outside expected range"
