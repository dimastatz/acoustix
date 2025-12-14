""" test audio module """
import os
import pytest
import numpy as np
from tests.utilities import get_resource_path
from sonix.core.audio import analyze_audio
from sonix.core.audio import get_audio_info
from sonix.core.audio import compare_voice_similarity
from sonix.core.audio import merge_intervals
from sonix.core.audio import entries_from_intervals
from sonix.core.audio import diarize_with_silence


def test_compare_voice_similarity():
    """test compare_voice_similarity function"""
    score = compare_voice_similarity("voice_sample_1.wav", "voice_sample_2.wav")
    assert 0.0 <= score <= 1.0


def test_get_audio_info():
    """test get_audio_info function"""

    # prepare test file path
    test_file = get_resource_path("3081-166546-0000", "wav")

    # call the function
    info = get_audio_info(test_file)

    # assertions
    assert info["channels"] == 1
    assert info["frames"] == 168000
    assert info["duration_sec"] == 10.5
    assert info["sample_rate"] == 16000
    assert info["codec"] == "WAV/PCM_16"


def test_analyze_audio():
    """this function runs the full audio analysis pipeline on a test file"""

    test_file = get_resource_path("phone_call", "mp3")
    analysis = analyze_audio(test_file)
    assert analysis["audio_info"]["sample_rate"] == 44100
    assert analysis["audio_info"]["channels"] == 1
    assert analysis["audio_info"]["duration_sec"] == 15.801179138321995
    assert analysis["audio_info"]["frames"] == 696832
    assert analysis["audio_info"]["codec"] == "MP3/MPEG_LAYER_III"

    segments = analysis["audio_info"]["segments"]
    assert len(segments) == 21

    token = os.environ["HF_TOKEN"]
    assert token is not None, "HF_TOKEN environment variable must be set"

    with pytest.raises(Exception):
        diarize_with_silence(test_file)


def test_entries_from_intervals_empty():
    """this function runs the full audio analysis pipeline on a test file"""

    # empty intervals should produce a single silence entry covering total
    intervals = np.empty((0, 2), dtype=int)
    total = 12345
    entries = entries_from_intervals(intervals, total)
    assert entries == [(0, total, False)]


def test_merge_intervals_merges_short_silences():
    """this function runs the full audio analysis pipeline on a test file"""

    # first entry is silence, next is a short silence separated by small gap
    entries = [(0, 10, False), (11, 12, False), (20, 30, True)]
    # min_gap large enough to merge the two silences (gap length 1)
    merged = merge_intervals(entries, min_gap=2)
    # the two silence entries should be merged into (0,12,False)
    assert merged[0] == (0, 12, False)
    # the final speech entry should remain
    assert merged[-1] == (20, 30, True)
