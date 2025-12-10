""" test audio module """
from tests.utilities import get_resource_path
from sonix.core.audio import analyze_audio
from sonix.core.audio import get_audio_info
from sonix.core.audio import compare_voice_similarity


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
    assert analysis["audio_info"]["sample_rate"] == 8000
    assert analysis["audio_info"]["channels"] == 1
    assert analysis["audio_info"]["duration_sec"] == 12.5
    assert analysis["audio_info"]["frames"] == 168000
    assert analysis["audio_info"]["codec"] == "WAV/PCM_16"

    segments = analysis["audio_info"]["segments"]
    assert len(segments) == 4

    assert segments[0]["start_time_sec"] == 0.0
    assert segments[0]["end_time_sec"] == 1.5
    assert segments[0]["label"] == "silence"
    assert segments[0]["transcript"] == ""

    assert segments[1]["start_time_sec"] == 1.5
    assert segments[1]["end_time_sec"] == 5.5
    assert segments[1]["label"] == "speaker_1"
    assert segments[1]["transcript"] == "Hello, thank you for calling."

    assert segments[2]["start_time_sec"] == 5.5
    assert segments[2]["end_time_sec"] == 7.0
    assert segments[2]["label"] == "silence"
    assert segments[2]["transcript"] == ""

    assert segments[3]["start_time_sec"] == 7.0
    assert segments[3]["end_time_sec"] == 12.5
    assert segments[3]["label"] == "speaker_2"
    assert segments[3]["transcript"] == "I appreciate your help today."
