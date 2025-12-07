""" test audio module """
from tests.utilities import get_resource_path
from acoustix.core.audio import get_audio_info
from acoustix.core.audio import compare_voice_similarity


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
    assert isinstance(info, dict)
    assert "duration_sec" in info
    assert "sample_rate" in info
    assert "channels" in info
    assert "frames" in info
    assert "codec" in info
