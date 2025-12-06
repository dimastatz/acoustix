""" test audio module """
from acoustix.core.audio import compare_voice_similarity


def test_compare_voice_similarity():
    """test compare_voice_similarity function"""
    score = compare_voice_similarity("voice_sample_1.wav", "voice_sample_2.wav")
    assert 0.0 <= score <= 1.0
