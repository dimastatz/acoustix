""" main audio processing functions """
from typing import Dict
import librosa
import soundfile as sf


def compare_voice_similarity(voice1, voice2):
    """
    Compares two voice samples and returns a similarity score.

    Parameters:
    voice1 (str): Path to the first voice sample file.
    voice2 (str): Path to the second voice sample file.

    Returns:
    float: Similarity score between 0 and 1, where 1 means identical voices.
    """
    print(voice1, voice2)
    return 1.0  # Placeholder implementation


def get_audio_info(path: str) -> Dict:
    """
    Return a small metadata dict for an audio file using librosa + soundfile.
    Keys:
      - duration_sec: float
      - sample_rate: int
      - channels: int
      - frames: int
      - codec: str (format/subtype if available)
    """
    # duration via librosa (efficient, can use filename directly)
    duration = float(librosa.get_duration(filename=path))

    # low-level metadata via soundfile
    with sf.SoundFile(path) as f:
        sample_rate = f.samplerate
        channels = f.channels
        frames = f.frames
        fmt = f.format or "unknown"
        subtype = f.subtype or "unknown"

    return {
        "duration_sec": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "codec": f"{fmt}/{subtype}",
    }
