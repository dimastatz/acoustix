"""Tests for VAD helpers in `sonix.core.audio`.

These tests exercise the RMS fallback and the pyannote-based path
via a lightweight mock of `pyannote.audio.Inference`.
"""
import sys
import types

import numpy as np
import soundfile as sf

from sonix.core import audio


def _write_sine(path, sr=16000, duration=1.0, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, data, sr)
    return data, sr


def test_vad_rms_fallback(tmp_path):
    """When pyannote isn't available, vad_with_pyanote should fall back to RMS VAD."""
    p = tmp_path / "sine.wav"
    _write_sine(str(p))

    segs = audio.vad_with_pyanote(str(p))
    assert isinstance(segs, list)
    # expect at least one segment and labels from {'speech','silence'}
    assert len(segs) >= 1
    assert all(s.get("label") in {"speech", "silence"} for s in segs)


def test_vad_with_pyanote_mock(tmp_path, monkeypatch):
    """Mock pyannote.audio.Inference to exercise the pyannote path."""
    p = tmp_path / "sine2.wav"
    _write_sine(str(p))

    # Create fake pyannote.audio module with Inference that returns controlled output
    mod = types.ModuleType("pyannote")
    sub = types.ModuleType("pyannote.audio")

    class FakeInference:
        # pylint: disable=too-few-public-methods
        """Minimal fake Inference callable returning predetermined probs."""

        def __init__(self, model, *_args, **_kwargs):
            self.model = model

        def __call__(self, arg):
            # two windows: [0,0.5), [0.5,1.0)
            window = [
                type("W", (), {"start": 0.0, "end": 0.5}),
                type("W", (), {"start": 0.5, "end": 1.0}),
            ]
            probs = np.array([[0.9], [0.1]])
            return types.SimpleNamespace(data=probs, sliding_window=window)

    sub.Inference = FakeInference
    monkeypatch.setitem(sys.modules, "pyannote", mod)
    monkeypatch.setitem(sys.modules, "pyannote.audio", sub)

    segs = audio.vad_with_pyanote(str(p), threshold=0.5)
    # two windows -> at least two segments
    assert len(segs) >= 2
    # first window had prob 0.9 -> speech, second 0.1 -> silence
    assert segs[0]["label"] == "speech"
    assert segs[1]["label"] == "silence"


def test_pyanote_unexpected_output_triggers_fallback(tmp_path, monkeypatch):
    """When pyannote returns an unexpected structure, fallback is used."""
    p = tmp_path / "sine_fail.wav"
    _write_sine(str(p), duration=0.5)

    # Create a fake pyannote.audio.Inference that returns invalid structure
    mod = types.ModuleType("pyannote")
    sub = types.ModuleType("pyannote.audio")

    # pylint: disable=too-few-public-methods
    class BadInference:
        """Fake Inference returning invalid output to trigger fallback."""

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, arg):
            # Return an object missing expected attributes to force an error
            return types.SimpleNamespace(data=None, sliding_window=None)

    sub.Inference = BadInference
    monkeypatch.setitem(sys.modules, "pyannote", mod)
    monkeypatch.setitem(sys.modules, "pyannote.audio", sub)

    segs = audio.vad_with_pyanote(str(p))
    assert isinstance(segs, list)
    assert len(segs) >= 1
    assert all(s.get("label") in {"speech", "silence"} for s in segs)
