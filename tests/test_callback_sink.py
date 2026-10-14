import math
from itertools import pairwise

import pytest

from verbix.voice.callback_sink import CallbackAudioSink, apply_fade


def test_fade_ramps_to_silence_in_10ms():
    faded = apply_fade([1.0] * 240, sample_rate=24000)
    assert faded[-1] == 0.0 and all(b <= a for a, b in zip([1.0] + faded, faded))


def test_import_does_not_require_sounddevice(monkeypatch):
    import importlib
    import sys

    import verbix.voice.callback_sink as m

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    importlib.reload(m)
    assert "sounddevice" not in sys.modules


def test_callback_sink_fails_fast_when_device_open_fails(monkeypatch):
    import sys
    import types

    class ExplodingStream:
        def __init__(self, *a, **k):
            raise RuntimeError("No output device")

        def start(self):
            pass

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = ExplodingStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    with pytest.raises(RuntimeError, match="libportaudio2|portaudio"):
        CallbackAudioSink()


def test_stop_applies_fade_then_silence(monkeypatch):
    import sys
    import types

    import numpy as np

    calls = {}

    class FakeStream:
        def __init__(self, *a, **k):
            calls["callback"] = k.get("callback")

        def start(self):
            calls["started"] = True

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from verbix.voice.callback_sink import CallbackAudioSink

    sink = CallbackAudioSink(sample_rate=24000)
    assert calls.get("started")

    sink._enqueue(np.ones(4800, dtype=np.float32).tobytes())
    out = np.zeros((240, 1), dtype=np.float32)
    sink._callback(out, 240, None, None)
    assert float(out.max()) > 0.9
    assert sink.played_samples == 240

    sink.stop()
    out2 = np.zeros((480, 1), dtype=np.float32)
    sink._callback(out2, 480, None, None)
    assert math.isclose(float(out2[0, 0]), 1.0, rel_tol=1e-6)
    assert float(out2[239, 0]) == 0.0
    assert float(out2[240:].max()) == 0.0
    assert sink.played_samples == 480

    out3 = np.zeros((240, 1), dtype=np.float32)
    sink._callback(out3, 240, None, None)
    assert float(out3.max()) == 0.0


def _stubbed_sink(monkeypatch):
    import sys
    import types

    class FakeStream:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from verbix.voice.callback_sink import CallbackAudioSink

    return CallbackAudioSink(sample_rate=24000)


def test_short_tail_fades_fully_to_zero_without_jump(monkeypatch):
    import numpy as np

    sink = _stubbed_sink(monkeypatch)
    sink._enqueue(np.ones(100, dtype=np.float32).tobytes())
    sink.stop()
    out = np.zeros((480, 1), dtype=np.float32)
    sink._callback(out, 480, None, None)
    head = [float(v) for v in out[:100, 0]]
    assert math.isclose(head[0], 1.0, rel_tol=1e-6)
    assert head[-1] == 0.0
    assert all(b <= a for a, b in zip([1.0] + head, head))
    assert max(a - b for a, b in pairwise(head)) < 0.05
    assert float(out[100:].max()) == 0.0
    assert sink.played_samples == 100

    sink.stop()
    out2 = np.zeros((240, 1), dtype=np.float32)
    sink._callback(out2, 240, None, None)
    assert float(out2.max()) == 0.0
    assert sink.played_samples == 100


def test_callback_sink_sets_20ms_blocksize_and_fails_fast_with_portaudio_hint(monkeypatch):
    import sys
    import types

    recorded_kwargs = {}

    class FakeStream:
        def __init__(self, *a, **k):
            recorded_kwargs.update(k)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from verbix.voice.callback_sink import CallbackAudioSink

    sink = CallbackAudioSink(sample_rate=24000)
    assert recorded_kwargs.get("blocksize") == 480
    sink.close()

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match="libportaudio2|portaudio"):
        CallbackAudioSink(sample_rate=24000)

