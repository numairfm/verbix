# Implementation Plan: Pure-Python Audio Engine, Cross-Platform Ducking & Conversational Voice Loop

## 1. Goal & Requirements
- **Zero OS CLI binaries**: Replace `pw-record`, `arecord`, `pw-play`, `aplay`, `ffplay` with pure-Python `sounddevice` across all platforms (macOS, Windows, Linux).
- **Cross-Platform Audio Ducking**: When the mic hatch opens, background media (YouTube, Spotify, etc.) is ducked by ~15-20 dB. When the hatch closes, volume restores smoothly.
- **Continuous Conversational Loop**: Pressing the hotkey opens the mic hatch. Silero VAD detects speech completion. Agent responds via TTS. Loop continues listening automatically until the user presses the hotkey again to close the hatch.
- **Echo & Barge-In Safety**: During TTS playback, STT audio feeding is gated so the assistant does not trigger on its own voice, while maintaining instant barge-in detection with a 10ms click-free fade.

---

## 2. Architecture & Seams

```
[User / Hotkey]
      │
      ▼
[VoicePipeline (Duplex Mode)]
      │
      ├──> AudioDucker (src/verbix/voice/ducking.py)
      │         ├──> Linux: pulsectl (ctypes libpulse, pipewire-pulse / pulseaudio)
      │         ├──> macOS: CoreAudio AudioObjectSetPropertyData (ctypes)
      │         ├──> Windows: pycaw / IAudioEndpointVolume (ctypes comtypes)
      │         └──> NullDucker: Headless CI / test fallback
      │
      ├──> PTTCapture (src/verbix/voice/capture.py)
      │         └──> sounddevice.RawInputStream (16kHz mono)
      │                   └──> _condition_s16le (drift filter)
      │                             └──> sherpa_vad (Speech Boundary & Silence Detection)
      │                                       └──> sherpa_stt (Live STT Session)
      │
      └──< CallbackAudioSink (src/verbix/voice/callback_sink.py)
                └──> sounddevice.OutputStream (24kHz f32, 10ms barge-in fade)
```

---

## 3. Slice Breakdown & Proof (TDD)

### Slice 1: Dependencies & Pure-Python Audio Capture
- **File**: `pyproject.toml`
  - Move `sounddevice>=0.5.6` from optional to core dependencies.
- **File**: `src/verbix/voice/capture.py`
  - Replace subprocess spawning (`pw-record`/`arecord`) with `sounddevice.RawInputStream`.
  - Expose backend injection / mock interface for headless test environments (CI without soundcard).
- **Proof / Failing Test First**:
  - `tests/test_sounddevice_capture.py`: Verify capture starts/stops, emits conditioned PCM chunks, handles audio device errors gracefully, and runs hermetically without real hardware.

### Slice 2: Unified Playback Sink
- **File**: `src/verbix/voice/callback_sink.py` & `sinks.py`
  - Standardize `VoicePipeline` and `AudioPlayer` on `CallbackAudioSink`.
  - Deprecate CLI subprocess sink.
- **Proof / Failing Test First**:
  - `tests/test_callback_sink.py`: Verify `play_stream`, `stop` with 10ms fade, and sample-accurate playback draining.

### Slice 3: In-Process Audio Ducking (Zero Subprocesses)
- **File**: `src/verbix/voice/ducking.py` (New module)
  - Create `AudioDucker` protocol and platform implementations:
    - `LinuxDucker`: Direct `ctypes` connection to `libpulse.so.0` / `pipewire-pulse` to read/set sink input volumes without spawning CLI processes.
    - `DarwinDucker`: Direct `ctypes` call to Apple's `/System/Library/Frameworks/CoreAudio.framework/CoreAudio` using `AudioObjectSetPropertyData` on master element volume.
    - `WindowsDucker`: Direct `ctypes` COM call to `IAudioEndpointVolume` via `ole32.dll`.
    - `NullDucker`: Headless fallback for CI and unsupported platforms.
- **Proof / Failing Test First**:
  - `tests/test_audio_ducking.py`: Verify `duck()` stores state, applies -15 dB / 30% level, and `restore()` returns volume to original level idempotently across mock/stub backends.

### Slice 4: Continuous Conversational Voice Loop
- **File**: `src/verbix/voice/pipeline.py`
  - Refactor `toggle()` and turn state machine:
    - `OPEN_HATCH`: `duck()` called. Capture begins.
    - `UTTERANCE_DETECTED`: Silero VAD flags `is_speech_ended()`. STT finalizes.
    - `AGENT_SPEAKING`: TTS streams out. Mic STT feed gated to avoid self-echo. Barge-in remains active.
    - `TURN_COMPLETE`: Capture unpauses and continues listening for user's next utterance.
    - `CLOSE_HATCH`: User presses hotkey. Capture stops. `restore()` unducks audio. Pipeline goes to IDLE.
- **Proof / Failing Test First**:
  - `tests/test_conversational_voice_loop.py`: Simulate multi-turn conversation without re-triggering hotkey, verify turn boundaries, verify duck/unduck lifecycle, and verify barge-in aborts agent speech immediately.

---

## 4. Verification Commands
- `uv run pytest tests/test_sounddevice_capture.py`
- `uv run pytest tests/test_audio_ducking.py`
- `uv run pytest tests/test_conversational_voice_loop.py`
- `uv run pytest` (Full 269+ test suite green)
- `uv run ruff check` (Clean linting)
