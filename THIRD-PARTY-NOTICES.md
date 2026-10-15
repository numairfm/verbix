# Third-Party Notices

Dependency code licenses are declared in `uv.lock` / package metadata (all
permissive: MIT/BSD/Apache/PSF/ISC; no GPL/AGPL anywhere in the tree).
This file covers heavy model and media assets by name.

## Voice Engine (sherpa-onnx)

- **sherpa-onnx** (Apache-2.0) — next-gen Kaldi / ONNX Runtime speech framework
  by k2-fsa (<https://github.com/k2-fsa/sherpa-onnx>).

## TTS Voice Engine (Bundled Application Resource)

- **Supertonic 3** (BigScience OpenRAIL-M) — on-device flow-matching multilingual
  TTS by Supertone Inc. (<https://huggingface.co/Supertone/supertonic-3>).
  - Pinned checkpoint: `csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11` (INT8 dynamic quantization),
    bundled as a fixed application resource under `verbix/voice/tts/supertonic-int8/`.
  - Use-based restrictions of the OpenRAIL-M license apply to downstream use and distribution
    (see `LICENSES/bigscience-openrail-m.md`, `src/verbix/voice/tts/supertonic-int8/LICENSE`, and `DISTRIBUTION-TERMS.md`).
  - Voice preset: `M1` (provisional pending voice-likeness clearance for commercial release;
    personality rights are tracked separately).

## STT Speech Recognition (User-Managed Models)

- **transcribe.cpp** (MIT) — C/C++ speech-to-text library built on ggml,
  with Python bindings by handy-computer (<https://github.com/handy-computer/transcribe.cpp>).
  - Pinned packages: `transcribe-cpp==0.4.0`, `transcribe-cpp-native==0.4.0`
    (native pulled transitively by `transcribe-cpp>=0.4.0` in `pyproject.toml`).
  - License source: PyPI `license_expression` = MIT on both distributions;
    upstream `LICENSE` (MIT, The transcribe.cpp authors).
- **Nemotron 3.5 ASR streaming 0.6B GGUF** (OpenMDW-1.1) — multilingual
  streaming FastConformer-RNN-T ASR by NVIDIA, GGUF conversion by
  handy-computer (<https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf>).
  - Pinned checkpoint: `nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf` (496 MB,
    fetched by `scripts/fetch_voice_models.py` into `~/.verbix/voice/stt/models/nemotron-gguf/`).
  - License source: model-card tag `License: openmdw-1.1`, inherited from the
    base model `nvidia/nemotron-3.5-asr-streaming-0.6b` (governing terms
    OpenMDW-1.1, <https://openmdw.ai/license/1-1/>); ready for commercial use.

## Gitignore Pattern Matching (pathspec)

- **pathspec** (MPL-2.0) — pure-Python gitignore pattern matching library by Caleb P. Burns (<https://github.com/cpburnz/python-path-specification>).
  - Pinned package: `pathspec>=0.12.1` (pure-Python `py3-none-any` wheel; unmodified distribution permitted under MPL-2.0).

## Desktop Control Libraries (psutil, pyperclip, dbus-fast, pulsectl, pycaw)

- **psutil** (BSD-3-Clause) — cross-platform process and system utilities by Giampaolo Rodola (<https://github.com/giampaolo/psutil>).
- **pyperclip** (BSD-3-Clause) — cross-platform clipboard module by Al Sweigart (<https://github.com/asweigart/pyperclip>).
- **dbus-fast** (MIT) — pure-Python async DBus library used for Linux MPRIS2 media control (<https://github.com/Bluetooth-Devices/dbus-fast>).
- **pulsectl** (MIT) — ctypes `libpulse.so.0` client used for Linux PipeWire/PulseAudio volume control (<https://github.com/mk-fg/python-pulse-control>).
- **pycaw** (MIT) — Python Core Audio Windows Library used for Windows volume control (<https://github.com/AndreMiras/pycaw>).


