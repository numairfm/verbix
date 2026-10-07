# Third-Party Notices

Dependency code licenses are declared in `uv.lock` / package metadata (all
permissive: MIT/BSD/Apache/PSF/ISC; no GPL/AGPL anywhere in the tree).
This file covers heavy model and media assets by name.

## TTS voice engine

- **PocketTTS.cpp** — single-file C++ runtime (ONNX Runtime). Pinned source
  commit recorded in `models/MANIFEST.json` at fetch time.
- **Kyutai PocketTTS weights** (CC-BY-4.0): int8 flow models + fp32 codec,
  sentencepiece tokenizer, BOS voice-conditioning file, via the community
  ONNX export line. Modification note: format conversion to ONNX only.
  Links: <https://github.com/kyutai-labs/pocket-tts>,
  <https://github.com/k2-fsa/sherpa-onnx> (reference implementation).
- **Alba voice reference** (`voices/alba.wav`): canonical Alba clip from
  `kyutai/tts-voices` (`alba-mackenna/casual.wav`), CC-BY-4.0. Voice-likeness
  clearance for commercial use is tracked separately (CC-BY-4.0 does not
  license personality rights).
- **ONNX Runtime** (MIT) — fetched by the helper's own CMake build.

## STT

- **faster-whisper tiny.en** (Systran, Apache-2.0), int8 CPU inference via
  CTranslate2 (MIT).
