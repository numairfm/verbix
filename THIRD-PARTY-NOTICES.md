# Third-Party Notices

Dependency code licenses are declared in `uv.lock` / package metadata (all
permissive: MIT/BSD/Apache/PSF/ISC; no GPL/AGPL anywhere in the tree).
This file covers heavy model and media assets by name.

## TTS voice engine

- **PocketTTS.cpp** (MIT) — single-file C++ runtime (ONNX Runtime), built
  from <https://github.com/numairfm/PocketTTS.cpp>. Pinned source commit
  recorded in `tts-runtime/models/MANIFEST.json`.
- **PocketTTS weights** (CC-BY-4.0): int8 flow models + int8 decoder, fp32
  encoder/conditioner, sentencepiece tokenizer. Forged from
  `Verylicious/pocket-tts-ungated` via
  <https://github.com/numairfm/pocket-tts-onnx-export> (pinned commit in
  the manifest); served from the project's bundle mirror with byte-size
  pins in `voice/tts_assets.py`. Modification note: format conversion to
  ONNX (+ int8 dynamic quantization of MatMuls) only.
- **Alba voice reference** (`voices/alba.wav`): canonical Alba clip from
  `kyutai/tts-voices` (`alba-mackenna/casual.wav`), CC-BY-4.0. Voice-likeness
  clearance for commercial use is tracked separately (CC-BY-4.0 does not
  license personality rights).
- **ONNX Runtime** (MIT) — fetched by the helper's own CMake build.

## STT

- **faster-whisper tiny.en** (Systran, Apache-2.0), int8 CPU inference via
  CTranslate2 (MIT).
