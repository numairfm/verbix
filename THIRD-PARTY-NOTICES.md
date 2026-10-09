# Third-Party Notices

Dependency code licenses are declared in `uv.lock` / package metadata (all
permissive: MIT/BSD/Apache/PSF/ISC; no GPL/AGPL anywhere in the tree).
This file covers heavy model and media assets by name.

## Voice Engine (sherpa-onnx)

- **sherpa-onnx** (Apache-2.0) — next-gen Kaldi / ONNX Runtime speech framework
  by k2-fsa (<https://github.com/k2-fsa/sherpa-onnx>).

## TTS Voice Engine

- **Supertonic 3** (BigScience OpenRAIL-M) — on-device flow-matching multilingual
  TTS by Supertone Inc. (<https://huggingface.co/Supertone/supertonic-3>).
  - Pinned checkpoint: `csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11` (INT8 dynamic quantization).
  - Use-based restrictions of the OpenRAIL-M license apply to downstream use and distribution
    (see `LICENSES/bigscience-openrail-m.md` and `DISTRIBUTION-TERMS.md`).
  - Voice preset: `M1` (provisional pending voice-likeness clearance for commercial release;
    personality rights are tracked separately).

## STT Speech Recognition

- **Canary 180M Flash** (NVIDIA, CC-BY-4.0) — multilingual FastConformer-CTC/AED ASR
  by NVIDIA NeMo (<https://huggingface.co/nvidia/canary-180m-flash>).
  - Pinned checkpoint: `csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8` (INT8 dynamic quantization).
