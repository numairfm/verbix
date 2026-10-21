# Supertonic 3 preset voice risk acceptance

Status: ACCEPTED RESIDUAL RISK
Date: 2026-10-21
Accepted by: numairfm (product owner)
Scope: Supertonic 3 model package, including the fixed preset voice styles
`M1`–`M5` and `F1`–`F5`; Verbix v1 bundles the `M1` style as `voice.bin`.

## Decision

Verbix treats the preset voice styles as part of the Supertonic 3 Model under
the BigScience OpenRAIL-M license. The upstream model repository distributes
the weights and the `voice_styles/*.json` presets together, under one
copyright holder and one license. No separate voice license is published.
Verbix relies on that grant for the presets it ships.

This is a product risk acceptance, not a legal clearance. It records that
Verbix will ship the preset while accepting the unresolved speaker-provenance
risk below. It does not assert speaker consent, identity, or a separate
voice-likeness right.

## Pinned assets

- Upstream model: `Supertone/supertonic-3`, revision
  `3cadd1ee6394adea1bd021217a0e650ede09a323` (read 2026-10-21).
  The repository contains `LICENSE` and `voice_styles/M1.json` through
  `voice_styles/M5.json` and `voice_styles/F1.json` through
  `voice_styles/F5.json`.
- Bundled INT8 conversion:
  `csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11`, revision
  `cca5a0e6c96e1d2c720986bf7e75fcc81dee3ae4` (read 2026-10-21).
  The bundled `voice.bin` has SHA-256
  `67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8`
  and 517,168 bytes.
- License: BigScience OpenRAIL-M. The exact text is shipped at
  `LICENSES/bigscience-openrail-m.md` and
  `src/verbix/voice/tts/supertonic-int8/LICENSE-OpenRAIL-M`.
  The latter is the upstream `LICENSE` at the pinned revision, 15,007 bytes,
  SHA-256
  `0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f`.
- The bundled `src/verbix/voice/tts/supertonic-int8/LICENSE` is the MIT
  license for the INT8 conversion package. It is not the model license.

## Evidence reviewed

- Upstream model card and repository tree at the pinned revision, showing the
  weights, `voice_styles/*.json` presets, and a single OpenRAIL-M `LICENSE`.
- Upstream license text at the pinned revision, including Attachment A.
- The quantized checkpoint's file list and `voice.bin` checksum at its pinned
  revision.
- The upstream model card describes the presets as fixed preset voice styles.
  It does not publish speaker identity, consent, or recording provenance for
  any preset.
- No separate voice license, voice grant, or commercial voice terms were found
  for the open weights.

## Residual risk

- No speaker identity, consent record, or recording provenance is published
  for the preset styles. This is a known, accepted residual risk.
- A person or rights holder could later assert a voice-likeness or related
  claim. If that happens, Verbix will remove or replace the affected preset
  rather than continue shipping it.
- The upstream repository is archived and support is discontinued, so the
  evidence is frozen rather than maintained.
- The INT8 conversion is a community quantization of the upstream model; its
  files are pinned by revision and SHA-256.

## Product conflicts

- **Synthetic-audio disclosure.** Verbix speaks with a synthetic voice. The
  panel footer and `DISTRIBUTION-TERMS.md` disclose it. The product does not
  prepend a spoken disclaimer to every answer; anyone who exports or
  redistributes generated audio must carry the disclosure required by
  Attachment A(e).
- **Medical advice.** Verbix is a general assistant, so a user can ask health
  questions. Attachment A(l) forbids using the Model to provide medical advice
  or interpret medical results. `DISTRIBUTION-TERMS.md` passes that restriction
  to users. This slice does not add a model-level medical guardrail; a health
  feature needs a separate product and legal decision before it ships.
- **Impersonation.** Verbix ships fixed preset styles only. It has no voice
  cloning and accepts no user-supplied voice styles. The preset identifiers are
  labels, not people.
- **Preset naming.** The upstream demo labels and the preset identifiers are
  not used as real-person names in the product. Verbix shows only the preset
  ID.

## Distribution conditions

- Ship the OpenRAIL-M license text and reproduce Attachment A in
  `DISTRIBUTION-TERMS.md`.
- Keep the upstream and quantized revisions pinned; do not fetch `main`.
- Disclose that all speech is synthetic and machine generated, in the product
  and in the distribution terms.
- Present `M1`–`M5` and `F1`–`F5` only as preset identifiers. Do not present
  them as names, identities, or portrayals of real people.
- Do not use the Model for medical advice or medical-results interpretation,
  impersonation, or any other use prohibited by Attachment A.
- Do not add voice cloning or user-supplied voice styles without a new review.

## Phonon comparison

No swap to Phonon 31M-en was made. Its code and model licenses are permissive,
but no evidence showed that its voice provenance is verifiably better than
Supertonic's. A licence swap alone would not resolve the speaker-provenance
question this record accepts.
