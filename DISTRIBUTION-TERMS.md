# Distribution Terms — Verbix Voice Runtime

This file is the user-facing distribution terms for the voice runtime that
Verbix ships. It exists because one heavy asset in that runtime is licensed
under a use-restricted license, and that license requires the restrictions to
travel with every copy.

## What binds

- Asset: **Supertonic 3** TTS model and its fixed preset voice styles
  (`M1`–`M5`, `F1`–`F5`), bundled with Verbix under
  `verbix/voice/tts/supertonic-int8/` and loaded in-process by the sherpa-onnx
  engine. Verbix v1 ships the `M1` style as `voice.bin`.
- Upstream model: `Supertone/supertonic-3`, revision
  `3cadd1ee6394adea1bd021217a0e650ede09a323`.
- Bundled INT8 conversion:
  `csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11`, revision
  `cca5a0e6c96e1d2c720986bf7e75fcc81dee3ae4`.
- License: **BigScience OpenRAIL-M** (Supertone, Inc.), read 2026-10-21 at
  <https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/LICENSE>.
  Commercial use is allowed; the use restrictions apply and must be passed on.
- License copies: [`LICENSES/bigscience-openrail-m.md`](LICENSES/bigscience-openrail-m.md),
  `src/verbix/voice/tts/supertonic-int8/LICENSE-OpenRAIL-M`, and the copy
  shipped in the AppImage under `licenses/`.
- Asset provenance: [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
- Preset risk record: maintained with the Verbix source at
  `docs/legal/2026-10-21-supertonic-3-preset-risk-acceptance.md`; the accepted
  risk is summarized below for recipients.
- Release gate: the dated record must exist in the source, and this file and the
  license text must ship. No unresolved asset or missing license text may ship.

These terms bind anyone who distributes the Model (the Supertonic 3 weights and
preset styles), including as part of Verbix, and every user of Verbix. They must
be included as an enforceable provision in any agreement governing that use or
distribution (license §4a). Notice to subsequent users: the Model is subject to
the use restrictions below; give every recipient a copy of the license (license
§4b).

## Voice presets and synthetic audio

The preset identifiers `M1`–`M5` and `F1`–`F5` are product labels for fixed
voice styles. They are not names, identities, or portrayals of real people.
Verbix does not present them as real people.

Supertone publishes no separate voice license for these preset styles. Verbix
treats them as part of the Model under the same OpenRAIL-M grant. No speaker
identity, consent record, or recording provenance is published. That is a
known, accepted residual risk, not a legal clearance; the dated record above
records the decision and its conditions. If a verified voice-likeness claim
arises, Verbix will remove or replace the affected preset.

All speech produced by Verbix is synthetic and machine generated. You must
expressly and intelligibly disclose that audio is machine generated whenever
you distribute it, as required by Attachment A(e).

## Use restrictions (license Attachment A, binding in full)

You agree not to use the Model or Derivatives of the Model:
(a) In any way that violates any applicable national, federal, state, local or international law
or regulation;
(b) For the purpose of exploiting, harming or attempting to exploit or harm minors in any
way;
(c) To generate or disseminate verifiably false information and/or content with the purpose of
harming others;
(d) To generate or disseminate personal identifiable information that can be used to harm an
individual;
(e) To generate or disseminate information and/or content (e.g. images, code, posts, articles),
and place the information and/or content in any context (e.g. bot generating tweets)
without expressly and intelligibly disclaiming that the information and/or content is
machine generated;
(f) To defame, disparage or otherwise harass others;
(g) To impersonate or attempt to impersonate (e.g. deepfakes) others without their consent;
(h) For fully automated decision making that adversely impacts an individual’s legal rights or
otherwise creates or modifies a binding, enforceable obligation;
(i) For any use intended to or which has the effect of discriminating against or harming
individuals or groups based on online or offline social behavior or known or predicted
personal or personality characteristics;
(j) To exploit any of the vulnerabilities of a specific group of persons based on their age,
social, physical or mental characteristics, in order to materially distort the behavior of a
person pertaining to that group in a manner that causes or is likely to cause that person or
another person physical or psychological harm;
(k) For any use intended to or which has the effect of discriminating against individuals or
groups based on legally protected characteristics or categories;
(l) To provide medical advice and medical results interpretation;
(m) To generate or disseminate information for the purpose to be used for administration of
justice, law enforcement, immigration or asylum processes, such as predicting an
individual will commit fraud/crime commitment (e.g. by text profiling, drawing causal
relationships between assertions made in documents, indiscriminate and
arbitrarily-targeted use).

This reproduces Attachment A of the license. In any disagreement, the license
copy in [`LICENSES/bigscience-openrail-m.md`](LICENSES/bigscience-openrail-m.md)
governs.

## Output

Except as set forth in the license, the licensor claims no rights in the
Output generated using the Model. **You are accountable for the Output you
generate and its subsequent uses** (license §6). No use of the output can
contravene any provision of the license.

## Updates and runtime restrictions

Licensor reserves the right to restrict usage of the Model in violation of the
license, update the Model through electronic means, or modify the Output based
on updates. **You shall undertake reasonable efforts to use the latest version
of the Model** (license §7).
