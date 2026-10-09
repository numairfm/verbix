# Distribution Terms — Verbix Voice Runtime

This file is the user-facing distribution terms for the voice runtime that
Verbix ships. It exists because one heavy asset in that runtime is licensed
under a use-restricted license, and that license requires the restrictions to
travel with every copy.

## What binds

- Asset: **Supertonic 3** TTS voice weights (package `sherpa-onnx-supertonic-3-tts-int8-2026-05-11`),
  loaded in-process by the sherpa-onnx engine.
- License: **BigScience OpenRAIL-M** (Supertone, Inc.), read 2026-10-09
  at <https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE>.
  Commercial use is allowed; the use restrictions apply and must be passed on.
- License copy: [`LICENSES/bigscience-openrail-m.md`](LICENSES/bigscience-openrail-m.md).
- Asset provenance: [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
- Legal sign-off blocks release; anything unclear stays `UNRESOLVED` and
  blocks the cutover.

These terms bind anyone who distributes the Model (the Supertonic 3 weights),
including as part of Verbix, and must be included as an enforceable provision
in any agreement governing that use or distribution (license §4a). Notice to
subsequent users: the Model is subject to the use restrictions below; give
every recipient a copy of the license (license §4b).

## Use restrictions (license Attachment A, binding in full)

You agree not to use the Model or Derivatives of the Model:

1. **Law.** In any way that violates any applicable national, federal, state,
   local or international law or regulation.
2. **Minors.** For the purpose of exploiting, harming or attempting to exploit
   or harm minors in any way.
3. **Verifiable falsehood.** To generate or disseminate verifiably false
   information and/or content with the purpose of harming others.
4. **PII.** To generate or disseminate personal identifiable information that
   can be used to harm an individual.
5. **Machine-generated disclosure.** To generate or disseminate information
   and/or content (e.g. images, code, posts, articles), and place the
   information and/or content in any context (e.g. bot generating tweets)
   **without expressly and intelligibly disclaiming that the information
   and/or content is machine generated.**
6. **Harassment.** To defame, disparage or otherwise harass others.
7. **Impersonation.** To impersonate or attempt to impersonate (e.g. deepfakes)
   others without their consent.
8. **Automated legal decisions.** For fully automated decision making that
   adversely impacts an individual's legal rights or otherwise creates or
   modifies a binding, enforceable obligation.
9. **Social scoring / behavioral discrimination.** For any use intended to or
   which has the effect of discriminating against or harming individuals or
   groups based on online or offline social behavior or known or predicted
   personal or personality characteristics.
10. **Exploiting vulnerabilities.** To exploit any of the vulnerabilities of a
    specific group of persons based on their age, social, physical or mental
    characteristics, in order to materially distort the behavior of a person
    pertaining to that group in a manner that causes or is likely to cause that
    person or another person physical or psychological harm.
11. **Protected characteristics.** For any use intended to or which has the
    effect of discriminating against individuals or groups based on legally
    protected characteristics or categories.
12. **Medical.** To provide medical advice and medical results
    interpretation.
13. **Justice / law enforcement.** To generate or disseminate information for
    the purpose to be used for administration of justice, law enforcement,
    immigration or asylum processes, such as predicting an individual will
    commit fraud/crime commitment (e.g. by text profiling, drawing causal
    relationships between assertions made in documents, indiscriminate and
    arbitrarily-targeted use).

The list above is an exact summary of Attachment A to the BigScience
OpenRAIL-M license. In any disagreement between this summary and the
license copy in [`LICENSES/bigscience-openrail-m.md`](LICENSES/bigscience-openrail-m.md),
the license copy governs.

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
