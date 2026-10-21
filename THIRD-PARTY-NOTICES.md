# Third-Party Notices

Verbix is a closed-source commercial product. This file records every code
dependency in the Cargo and Python trees, every bundled model and voice asset,
and the native libraries that the Linux AppImage bundles for the desktop shell.
`DISTRIBUTION-TERMS.md` carries the use restrictions that must travel with the
Supertonic TTS weights.

The generated dependency inventory at the end of this file is the machine
checked record for direct and transitive dependencies. The curated sections
below record the obligations that need a human-readable explanation.

## Desktop Shell (Tauri 2)

- **Tauri 2** (2.12.4, MIT OR Apache-2.0) — Rust desktop shell, tray, window,
  and webview host by the Tauri Working Group
  (<https://github.com/tauri-apps/tauri>). The AppImage bundles the Tauri
  runtime and the Tauri CLI's Linux deployment output.
- **tauri-build** (2.7.3, MIT OR Apache-2.0) — build-time code generator for
  Tauri (<https://github.com/tauri-apps/tauri>). Build tool only; not shipped.
- **tauri-plugin-single-instance** (2.5.4, MIT OR Apache-2.0) — v1
  single-instance and argument-forwarding plugin
  (<https://github.com/tauri-apps/plugins-workspace>).
- **tauri-plugin-global-shortcut** (2.4.1, MIT OR Apache-2.0) — v1 global
  shortcut plugin (<https://github.com/tauri-apps/plugins-workspace>).
- **serde** (1.0.229, MIT OR Apache-2.0) — serialization framework
  (<https://github.com/serde-rs/serde>).
- **serde_json** (1.0.154, MIT OR Apache-2.0) — JSON support for the shell and
  IPC parser (<https://github.com/serde-rs/json>).
- **tokio** (1.54.0, MIT) — asynchronous runtime used by Tauri
  (<https://github.com/tokio-rs/tokio>).
- **libc** (0.2.191, MIT OR Apache-2.0) — libc bindings used for the Linux
  parent-death signal (<https://github.com/rust-lang/libc>).

## Bundled CPython 3.13

- **CPython 3.13.16** (PSF-2.0) — the relocatable interpreter installed by
  `uv python install`, bundled under `voice/python` and used to run
  `verbix.cli` (<https://github.com/python/cpython>, build provenance
  <https://github.com/astral-sh/python-build-standalone>). The licence text is
  `LICENSES/psf-2.0.md`; the Python licence also carries the CNRI and BeOpen
  notices in `LICENSES/cnri-python.md`.
- The interpreter build statically includes its own runtime libraries. The
  ones with notice obligations are OpenSSL 3 (Apache-2.0), libffi (MIT), zlib
  (Zlib), bzip2 (bzip2-1.0.6), XZ/liblzma (0BSD), SQLite (public domain, no
  notice required), and ncurses (X11). Their texts are in `LICENSES/`; the
  build does not require a system Python, `uv`, or development headers at
  runtime.
- The AppImage trim step removes development headers, `pip`, `ensurepip`,
  `idlelib`, `tkinter`, Tcl/Tk, and duplicate interpreter binaries. CPython
  itself and the standard library stay bundled.

## Bundled System Libraries (AppImage)

The Linux AppImage bundles the following native libraries so it runs on a host
without system WebKitGTK, GTK, or AppIndicator. They are **dynamically linked**
from the shell and the wheel extensions; they are not statically linked into
Verbix code. Each library is shipped unmodified, keeps its own upstream
licence, and can be replaced by a user who extracts the AppImage.

### WebKitGTK 4.1

- Package: Fedora 44 `webkit2gtk4.1` 2.54.1-1.fc44, libraries
  `libwebkit2gtk-4.1.so.0` and `libjavascriptcoregtk-4.1.so.0`
  (<https://www.webkitgtk.org/>, <https://src.fedoraproject.org/rpms/webkit2gtk4.1>).
- Licence: **LGPL-2.1-or-later** for the WebKitGTK port, plus the third-party
  licences reproduced in `LICENSES/system/webkit2gtk-4.1/` (Apple BSD, ICU,
  ANGLE, WTF, Web Inspector, and others). The Fedora source package is
  `webkitgtk-2.54.1-1.fc44.src.rpm`.
- Obligation: dynamic linking, licence copy, and source availability. The full
  licence set ships in `LICENSES/system/webkit2gtk-4.1/`; the source offer
  below names the exact source.

### GTK 3

- Package: Fedora 44 `gtk3` 3.24.52-2.fc44, libraries including
  `libgtk-3.so.0`, `libgdk-3.so.0`, and the GTK modules
  (<https://gtk.org/>, <https://src.fedoraproject.org/rpms/gtk3>).
- Licence: **LGPL-2.0-or-later**. Text: `LICENSES/system/gtk3/COPYING` and
  `LICENSES/lgpl-2.0-or-later.md`. Source package: `gtk3-3.24.52-2.fc44.src.rpm`.

### AppIndicator

- Package: Fedora 44 `libappindicator-gtk3` 12.10.1-10.fc44, library
  `libappindicator3.so.1` and its dbusmenu dependencies
  (<https://launchpad.net/libappindicator>,
  <https://src.fedoraproject.org/rpms/libappindicator>).
- Licence: Fedora declares `LGPL-2.1-only AND LGPL-3.0-only` for this package.
  Texts: `LICENSES/system/libappindicator-gtk3/COPYING` (LGPL-3.0),
  `COPYING.LGPL.2` (LGPL-2.0), and `COPYING.LGPL.2.1` (LGPL-2.1). Source
  package: `libappindicator-12.10.1-10.fc44.src.rpm`.

### Host runtime libraries (not bundled)

The AppImage leaves the base desktop stack to the host: X11/Wayland client
libraries, Mesa EGL/GL, fontconfig, FreeType, FriBidi, and HarfBuzz. Those
libraries keep their upstream licences and are redistributed by the host
distribution, not by Verbix. They are dynamically linked, never statically
copied into Verbix code.

### Other bundled native libraries

The wheel and Linux deployment step also bundle these. They are dynamically
linked and retain their upstream licences; their licence texts are in
`LICENSES/` where the licence requires one.

- **PortAudio** (MIT) — audio input/output backend loaded through
  `sounddevice` (<https://www.portaudio.com/>). The AppImage **bundles**
  `libportaudio.so.2` under `usr/lib/`, plus `libportaudio.so` for library
  lookup. The release build uses Fedora 44 `portaudio-19.7.0-3`; the CI build
  on Ubuntu bundles that distribution's `libportaudio2` package. Both are the
  same MIT-licensed PortAudio, unmodified and dynamically loaded. The shell
  exports `VERBIX_PORTAUDIO_LIB=<AppDir>/usr/lib/libportaudio.so.2` and the
  voice service resolves PortAudio through that path before `sounddevice`
  imports, so no system PortAudio package is required. Text:
  `LICENSES/system/portaudio/LICENSE.txt`. Source:
  <https://src.fedoraproject.org/rpms/portaudio> and
  <https://packages.ubuntu.com/source/libportaudio2>.
- **PipeWire libs, JACK, and JACK2** — PortAudio's runtime closure is copied
  from the build host. The Fedora release build bundles `libasound.so.2`,
  `libpipewire-0.3.so.0`, and PipeWire's `libjack.so.0`; the Ubuntu CI build
  bundles `libasound.so.2`, JACK2's `libjack.so.0`, and JACK2's
  `libdb-5.3.so`. All are unmodified and dynamically loaded. The pipewire
  libraries are MIT AND GPL-2.0-or-later AND BSD-2-Clause AND
  LGPL-2.0-or-later (`pipewire-libs-1.6.9-1.fc44`); JACK2 is
  LGPL-2.1-or-later; Berkeley DB 5.3 is under the Sleepycat licence. Texts:
  `LICENSES/mit.md`, `LICENSES/bsd-2-clause.md`,
  `LICENSES/lgpl-2.0-or-later.md`, `LICENSES/lgpl-2.1-or-later.md`,
  `LICENSES/system/pipewire/COPYING`,
  `LICENSES/system/pipewire/LICENSE`, `LICENSES/system/libdb/COPYING`.
  Sources: <https://src.fedoraproject.org/rpms/pipewire> and
  <https://packages.ubuntu.com/source/libdb>. Berkeley DB is present only in
  the CI Ubuntu artifact, not in the Fedora release artifact.

- **ONNX Runtime** (MIT) — `libonnxruntime.so` from
  `sherpa-onnx` (<https://github.com/microsoft/onnxruntime>).
- **sherpa-onnx** (Apache-2.0) — `libsherpa-onnx-c-api.so`
  (<https://github.com/k2-fsa/sherpa-onnx>).
- **OpenBLAS** (BSD-3-Clause) — `libscipy_openblas64_-*.so` from NumPy
  (<https://github.com/OpenMathLib/OpenBLAS>).
- **GCC runtime** — `libgfortran` and `libquadmath` are covered by
  GPL-3.0-or-later WITH GCC-exception-3.1 and LGPL-2.1-or-later; texts are
  `LICENSES/gcc-exception-3.1.md` and `LICENSES/lgpl-2.1-or-later.md`.
- **ALSA lib** (LGPL-2.1-or-later) — `libasound-*.so` from the sherpa-onnx
  wheel (<https://www.alsa-project.org/>).
- **ggml / transcribe.cpp** (MIT) — `libggml*.so` from
  `transcribe-cpp-native` (<https://github.com/handy-computer/transcribe.cpp>).
- **Pillow bundled codecs** — libjpeg-turbo (IJG/BSD-3-Clause), libpng
  (libpng-2.0), libtiff (libtiff), libwebp (BSD-3-Clause), freetype
  (FTL/GPL-2.0 dual), harfbuzz (MIT), and the other codecs listed in the wheel
  `pillow.libs` directory
  (<https://github.com/python-pillow/Pillow>). Their notice texts are in the
  wheel metadata and the Pillow licence; `LICENSES/mit.md` and
  `LICENSES/bsd-3-clause.md` cover the permissive ones.

### Source offer

The exact source for each bundled system library is the matching Fedora 44
source RPM named above, available from
<https://src.fedoraproject.org/rpms/webkit2gtk4.1>,
<https://src.fedoraproject.org/rpms/gtk3>,
<https://src.fedoraproject.org/rpms/libappindicator>,
<https://src.fedoraproject.org/rpms/portaudio>, and
<https://src.fedoraproject.org/rpms/pipewire>. The corresponding
upstream releases are WebKitGTK 2.54.1, GTK 3.24.52, and libappindicator
12.10.1. For the CPython runtime, the source is CPython 3.13.16 plus the
python-build-standalone build recipe. Verbix ships these libraries unmodified.
A recipient who needs source beyond these public locations can request it from
the distributor; the written offer is valid for as long as the AppImage is
distributed.

## Voice Engine (sherpa-onnx)

- **sherpa-onnx** (Apache-2.0) — next-gen Kaldi / ONNX Runtime speech framework
  by k2-fsa (<https://github.com/k2-fsa/sherpa-onnx>).

## TTS Voice Engine (Bundled Application Resource)

- **Supertonic 3** (BigScience OpenRAIL-M) — on-device flow-matching multilingual
  TTS by Supertone Inc. (<https://huggingface.co/Supertone/supertonic-3>).
  - Upstream model: `Supertone/supertonic-3` at revision
    `3cadd1ee6394adea1bd021217a0e650ede09a323`, including the fixed preset
    voice styles `M1`–`M5` and `F1`–`F5`.
  - Pinned checkpoint: `csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11`
    at revision `cca5a0e6c96e1d2c720986bf7e75fcc81dee3ae4` (INT8 dynamic
    quantization), bundled as a fixed application resource under
    `verbix/voice/tts/supertonic-int8/`. Verbix v1 ships the `M1` style as
    `voice.bin`.
  - License text: `LICENSES/bigscience-openrail-m.md` and
    `src/verbix/voice/tts/supertonic-int8/LICENSE-OpenRAIL-M` (the upstream
    license at the pinned revision). The bundled
    `src/verbix/voice/tts/supertonic-int8/LICENSE` is the MIT license for the
    INT8 conversion package, not the model license.
  - Use-based restrictions of the OpenRAIL-M license apply to downstream use
    and distribution; Attachment A is reproduced in `DISTRIBUTION-TERMS.md`.
  - Preset voice risk acceptance: **ACCEPTED** — internal source record `docs/legal/2026-10-21-supertonic-3-preset-risk-acceptance.md`. The presets
    are treated as part of the Model under the same OpenRAIL-M grant; no
    separate voice license or speaker-consent record is published. This is a
    known, accepted residual risk, not a legal clearance.
  - Preset identifiers are labels: `M1`–`M5` and `F1`–`F5` are not names,
    identities, or portrayals of real people, and Verbix does not present them
    as real people.
  - Synthetic audio disclosure: Verbix speaks with a synthetic, machine-generated
    voice. The panel and `DISTRIBUTION-TERMS.md` disclose this; recipients of
    generated audio must disclose it too.

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
    OpenMDW-1.1, <https://openmdw.ai/license/1-1/>; copy in
    `LICENSES/openmdw-1.1.md`); ready for commercial use.

## Gitignore Pattern Matching (pathspec)

- **pathspec** (MPL-2.0) — pure-Python gitignore pattern matching library by Caleb P. Burns (<https://github.com/cpburnz/python-path-specification>).
  - Pinned package: `pathspec>=0.12.1` (pure-Python `py3-none-any` wheel; unmodified distribution permitted under MPL-2.0).

## Desktop Control Libraries (psutil, pyperclip, dbus-fast, pulsectl, pycaw)

- **psutil** (BSD-3-Clause) — cross-platform process and system utilities by Giampaolo Rodola (<https://github.com/giampaolo/psutil>).
- **pyperclip** (BSD-3-Clause) — cross-platform clipboard module by Al Sweigart (<https://github.com/asweigart/pyperclip>).
- **dbus-fast** (MIT) — pure-Python async DBus library used for Linux MPRIS2 media control (<https://github.com/Bluetooth-Devices/dbus-fast>).
- **pulsectl** (MIT) — ctypes `libpulse.so.0` client used for Linux PipeWire/PulseAudio volume control (<https://github.com/mk-fg/python-pulse-control>).
- **pycaw** (MIT) — Python Core Audio Windows Library used for Windows volume control (<https://github.com/AndreMiras/pycaw>).

## Copyleft Python Dependencies

- **num2words** (0.5.14, LGPL-2.1-or-later) — number-to-words conversion used
  by the voice pipeline (<https://github.com/savoirfairelinux/num2words>). It
  ships as **unmodified source inside the bundled `site-packages`**; Verbix
  imports it dynamically as a separate Python module and does not link it into
  any binary. The LGPL-2.1-or-later obligation is met by shipping that source,
  this notice, and the license text at `LICENSES/lgpl-2.1-or-later.md`. A
  recipient can replace the module in `site-packages` with another version.

## Generated Dependency Inventory

The block below is generated by `scripts/check_licenses.py --write` from
`cargo metadata --locked` and the installed Python `dist-info` metadata. Do not
edit it by hand. `--check` fails if it drifts from the scanned tree.

<!-- BEGIN GENERATED DEPENDENCY INVENTORY -->
- `cargo:adler2@2.0.1` — 0BSD OR MIT OR Apache-2.0
- `cargo:aho-corasick@1.1.5` — Unlicense OR MIT
- `cargo:alloc-no-stdlib@3.0.0` — BSD-3-Clause
- `cargo:alloc-stdlib@0.3.0` — BSD-3-Clause
- `cargo:android_system_properties@0.1.6` — MIT OR Apache-2.0
- `cargo:anyhow@1.0.104` — MIT OR Apache-2.0
- `cargo:async-broadcast@0.7.2` — MIT OR Apache-2.0
- `cargo:async-channel@2.5.0` — Apache-2.0 OR MIT
- `cargo:async-executor@1.14.0` — Apache-2.0 OR MIT
- `cargo:async-io@2.6.0` — Apache-2.0 OR MIT
- `cargo:async-lock@3.4.2` — Apache-2.0 OR MIT
- `cargo:async-process@2.5.0` — Apache-2.0 OR MIT
- `cargo:async-recursion@1.2.0` — MIT OR Apache-2.0
- `cargo:async-signal@0.2.14` — Apache-2.0 OR MIT
- `cargo:async-task@4.7.1` — Apache-2.0 OR MIT
- `cargo:async-trait@0.1.92` — MIT OR Apache-2.0
- `cargo:atk@0.18.2` — MIT
- `cargo:atk-sys@0.18.2` — MIT
- `cargo:atomic-waker@1.1.2` — Apache-2.0 OR MIT
- `cargo:autocfg@1.5.1` — Apache-2.0 OR MIT
- `cargo:base64@0.21.7` — MIT OR Apache-2.0
- `cargo:base64@0.22.1` — MIT OR Apache-2.0
- `cargo:base64@0.23.1` — MIT OR Apache-2.0
- `cargo:bit-set@0.8.0` — Apache-2.0 OR MIT
- `cargo:bit-vec@0.8.0` — Apache-2.0 OR MIT
- `cargo:bitflags@1.3.2` — MIT OR Apache-2.0
- `cargo:bitflags@2.13.2` — MIT OR Apache-2.0
- `cargo:block-buffer@0.10.4` — MIT OR Apache-2.0
- `cargo:block2@0.6.2` — MIT
- `cargo:blocking@1.7.0` — Apache-2.0 OR MIT
- `cargo:brotli@9.0.0` — BSD-3-Clause AND MIT
- `cargo:brotli-decompressor@6.0.1` — BSD-3-Clause OR MIT
- `cargo:bs58@0.5.1` — MIT OR Apache-2.0
- `cargo:bumpalo@3.20.3` — MIT OR Apache-2.0
- `cargo:bytemuck@1.25.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:byteorder@1.5.0` — Unlicense OR MIT
- `cargo:byteorder-lite@0.1.0` — Unlicense OR MIT
- `cargo:bytes@1.12.1` — MIT
- `cargo:cairo-rs@0.18.5` — MIT
- `cargo:cairo-sys-rs@0.18.2` — MIT
- `cargo:camino@1.2.6` — MIT OR Apache-2.0
- `cargo:cargo-platform@0.1.9` — MIT OR Apache-2.0
- `cargo:cargo_metadata@0.19.2` — MIT
- `cargo:cargo_toml@1.0.1` — Apache-2.0 OR MIT
- `cargo:cc@1.9.0` — MIT OR Apache-2.0
- `cargo:cesu8@1.1.0` — Apache-2.0 OR MIT
- `cargo:cfb@0.14.0` — MIT
- `cargo:cfg-expr@0.15.8` — MIT OR Apache-2.0
- `cargo:cfg-if@1.0.5` — MIT OR Apache-2.0
- `cargo:chrono@0.4.45` — MIT OR Apache-2.0
- `cargo:combine@4.6.9` — MIT
- `cargo:concurrent-queue@2.5.0` — Apache-2.0 OR MIT
- `cargo:cookie@0.18.2` — MIT OR Apache-2.0
- `cargo:core-foundation@0.10.1` — MIT OR Apache-2.0
- `cargo:core-foundation-sys@0.8.7` — MIT OR Apache-2.0
- `cargo:core-graphics@0.25.0` — MIT OR Apache-2.0
- `cargo:core-graphics-types@0.2.0` — MIT OR Apache-2.0
- `cargo:cpufeatures@0.2.17` — MIT OR Apache-2.0
- `cargo:crc32fast@1.5.2` — MIT OR Apache-2.0
- `cargo:crossbeam-channel@0.5.17` — MIT OR Apache-2.0
- `cargo:crossbeam-utils@0.8.23` — MIT OR Apache-2.0
- `cargo:crypto-common@0.1.7` — MIT OR Apache-2.0
- `cargo:cssparser@0.37.0` — MPL-2.0
- `cargo:cssparser-macros@0.7.1` — MPL-2.0
- `cargo:ctor@1.0.13` — Apache-2.0 OR MIT
- `cargo:darling@0.24.1` — MIT
- `cargo:darling_core@0.24.1` — MIT
- `cargo:darling_macro@0.24.1` — MIT
- `cargo:dbus@0.9.12` — Apache-2.0 OR MIT
- `cargo:defmt@1.2.0` — MIT OR Apache-2.0
- `cargo:defmt-macros@1.2.0` — MIT OR Apache-2.0
- `cargo:defmt-parser@1.1.0` — MIT OR Apache-2.0
- `cargo:deranged@0.5.9` — MIT OR Apache-2.0
- `cargo:derive_more@2.1.1` — MIT
- `cargo:derive_more-impl@2.1.1` — MIT
- `cargo:digest@0.10.7` — MIT OR Apache-2.0
- `cargo:dirs@7.0.0` — MIT OR Apache-2.0
- `cargo:dirs-sys@0.5.0` — MIT OR Apache-2.0
- `cargo:dispatch2@0.3.1` — Zlib OR Apache-2.0 OR MIT
- `cargo:displaydoc@0.2.7` — MIT OR Apache-2.0
- `cargo:dlopen2@0.8.2` — MIT
- `cargo:dlopen2_derive@0.4.3` — MIT
- `cargo:dom_query@0.28.0` — MIT
- `cargo:dpi@0.1.2` — Apache-2.0 AND MIT
- `cargo:dtoa@1.0.11` — MIT OR Apache-2.0
- `cargo:dtoa-short@0.3.5` — MPL-2.0
- `cargo:dunce@1.0.5` — CC0-1.0 OR MIT-0 OR Apache-2.0
- `cargo:dyn-clone@1.0.20` — MIT OR Apache-2.0
- `cargo:embed-resource@3.0.12` — MIT
- `cargo:embed_plist@1.2.2` — MIT OR Apache-2.0
- `cargo:endi@1.1.1` — MIT
- `cargo:enumflags2@0.7.12` — MIT OR Apache-2.0
- `cargo:enumflags2_derive@0.7.12` — MIT OR Apache-2.0
- `cargo:equivalent@1.0.3` — Apache-2.0 OR MIT
- `cargo:erased-serde@0.4.10` — MIT OR Apache-2.0
- `cargo:errno@0.3.14` — MIT OR Apache-2.0
- `cargo:event-listener@5.4.2` — Apache-2.0 OR MIT
- `cargo:event-listener-strategy@0.5.4` — Apache-2.0 OR MIT
- `cargo:fastrand@2.5.0` — Apache-2.0 OR MIT
- `cargo:fdeflate@0.3.7` — MIT OR Apache-2.0
- `cargo:field-offset@0.3.6` — MIT OR Apache-2.0
- `cargo:find-msvc-tools@0.1.15` — MIT OR Apache-2.0
- `cargo:flate2@1.1.10` — MIT OR Apache-2.0
- `cargo:fnv@1.0.7` — Apache-2.0 OR MIT
- `cargo:foldhash@0.2.0` — Zlib
- `cargo:foreign-types@0.5.0` — MIT OR Apache-2.0
- `cargo:foreign-types-macros@0.2.4` — MIT OR Apache-2.0
- `cargo:foreign-types-shared@0.3.1` — MIT OR Apache-2.0
- `cargo:form_urlencoded@1.2.2` — MIT OR Apache-2.0
- `cargo:futures-channel@0.3.34` — MIT OR Apache-2.0
- `cargo:futures-core@0.3.34` — MIT OR Apache-2.0
- `cargo:futures-executor@0.3.34` — MIT OR Apache-2.0
- `cargo:futures-io@0.3.34` — MIT OR Apache-2.0
- `cargo:futures-lite@2.6.1` — Apache-2.0 OR MIT
- `cargo:futures-macro@0.3.34` — MIT OR Apache-2.0
- `cargo:futures-sink@0.3.34` — MIT OR Apache-2.0
- `cargo:futures-task@0.3.34` — MIT OR Apache-2.0
- `cargo:futures-util@0.3.34` — MIT OR Apache-2.0
- `cargo:gdk@0.18.2` — MIT
- `cargo:gdk-pixbuf@0.18.5` — MIT
- `cargo:gdk-pixbuf-sys@0.18.0` — MIT
- `cargo:gdk-sys@0.18.2` — MIT
- `cargo:gdkwayland-sys@0.18.2` — MIT
- `cargo:gdkx11@0.18.2` — MIT
- `cargo:gdkx11-sys@0.18.2` — MIT
- `cargo:generic-array@0.14.7` — MIT
- `cargo:gethostname@1.1.0` — Apache-2.0
- `cargo:getrandom@0.3.4` — MIT OR Apache-2.0
- `cargo:getrandom@0.4.3` — MIT OR Apache-2.0
- `cargo:gio@0.18.4` — MIT
- `cargo:gio-sys@0.18.1` — MIT
- `cargo:glib@0.18.5` — MIT
- `cargo:glib-macros@0.18.5` — MIT
- `cargo:glib-sys@0.18.1` — MIT
- `cargo:glob@0.3.4` — MIT OR Apache-2.0
- `cargo:global-hotkey@0.8.0` — Apache-2.0 OR MIT
- `cargo:gobject-sys@0.18.0` — MIT
- `cargo:gtk@0.18.2` — MIT
- `cargo:gtk-sys@0.18.2` — MIT
- `cargo:gtk3-macros@0.18.2` — MIT
- `cargo:hashbrown@0.12.3` — MIT OR Apache-2.0
- `cargo:hashbrown@0.17.1` — MIT OR Apache-2.0
- `cargo:heck@0.4.1` — MIT OR Apache-2.0
- `cargo:heck@0.5.0` — MIT OR Apache-2.0
- `cargo:hermit-abi@0.5.3` — MIT OR Apache-2.0
- `cargo:hex@0.4.3` — MIT OR Apache-2.0
- `cargo:html5ever@0.39.0` — MIT OR Apache-2.0
- `cargo:http@1.5.0` — MIT OR Apache-2.0
- `cargo:http-body@1.1.0` — MIT
- `cargo:http-body-util@0.1.5` — MIT
- `cargo:httparse@1.10.1` — MIT OR Apache-2.0
- `cargo:hyper@1.12.0` — MIT
- `cargo:hyper-util@0.1.21` — MIT
- `cargo:iana-time-zone@0.1.65` — MIT OR Apache-2.0
- `cargo:iana-time-zone-haiku@0.1.2` — MIT OR Apache-2.0
- `cargo:ico@0.5.0` — MIT
- `cargo:icu_collections@2.3.0` — Unicode-3.0
- `cargo:icu_locale_core@2.3.0` — Unicode-3.0
- `cargo:icu_normalizer@2.3.0` — Unicode-3.0
- `cargo:icu_normalizer_data@2.3.0` — Unicode-3.0
- `cargo:icu_properties@2.3.0` — Unicode-3.0
- `cargo:icu_properties_data@2.3.0` — Unicode-3.0
- `cargo:icu_provider@2.3.1` — Unicode-3.0
- `cargo:ident_case@1.0.1` — MIT OR Apache-2.0
- `cargo:idna@1.1.0` — MIT OR Apache-2.0
- `cargo:idna_adapter@1.2.2` — Apache-2.0 OR MIT
- `cargo:image@0.25.10` — MIT OR Apache-2.0
- `cargo:indexmap@1.9.3` — Apache-2.0 OR MIT
- `cargo:indexmap@2.14.2` — Apache-2.0 OR MIT
- `cargo:infer@0.22.0` — MIT
- `cargo:ipnet@2.12.2` — MIT OR Apache-2.0
- `cargo:itoa@1.0.18` — MIT OR Apache-2.0
- `cargo:javascriptcore-rs@1.1.2` — MIT
- `cargo:javascriptcore-rs-sys@1.1.1` — MIT
- `cargo:jiff@0.2.38` — Unlicense OR MIT
- `cargo:jiff-core@0.1.1` — Unlicense OR MIT
- `cargo:jiff-static@0.2.38` — Unlicense OR MIT
- `cargo:jiff-tzdb@0.1.9` — Unlicense OR MIT
- `cargo:jiff-tzdb-platform@0.1.3` — Unlicense OR MIT
- `cargo:jni@0.21.1` — MIT OR Apache-2.0
- `cargo:jni-sys@0.3.1` — MIT OR Apache-2.0
- `cargo:jni-sys@0.4.1` — MIT OR Apache-2.0
- `cargo:jni-sys-macros@0.4.1` — MIT OR Apache-2.0
- `cargo:js-sys@0.3.106` — MIT OR Apache-2.0
- `cargo:json-patch@4.2.0` — MIT OR Apache-2.0
- `cargo:jsonptr@0.7.1` — MIT OR Apache-2.0
- `cargo:keyboard-types@0.7.0` — MIT OR Apache-2.0
- `cargo:keyboard-types@0.8.3` — MIT OR Apache-2.0
- `cargo:libappindicator@0.9.0` — Apache-2.0 OR MIT
- `cargo:libappindicator-sys@0.9.0` — Apache-2.0 OR MIT
- `cargo:libc@0.2.191` — MIT OR Apache-2.0
- `cargo:libdbus-sys@0.2.7` — Apache-2.0 OR MIT
- `cargo:libloading@0.7.4` — ISC
- `cargo:libredox@0.1.25` — MIT
- `cargo:linux-raw-sys@0.12.1` — Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT
- `cargo:litemap@0.8.3` — Unicode-3.0
- `cargo:lock_api@0.4.14` — MIT OR Apache-2.0
- `cargo:log@0.4.34` — MIT OR Apache-2.0
- `cargo:markup5ever@0.39.0` — MIT OR Apache-2.0
- `cargo:memchr@2.8.3` — Unlicense OR MIT
- `cargo:memoffset@0.9.1` — MIT
- `cargo:mime@0.3.17` — MIT OR Apache-2.0
- `cargo:miniz_oxide@0.8.9` — MIT OR Zlib OR Apache-2.0
- `cargo:miniz_oxide@0.9.1` — MIT OR Zlib OR Apache-2.0
- `cargo:mio@1.2.4` — MIT
- `cargo:moxcms@0.8.1` — BSD-3-Clause OR Apache-2.0
- `cargo:muda@0.20.0` — Apache-2.0 OR MIT
- `cargo:ndk@0.9.0` — MIT OR Apache-2.0
- `cargo:ndk-context@0.1.1` — MIT OR Apache-2.0
- `cargo:ndk-sys@0.6.0+11769913` — MIT OR Apache-2.0
- `cargo:new_debug_unreachable@1.0.6` — MIT
- `cargo:num-conv@0.2.2` — MIT OR Apache-2.0
- `cargo:num-traits@0.2.19` — MIT OR Apache-2.0
- `cargo:num_enum@0.7.6` — BSD-3-Clause OR MIT OR Apache-2.0
- `cargo:num_enum_derive@0.7.6` — BSD-3-Clause OR MIT OR Apache-2.0
- `cargo:objc2@0.6.5` — MIT
- `cargo:objc2-app-kit@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-cloud-kit@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-core-data@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-core-foundation@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-core-graphics@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-core-image@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-core-location@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-core-text@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-encode@4.1.0` — MIT
- `cargo:objc2-exception-helper@0.1.1` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-foundation@0.3.2` — MIT
- `cargo:objc2-io-surface@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-quartz-core@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-ui-kit@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-user-notifications@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:objc2-web-kit@0.3.2` — Zlib OR Apache-2.0 OR MIT
- `cargo:once_cell@1.21.4` — MIT OR Apache-2.0
- `cargo:option-ext@0.2.0` — MPL-2.0
- `cargo:ordered-stream@0.2.0` — MIT OR Apache-2.0
- `cargo:pango@0.18.3` — MIT
- `cargo:pango-sys@0.18.0` — MIT
- `cargo:parking@2.2.1` — Apache-2.0 OR MIT
- `cargo:parking_lot@0.12.5` — MIT OR Apache-2.0
- `cargo:parking_lot_core@0.9.12` — MIT OR Apache-2.0
- `cargo:percent-encoding@2.3.2` — MIT OR Apache-2.0
- `cargo:phf@0.13.1` — MIT
- `cargo:phf_codegen@0.13.1` — MIT
- `cargo:phf_generator@0.13.1` — MIT
- `cargo:phf_macros@0.13.1` — MIT
- `cargo:phf_shared@0.13.1` — MIT
- `cargo:pin-project-lite@0.2.17` — Apache-2.0 OR MIT
- `cargo:piper@0.2.5` — MIT OR Apache-2.0
- `cargo:pkg-config@0.3.34` — MIT OR Apache-2.0
- `cargo:plist@1.10.2` — MIT
- `cargo:png@0.17.16` — MIT OR Apache-2.0
- `cargo:png@0.18.1` — MIT OR Apache-2.0
- `cargo:polling@3.11.0` — Apache-2.0 OR MIT
- `cargo:portable-atomic@1.15.0` — Apache-2.0 OR MIT
- `cargo:portable-atomic-util@0.2.8` — Apache-2.0 OR MIT
- `cargo:potential_utf@0.1.6` — Unicode-3.0
- `cargo:powerfmt@0.2.1` — MIT OR Apache-2.0
- `cargo:precomputed-hash@0.1.1` — MIT
- `cargo:proc-macro-crate@1.3.1` — MIT OR Apache-2.0
- `cargo:proc-macro-crate@2.0.2` — MIT OR Apache-2.0
- `cargo:proc-macro-crate@3.5.0` — MIT OR Apache-2.0
- `cargo:proc-macro-error@1.0.4` — MIT OR Apache-2.0
- `cargo:proc-macro-error-attr@1.0.4` — MIT OR Apache-2.0
- `cargo:proc-macro2@1.0.107` — MIT OR Apache-2.0
- `cargo:pxfm@0.1.30` — BSD-3-Clause OR Apache-2.0
- `cargo:quick-xml@0.42.0` — MIT
- `cargo:quote@1.0.47` — MIT OR Apache-2.0
- `cargo:r-efi@5.3.0` — MIT OR Apache-2.0 OR LGPL-2.1-or-later
- `cargo:r-efi@6.0.0` — MIT OR Apache-2.0 OR LGPL-2.1-or-later
- `cargo:raw-window-handle@0.6.2` — MIT OR Apache-2.0 OR Zlib
- `cargo:redox_syscall@0.5.18` — MIT
- `cargo:redox_users@0.5.3` — MIT
- `cargo:ref-cast@1.0.27` — MIT OR Apache-2.0
- `cargo:ref-cast-impl@1.0.27` — MIT OR Apache-2.0
- `cargo:regex@1.13.1` — MIT OR Apache-2.0
- `cargo:regex-automata@0.4.18` — MIT OR Apache-2.0
- `cargo:regex-syntax@0.8.11` — MIT OR Apache-2.0
- `cargo:reqwest@0.13.5` — MIT OR Apache-2.0
- `cargo:rustc-hash@2.1.3` — Apache-2.0 OR MIT
- `cargo:rustc_version@0.4.1` — MIT OR Apache-2.0
- `cargo:rustix@1.1.5` — Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT
- `cargo:rustversion@1.0.23` — MIT OR Apache-2.0
- `cargo:same-file@1.0.6` — Unlicense OR MIT
- `cargo:schemars@0.8.22` — MIT
- `cargo:schemars@0.9.0` — MIT
- `cargo:schemars@1.2.3` — MIT
- `cargo:schemars_derive@0.8.22` — MIT
- `cargo:scopeguard@1.2.0` — MIT OR Apache-2.0
- `cargo:selectors@0.38.0` — MPL-2.0
- `cargo:semver@1.0.28` — MIT OR Apache-2.0
- `cargo:serde@1.0.229` — MIT OR Apache-2.0
- `cargo:serde-untagged@0.1.9` — MIT OR Apache-2.0
- `cargo:serde_core@1.0.229` — MIT OR Apache-2.0
- `cargo:serde_derive@1.0.229` — MIT OR Apache-2.0
- `cargo:serde_derive_internals@0.29.1` — MIT OR Apache-2.0
- `cargo:serde_json@1.0.154` — MIT OR Apache-2.0
- `cargo:serde_repr@0.1.21` — MIT OR Apache-2.0
- `cargo:serde_spanned@0.6.9` — MIT OR Apache-2.0
- `cargo:serde_spanned@1.1.2` — MIT OR Apache-2.0
- `cargo:serde_with@3.24.0` — MIT OR Apache-2.0
- `cargo:serde_with_macros@3.24.0` — MIT OR Apache-2.0
- `cargo:serialize-to-javascript@0.1.2` — MIT OR Apache-2.0
- `cargo:serialize-to-javascript-impl@0.1.2` — MIT OR Apache-2.0
- `cargo:servo_arc@0.4.3` — MIT OR Apache-2.0
- `cargo:sha2@0.10.9` — MIT OR Apache-2.0
- `cargo:shlex@2.0.1` — MIT OR Apache-2.0
- `cargo:signal-hook-registry@1.4.8` — MIT OR Apache-2.0
- `cargo:simd-adler32@0.3.10` — MIT
- `cargo:siphasher@1.0.4` — MIT OR Apache-2.0
- `cargo:slab@0.4.12` — MIT
- `cargo:smallvec@1.16.3` — MIT OR Apache-2.0
- `cargo:socket2@0.6.5` — MIT OR Apache-2.0
- `cargo:softbuffer@0.4.8` — MIT OR Apache-2.0
- `cargo:soup3@0.5.0` — MIT
- `cargo:soup3-sys@0.5.0` — MIT
- `cargo:stable_deref_trait@1.2.1` — MIT OR Apache-2.0
- `cargo:static_vcruntime@3.0.0` — MIT OR Apache-2.0 OR Zlib
- `cargo:string_cache@0.9.0` — MIT OR Apache-2.0
- `cargo:string_cache_codegen@0.6.1` — MIT OR Apache-2.0
- `cargo:strsim@0.11.1` — MIT
- `cargo:swift-rs@1.0.9` — MIT OR Apache-2.0
- `cargo:syn@1.0.109` — MIT OR Apache-2.0
- `cargo:syn@2.0.119` — MIT OR Apache-2.0
- `cargo:syn@3.0.9` — MIT OR Apache-2.0
- `cargo:sync_wrapper@1.0.2` — Apache-2.0
- `cargo:synstructure@0.14.0` — MIT
- `cargo:system-deps@6.2.2` — MIT OR Apache-2.0
- `cargo:tao@0.37.1` — Apache-2.0
- `cargo:tao-macros@0.1.4` — MIT OR Apache-2.0
- `cargo:target-lexicon@0.12.16` — Apache-2.0 WITH LLVM-exception
- `cargo:tauri@2.12.4` — Apache-2.0 OR MIT
- `cargo:tauri-build@2.7.3` — Apache-2.0 OR MIT
- `cargo:tauri-codegen@2.7.3` — Apache-2.0 OR MIT
- `cargo:tauri-macros@2.7.3` — Apache-2.0 OR MIT
- `cargo:tauri-plugin@2.7.2` — Apache-2.0 OR MIT
- `cargo:tauri-plugin-global-shortcut@2.4.1` — Apache-2.0 OR MIT
- `cargo:tauri-plugin-single-instance@2.5.4` — Apache-2.0 OR MIT
- `cargo:tauri-runtime@2.12.2` — Apache-2.0 OR MIT
- `cargo:tauri-runtime-wry@2.12.2` — Apache-2.0 OR MIT
- `cargo:tauri-utils@2.10.2` — Apache-2.0 OR MIT
- `cargo:tauri-winres@0.3.7` — MIT
- `cargo:tempfile@3.27.0` — MIT OR Apache-2.0
- `cargo:tendril@0.5.1` — MIT OR Apache-2.0
- `cargo:thiserror@1.0.69` — MIT OR Apache-2.0
- `cargo:thiserror@2.0.21` — MIT OR Apache-2.0
- `cargo:thiserror-impl@1.0.69` — MIT OR Apache-2.0
- `cargo:thiserror-impl@2.0.21` — MIT OR Apache-2.0
- `cargo:time@0.3.55` — MIT OR Apache-2.0
- `cargo:time-core@0.1.9` — MIT OR Apache-2.0
- `cargo:time-macros@0.2.32` — MIT OR Apache-2.0
- `cargo:tinystr@0.8.4` — Unicode-3.0
- `cargo:tinyvec@1.13.3` — Zlib OR Apache-2.0 OR MIT
- `cargo:tokio@1.54.0` — MIT
- `cargo:tokio-util@0.7.20` — MIT
- `cargo:toml@0.8.2` — MIT OR Apache-2.0
- `cargo:toml@1.1.8+spec-1.1.0` — MIT OR Apache-2.0
- `cargo:toml_datetime@0.6.3` — MIT OR Apache-2.0
- `cargo:toml_datetime@1.1.2+spec-1.1.0` — MIT OR Apache-2.0
- `cargo:toml_edit@0.19.15` — MIT OR Apache-2.0
- `cargo:toml_edit@0.20.2` — MIT OR Apache-2.0
- `cargo:toml_edit@0.25.17+spec-1.1.0` — MIT OR Apache-2.0
- `cargo:toml_parser@1.1.5+spec-1.1.0` — MIT OR Apache-2.0
- `cargo:toml_writer@1.1.3+spec-1.1.0` — MIT OR Apache-2.0
- `cargo:tower@0.5.3` — MIT
- `cargo:tower-http@0.6.11` — MIT
- `cargo:tower-layer@0.3.3` — MIT
- `cargo:tower-service@0.3.3` — MIT
- `cargo:tracing@0.1.44` — MIT
- `cargo:tracing-attributes@0.1.31` — MIT
- `cargo:tracing-core@0.1.36` — MIT
- `cargo:tray-icon@0.25.1` — MIT OR Apache-2.0
- `cargo:try-lock@0.2.5` — MIT
- `cargo:typeid@1.0.3` — MIT OR Apache-2.0
- `cargo:typenum@1.20.1` — MIT OR Apache-2.0
- `cargo:uds_windows@1.2.1` — MIT
- `cargo:unicode-ident@1.0.27` — (MIT OR Apache-2.0) AND Unicode-3.0
- `cargo:unicode-segmentation@1.13.3` — MIT OR Apache-2.0
- `cargo:url@2.5.8` — MIT OR Apache-2.0
- `cargo:urlpattern@0.6.0` — MIT
- `cargo:utf8_iter@1.0.4` — Apache-2.0 OR MIT
- `cargo:uuid@1.28.0` — Apache-2.0 OR MIT
- `cargo:version-compare@0.2.1` — MIT
- `cargo:version_check@0.9.5` — MIT OR Apache-2.0
- `cargo:vswhom@0.1.0` — MIT
- `cargo:vswhom-sys@0.1.3` — MIT
- `cargo:walkdir@2.5.0` — Unlicense OR MIT
- `cargo:want@0.3.2` — MIT
- `cargo:wasi@0.11.1+wasi-snapshot-preview1` — Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT
- `cargo:wasip2@1.0.4+wasi-0.2.12` — Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT
- `cargo:wasm-bindgen@0.2.129` — MIT OR Apache-2.0
- `cargo:wasm-bindgen-futures@0.4.79` — MIT OR Apache-2.0
- `cargo:wasm-bindgen-macro@0.2.129` — MIT OR Apache-2.0
- `cargo:wasm-bindgen-macro-support@0.2.129` — MIT OR Apache-2.0
- `cargo:wasm-bindgen-shared@0.2.129` — MIT OR Apache-2.0
- `cargo:wasm-streams@0.5.0` — MIT OR Apache-2.0
- `cargo:web-sys@0.3.106` — MIT OR Apache-2.0
- `cargo:web-time@1.1.0` — MIT OR Apache-2.0
- `cargo:web_atoms@0.2.6` — MIT OR Apache-2.0
- `cargo:webkit2gtk@2.0.2` — MIT
- `cargo:webkit2gtk-sys@2.0.2` — MIT
- `cargo:webview2-com@0.39.1` — MIT
- `cargo:webview2-com-macros@0.8.1` — MIT
- `cargo:webview2-com-sys@0.39.1` — MIT
- `cargo:winapi@0.3.9` — MIT OR Apache-2.0
- `cargo:winapi-i686-pc-windows-gnu@0.4.0` — MIT OR Apache-2.0
- `cargo:winapi-util@0.1.11` — Unlicense OR MIT
- `cargo:winapi-x86_64-pc-windows-gnu@0.4.0` — MIT OR Apache-2.0
- `cargo:window-vibrancy@0.8.1` — Apache-2.0 OR MIT
- `cargo:windows@0.62.2` — MIT OR Apache-2.0
- `cargo:windows-collections@0.3.2` — MIT OR Apache-2.0
- `cargo:windows-core@0.62.2` — MIT OR Apache-2.0
- `cargo:windows-future@0.3.2` — MIT OR Apache-2.0
- `cargo:windows-implement@0.60.2` — MIT OR Apache-2.0
- `cargo:windows-interface@0.59.3` — MIT OR Apache-2.0
- `cargo:windows-link@0.2.1` — MIT OR Apache-2.0
- `cargo:windows-numerics@0.3.1` — MIT OR Apache-2.0
- `cargo:windows-result@0.4.1` — MIT OR Apache-2.0
- `cargo:windows-strings@0.5.1` — MIT OR Apache-2.0
- `cargo:windows-sys@0.45.0` — MIT OR Apache-2.0
- `cargo:windows-sys@0.59.0` — MIT OR Apache-2.0
- `cargo:windows-sys@0.61.2` — MIT OR Apache-2.0
- `cargo:windows-targets@0.42.2` — MIT OR Apache-2.0
- `cargo:windows-targets@0.52.6` — MIT OR Apache-2.0
- `cargo:windows-threading@0.2.1` — MIT OR Apache-2.0
- `cargo:windows-version@0.1.7` — MIT OR Apache-2.0
- `cargo:windows_aarch64_gnullvm@0.42.2` — MIT OR Apache-2.0
- `cargo:windows_aarch64_gnullvm@0.52.6` — MIT OR Apache-2.0
- `cargo:windows_aarch64_msvc@0.42.2` — MIT OR Apache-2.0
- `cargo:windows_aarch64_msvc@0.52.6` — MIT OR Apache-2.0
- `cargo:windows_i686_gnu@0.42.2` — MIT OR Apache-2.0
- `cargo:windows_i686_gnu@0.52.6` — MIT OR Apache-2.0
- `cargo:windows_i686_gnullvm@0.52.6` — MIT OR Apache-2.0
- `cargo:windows_i686_msvc@0.42.2` — MIT OR Apache-2.0
- `cargo:windows_i686_msvc@0.52.6` — MIT OR Apache-2.0
- `cargo:windows_x86_64_gnu@0.42.2` — MIT OR Apache-2.0
- `cargo:windows_x86_64_gnu@0.52.6` — MIT OR Apache-2.0
- `cargo:windows_x86_64_gnullvm@0.42.2` — MIT OR Apache-2.0
- `cargo:windows_x86_64_gnullvm@0.52.6` — MIT OR Apache-2.0
- `cargo:windows_x86_64_msvc@0.42.2` — MIT OR Apache-2.0
- `cargo:windows_x86_64_msvc@0.52.6` — MIT OR Apache-2.0
- `cargo:winnow@0.5.40` — MIT
- `cargo:winnow@1.0.4` — MIT
- `cargo:winreg@0.56.0` — MIT
- `cargo:wit-bindgen@0.57.1` — Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT
- `cargo:writeable@0.6.4` — Unicode-3.0
- `cargo:wry@0.57.0` — Apache-2.0 OR MIT
- `cargo:x11@2.21.0` — MIT
- `cargo:x11-dl@2.21.0` — MIT
- `cargo:x11rb@0.13.2` — MIT OR Apache-2.0
- `cargo:x11rb-protocol@0.13.2` — MIT OR Apache-2.0
- `cargo:xkeysym@0.2.1` — MIT OR Apache-2.0 OR Zlib
- `cargo:yoke@0.8.3` — Unicode-3.0
- `cargo:yoke-derive@0.8.4` — Unicode-3.0
- `cargo:zbus@5.19.0` — MIT
- `cargo:zbus_macros@5.19.0` — MIT
- `cargo:zbus_names@4.3.5` — MIT
- `cargo:zcheapstr@1.1.0` — MIT
- `cargo:zerofrom@0.1.8` — Unicode-3.0
- `cargo:zerofrom-derive@0.1.8` — Unicode-3.0
- `cargo:zerotrie@0.2.5` — Unicode-3.0
- `cargo:zerovec@0.11.8` — Unicode-3.0
- `cargo:zerovec-derive@0.11.6` — Unicode-3.0
- `cargo:zlib-rs@0.6.8` — Zlib
- `cargo:zmij@1.0.23` — MIT
- `cargo:zvariant@5.15.1` — MIT
- `cargo:zvariant_derive@5.15.1` — MIT
- `cargo:zvariant_utils@4.2.0` — MIT
- `python:Jinja2@3.1.6` — BSD-3-Clause
- `python:MarkupSafe@3.0.4` — BSD-3-Clause
- `python:PyYAML@6.0.3` — MIT
- `python:Pygments@2.21.0` — BSD-2-Clause
- `python:aiohappyeyeballs@2.7.1` — PSF-2.0
- `python:aiohttp@3.14.3` — Apache-2.0 AND MIT
- `python:aiosignal@1.4.0` — Apache-2.0
- `python:annotated-types@0.8.0` — MIT
- `python:anyio@4.15.1` — MIT
- `python:attrs@26.1.0` — MIT
- `python:boto3@1.43.108` — Apache-2.0
- `python:botocore@1.43.108` — Apache-2.0
- `python:certifi@2026.7.22` — MPL-2.0
- `python:cffi@2.1.1` — MIT-0
- `python:charset-normalizer@3.5.2` — MIT
- `python:click@8.5.0` — BSD-3-Clause
- `python:dbus-fast@5.2.0` — MIT
- `python:distro@1.9.0` — Apache-2.0
- `python:docopt@0.6.2` — MIT
- `python:fastuuid@0.14.0` — BSD-3-Clause
- `python:filelock@3.32.7` — MIT
- `python:frozenlist@1.8.0` — Apache-2.0
- `python:fsspec@2026.9.0` — BSD-3-Clause
- `python:h11@0.16.0` — MIT
- `python:h2@4.4.1` — MIT
- `python:hf-xet@1.6.0` — Apache-2.0
- `python:hpack@4.2.0` — MIT
- `python:httpcore@1.0.9` — BSD-3-Clause
- `python:httpx@0.28.1` — BSD-3-Clause
- `python:huggingface_hub@1.33.0` — Apache-2.0
- `python:hyperframe@6.1.0` — MIT
- `python:idna@3.20` — BSD-3-Clause
- `python:importlib_metadata@8.9.0` — Apache-2.0
- `python:iniconfig@2.3.0` — MIT
- `python:jiter@0.17.0` — MIT
- `python:jmespath@1.1.0` — MIT
- `python:jsonschema@4.26.0` — MIT
- `python:jsonschema-specifications@2025.9.1` — MIT
- `python:litellm@1.104.0` — MIT
- `python:multidict@6.9.1` — Apache-2.0
- `python:nodeenv@1.11.0` — BSD-3-Clause
- `python:num2words@0.5.14` — LGPL-2.1-or-later
- `python:numpy@2.5.3` — BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0
- `python:openai@2.54.0` — Apache-2.0
- `python:packaging@26.3` — Apache-2.0 OR BSD-2-Clause
- `python:pathspec@1.1.1` — MPL-2.0
- `python:pillow@12.3.0` — MIT-CMU
- `python:pluggy@1.6.0` — MIT
- `python:propcache@0.5.4` — Apache-2.0
- `python:psutil@7.2.2` — BSD-3-Clause
- `python:pulsectl@24.12.0` — MIT
- `python:pycparser@3.0` — BSD-3-Clause
- `python:pydantic@2.13.5` — MIT
- `python:pydantic-settings@2.15.0` — MIT
- `python:pydantic_core@2.46.5` — MIT
- `python:pyperclip@1.11.0` — BSD-3-Clause
- `python:pyright@1.1.414` — MIT
- `python:pytest@9.1.1` — MIT
- `python:pytest-asyncio@1.4.0` — Apache-2.0
- `python:python-dateutil@2.9.0.post0` — Apache-2.0 OR BSD-3-Clause
- `python:python-dotenv@1.2.4` — BSD-3-Clause
- `python:referencing@0.37.0` — MIT
- `python:regex@2026.9.29` — Apache-2.0 AND CNRI-Python
- `python:requests@2.34.2` — Apache-2.0
- `python:rpds-py@2026.9.1` — MIT
- `python:ruff@0.16.10` — MIT
- `python:s3transfer@0.19.2` — Apache-2.0
- `python:sherpa-onnx-core@1.13.8` — Apache-2.0
- `python:sherpa_onnx@1.13.8` — Apache-2.0
- `python:six@1.17.0` — MIT
- `python:sniffio@1.3.1` — MIT OR Apache-2.0
- `python:sounddevice@0.5.6` — MIT
- `python:tenacity@9.1.4` — Apache-2.0
- `python:tiktoken@0.14.0` — MIT
- `python:tokenizers@0.23.2` — Apache-2.0
- `python:tqdm@4.70.1` — MPL-2.0 AND MIT
- `python:transcribe-cpp@0.4.0` — MIT
- `python:transcribe-cpp-native@0.4.0` — MIT
- `python:typing-inspection@0.4.4` — MIT
- `python:typing_extensions@4.16.0` — PSF-2.0
- `python:urllib3@2.8.0` — MIT
- `python:yarl@1.25.1` — Apache-2.0
- `python:zipp@4.1.1` — MIT
<!-- END GENERATED DEPENDENCY INVENTORY -->
