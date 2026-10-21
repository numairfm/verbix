#!/usr/bin/env python3
"""Release gate for dependency licenses and shipped legal files.

The gate has two jobs:

1. Scan every Cargo and Python dependency, including transitive dependencies.
2. Prove that ``THIRD-PARTY-NOTICES.md``, ``DISTRIBUTION-TERMS.md``, and the
   ``LICENSES/`` directory are complete and ship in the AppImage, and that the
   dated Supertonic preset risk acceptance is valid in the source.

``--check`` is read-only. ``--write`` regenerates the generated dependency
inventory section in ``THIRD-PARTY-NOTICES.md`` and changes nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

BEGIN_MARKER = "<!-- BEGIN GENERATED DEPENDENCY INVENTORY -->"
END_MARKER = "<!-- END GENERATED DEPENDENCY INVENTORY -->"

# Supertonic preset voices are treated as part of the OpenRAIL-M model, not as
# separately licensed voice assets. The product owner accepts the unpublished
# speaker-provenance risk in a dated record. This is deliberately not a
# clearance: there is no speaker-consent evidence to clear.
VOICE_RISK_MARKER = "Preset voice risk acceptance:"
VOICE_RISK_RECORD = "docs/legal/2026-10-21-supertonic-3-preset-risk-acceptance.md"
VOICE_RISK_STATUS = "Status: ACCEPTED RESIDUAL RISK"
VOICE_RISK_REVISIONS = (
    "cca5a0e6c96e1d2c720986bf7e75fcc81dee3ae4",
    "3cadd1ee6394adea1bd021217a0e650ede09a323",
)
BUNDLED_TTS_LICENSE = (
    "src/verbix/voice/tts/supertonic-int8/LICENSE-OpenRAIL-M"
)
BUNDLED_TTS_LICENSE_SHA256 = (
    "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f"
)
ATTACHMENT_A_START = "You agree not to use the Model or Derivatives of the Model:"

# The closed-source commercial track needs a human statement for these. The
# generated inventory proves the versions and licenses; this list proves that
# the notice file explains the obligations that matter for the product.
REQUIRED_CURATED_RECORDS = (
    "Tauri 2",
    "tauri-plugin-single-instance",
    "tauri-plugin-global-shortcut",
    "serde",
    "serde_json",
    "tokio",
    "libc",
    "CPython 3.13",
    "WebKitGTK 4.1",
    "GTK 3",
    "AppIndicator",
    "PortAudio",
    "PipeWire",
    "num2words",
    "Supertonic 3",
    "Synthetic audio disclosure",
    "Preset identifiers are labels",
    "Nemotron 3.5 ASR",
    "dynamically linked",
    "Source offer",
)

# Every SPDX licence id the gate accepts must have a shipped text. The value is
# the path under LICENSES/. Keep this closed: a new dependency with a new
# licence fails the gate until a human adds the text and a record.
LICENSE_FILES: dict[str, str] = {
    "0BSD": "0bsd.md",
    "Apache-2.0": "apache-2.0.md",
    "BSD-2-Clause": "bsd-2-clause.md",
    "BSD-3-Clause": "bsd-3-clause.md",
    "CC0-1.0": "cc0-1.0.md",
    "CNRI-Python": "cnri-python.md",
    "GCC-exception-3.1": "gcc-exception-3.1.md",
    "ISC": "isc.md",
    "LGPL-2.0-or-later": "lgpl-2.0-or-later.md",
    "LGPL-2.1-only": "lgpl-2.1-or-later.md",
    "LGPL-2.1-or-later": "lgpl-2.1-or-later.md",
    "LGPL-3.0": "lgpl-3.0.md",
    "LGPL-3.0-only": "lgpl-3.0.md",
    "LLVM-exception": "llvm-exception.md",
    "MIT": "mit.md",
    "MIT-0": "mit-0.md",
    "MIT-CMU": "mit-cmu.md",
    "MPL-2.0": "mpl-2.0.md",
    "PSF-2.0": "psf-2.0.md",
    "Unicode-3.0": "unicode-3.0.md",
    "Unlicense": "unlicense.md",
    "Zlib": "zlib.md",
}

# The voice assets are not code dependencies in Cargo or uv, but they ship with
# the product and their notices are contractual.
ASSET_LICENSE_FILES: dict[str, str] = {
    "BigScience OpenRAIL-M": "bigscience-openrail-m.md",
    "OpenMDW-1.1": "openmdw-1.1.md",
}

LICENSE_ALIASES: dict[str, str] = {
    "Apache 2.0": "Apache-2.0",
    "Apache License 2.0": "Apache-2.0",
    "Apache License, Version 2.0": "Apache-2.0",
    "Apache Licensed": "Apache-2.0",
    "BSD": "BSD-3-Clause",
    "BSD License": "BSD-3-Clause",
    "Dual License": "Apache-2.0 OR BSD-3-Clause",
    "LGPL": "LGPL-2.1-or-later",
    "MIT License": "MIT",
    "PSF": "PSF-2.0",
    "Python Software Foundation License": "PSF-2.0",
}

CLASSIFIER_LICENSES: dict[str, str] = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)": (
        "LGPL-2.1-or-later"
    ),
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}

_BAD_LICENSE_VALUES = {"", "UNKNOWN", "UNRESOLVED", "NOASSERTION", "NONE"}
_FULL_LICENSE_HINTS = (
    ("permission is hereby granted", "MIT"),
    ("apache license", "Apache-2.0"),
    ("gnu lesser general public license", "LGPL-2.1-or-later"),
    ("mozilla public license", "MPL-2.0"),
    ("redistribution and use in source and binary forms", "BSD-3-Clause"),
)


@dataclass(frozen=True)
class Dependency:
    ecosystem: str
    name: str
    version: str
    license: str

    @property
    def key(self) -> str:
        return f"{self.ecosystem}:{self.name}@{self.version}"


class LicenseError(ValueError):
    """Raised when a dependency does not declare a usable licence."""


def canonicalize_expression(expression: str) -> str:
    """Normalises Cargo and Python spelling variants into a stable expression."""
    value = " ".join(expression.strip().split())
    if not value:
        raise LicenseError("missing licence expression")
    if value.upper() in _BAD_LICENSE_VALUES:
        raise LicenseError(f"unresolved licence expression: {value}")
    if value.upper().startswith(("UNRESOLVED", "UNKNOWN", "NOASSERTION")):
        raise LicenseError(f"unresolved licence expression: {value}")
    if value.startswith("LicenseRef-"):
        raise LicenseError(f"unresolved licence expression: {value}")

    lowered = value.lower()
    if lowered in {"mit", "the mit license (mit)"} or lowered.startswith(
        "mit license"
    ):
        return "MIT"
    if "apache licensed, as found in the license file" in lowered:
        return "Apache-2.0"
    if lowered in {"apache 2.0", "apache license 2.0", "apache license, version 2.0"}:
        return "Apache-2.0"
    if lowered == "bsd":
        return "BSD-3-Clause"
    if lowered == "dual license":
        return "Apache-2.0 OR BSD-3-Clause"
    if lowered == "lgpl":
        return "LGPL-2.1-or-later"

    # Cargo allows the legacy slash form (MIT/Apache-2.0).
    if "/" in value and " OR " not in value and " AND " not in value:
        parts = [part.strip() for part in value.split("/") if part.strip()]
        value = " OR ".join(parts)

    # Long licence fields are common in older Python metadata. Match the
    # canonical opening line rather than guessing from a substring.
    if len(value) > 80:
        for hint, licence in _FULL_LICENSE_HINTS:
            if hint in lowered:
                return licence

    aliases = {key.lower(): val for key, val in LICENSE_ALIASES.items()}
    return aliases.get(lowered, value)


def python_license(metadata: Mapping[str, object]) -> str:
    """Reads a licence from PEP 639 metadata, classifiers, or the legacy field."""
    expression = metadata.get("License-Expression")
    if isinstance(expression, str) and expression.strip():
        return canonicalize_expression(expression)

    classifier_values: list[str] = []
    classifiers = metadata.get("Classifier")
    if isinstance(classifiers, str):
        classifier_values = [classifiers]
    elif isinstance(classifiers, Iterable):
        classifier_values = [str(item) for item in classifiers]

    classifier_licenses = [
        CLASSIFIER_LICENSES[item]
        for item in classifier_values
        if item in CLASSIFIER_LICENSES
    ]
    if classifier_licenses:
        unique = list(dict.fromkeys(classifier_licenses))
        if len(unique) == 1:
            return unique[0]
        if set(unique) == {"Apache-2.0", "BSD-3-Clause"}:
            return "Apache-2.0 OR BSD-3-Clause"
        return " OR ".join(unique)

    legacy = metadata.get("License")
    if isinstance(legacy, str) and legacy.strip():
        return canonicalize_expression(legacy)
    raise LicenseError("missing licence metadata")


def render_inventory(dependencies: Iterable[Dependency]) -> str:
    """Renders the deterministic generated section body, without markers."""
    lines = [
        f"- `{dependency.key}` — {dependency.license}"
        for dependency in sorted(
            dependencies, key=lambda item: (item.ecosystem, item.name, item.version)
        )
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def curated_text(notices: str) -> str:
    """Returns everything a human wrote, excluding the generated inventory."""
    if BEGIN_MARKER not in notices:
        return notices
    before, _, remainder = notices.partition(BEGIN_MARKER)
    _, _, after = remainder.partition(END_MARKER)
    return before + after


def _generated_block(notices: str) -> str | None:
    if notices.count(BEGIN_MARKER) != 1 or notices.count(END_MARKER) != 1:
        return None
    before, _, remainder = notices.partition(BEGIN_MARKER)
    body, _, after = remainder.partition(END_MARKER)
    if BEGIN_MARKER in after or END_MARKER in before:
        return None
    return body.lstrip("\n")


def update_notices_text(notices: str, dependencies: Iterable[Dependency]) -> str:
    """Replaces only the generated block. Never touches curated text."""
    block = f"{BEGIN_MARKER}\n{render_inventory(dependencies)}{END_MARKER}"
    if BEGIN_MARKER in notices and END_MARKER in notices:
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        replaced, count = pattern.subn(block, notices, count=1)
        if count == 1:
            return replaced
    suffix = "" if notices.endswith("\n") else "\n"
    return f"{notices}{suffix}\n## Generated Dependency Inventory\n\n{block}\n"


def _inventory_errors(notices: str, dependencies: list[Dependency]) -> list[str]:
    block = _generated_block(notices)
    if block is None:
        return [
            "THIRD-PARTY-NOTICES.md must contain exactly one generated inventory block"
        ]
    expected = render_inventory(dependencies)
    if block == expected:
        return []

    expected_lines = set(expected.splitlines())
    actual_lines = set(block.splitlines())
    missing = sorted(expected_lines - actual_lines)
    extra = sorted(actual_lines - expected_lines)
    errors = ["generated dependency inventory has drifted; run --write"]
    errors.extend(f"missing from inventory: {line}" for line in missing)
    errors.extend(f"not found in scanned dependency set: {line}" for line in extra)
    return errors


def _license_expression_errors(dependencies: list[Dependency]) -> list[str]:
    errors: list[str] = []
    for dependency in dependencies:
        try:
            canonical = canonicalize_expression(dependency.license)
        except LicenseError as error:
            errors.append(f"{dependency.key}: {error}")
            continue
        if canonical != dependency.license:
            errors.append(
                f"{dependency.key}: non-canonical licence {dependency.license!r}; "
                f"use {canonical!r}"
            )
    return errors


def license_tokens(dependencies: Iterable[Dependency]) -> set[str]:
    tokens: set[str] = set()
    for dependency in dependencies:
        try:
            expression = canonicalize_expression(dependency.license)
        except LicenseError:
            continue
        for token in re.findall(r"[A-Za-z0-9.+-]+", expression):
            if token not in {"OR", "AND", "WITH"}:
                tokens.add(token)
    return tokens


def required_license_files(dependencies: Iterable[Dependency]) -> list[str]:
    """Returns every license text path needed by the scanned tree."""
    files: set[str] = set(ASSET_LICENSE_FILES.values())
    for token in license_tokens(dependencies):
        filename = LICENSE_FILES.get(token)
        if filename is not None:
            files.add(filename)
    return sorted(files)


def _license_file_errors(root: Path, dependencies: list[Dependency]) -> list[str]:
    errors: list[str] = []
    known_tokens = set(LICENSE_FILES)
    for token in sorted(license_tokens(dependencies) - known_tokens):
        errors.append(
            f"no shipped licence text mapping for {token!r}; add it to LICENSES/ "
            "and LICENSE_FILES in scripts/check_licenses.py"
        )
    for filename in required_license_files(dependencies):
        path = root / "LICENSES" / filename
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty licence text: {path}")
    return errors


def _curated_record_errors(notices: str) -> list[str]:
    visible = curated_text(notices)
    return [
        f"THIRD-PARTY-NOTICES.md is missing the curated record: {record}"
        for record in REQUIRED_CURATED_RECORDS
        if record not in visible
    ]


def _normalise_legal_text(text: str) -> str:
    return " ".join(text.split())


def _distribution_terms_errors(root: Path) -> list[str]:
    """Checks that the terms carry the pinned license's Attachment A verbatim."""
    terms_path = root / "DISTRIBUTION-TERMS.md"
    license_path = root / "LICENSES" / "bigscience-openrail-m.md"
    bundled_path = root / BUNDLED_TTS_LICENSE
    if not terms_path.is_file():
        return ["DISTRIBUTION-TERMS.md is missing"]
    if not license_path.is_file():
        return ["LICENSES/bigscience-openrail-m.md is missing"]
    if not bundled_path.is_file():
        return [f"{BUNDLED_TTS_LICENSE} is missing"]
    terms = terms_path.read_text(encoding="utf-8")
    license_text = license_path.read_text(encoding="utf-8")
    bundled_text = bundled_path.read_text(encoding="utf-8")
    errors: list[str] = []
    if _normalise_legal_text(bundled_text) not in _normalise_legal_text(license_text):
        errors.append(
            "LICENSES/bigscience-openrail-m.md does not reproduce the pinned "
            "upstream license text"
        )
    if ATTACHMENT_A_START not in bundled_text:
        errors.append(f"{BUNDLED_TTS_LICENSE} does not contain Attachment A")
    elif ATTACHMENT_A_START not in terms:
        errors.append(
            "DISTRIBUTION-TERMS.md does not include the binding Attachment A text"
        )
    else:
        license_attachment = _normalise_legal_text(
            bundled_text[bundled_text.index(ATTACHMENT_A_START) :]
        )
        terms_attachment = _normalise_legal_text(
            terms[terms.index(ATTACHMENT_A_START) :]
        )
        if license_attachment not in terms_attachment:
            errors.append(
                "DISTRIBUTION-TERMS.md does not reproduce Attachment A verbatim"
            )
    lowered = terms.lower()
    if "synthetic" not in lowered or "machine generated" not in lowered:
        errors.append(
            "DISTRIBUTION-TERMS.md must disclose synthetic machine-generated audio"
        )
    if "not names" not in lowered or "real people" not in lowered:
        errors.append(
            "DISTRIBUTION-TERMS.md must state that preset identifiers are labels, "
            "not names of real people"
        )
    return errors


def _voice_risk_errors(root: Path, notices: str) -> list[str]:
    errors: list[str] = []
    visible = curated_text(notices)
    lines = [line for line in visible.splitlines() if VOICE_RISK_MARKER in line]
    if not lines:
        errors.append(
            "voice risk acceptance: THIRD-PARTY-NOTICES.md is missing "
            f"{VOICE_RISK_MARKER}"
        )
    elif len(lines) > 1:
        errors.append(
            "voice risk acceptance: THIRD-PARTY-NOTICES.md has more than one "
            f"{VOICE_RISK_MARKER} line"
        )
    else:
        line = lines[0]
        if "**ACCEPTED**" not in line:
            errors.append(
                "voice risk acceptance: THIRD-PARTY-NOTICES.md does not mark the "
                "risk accepted"
            )
        if VOICE_RISK_RECORD not in line:
            errors.append(
                "voice risk acceptance: THIRD-PARTY-NOTICES.md does not name the "
                f"dated record {VOICE_RISK_RECORD}"
            )

    record_path = root / VOICE_RISK_RECORD
    if not record_path.is_file() or record_path.stat().st_size == 0:
        errors.append(f"voice risk acceptance: missing dated record {record_path}")
    else:
        record = record_path.read_text(encoding="utf-8")
        if VOICE_RISK_STATUS not in record:
            errors.append(
                "voice risk acceptance: record Status must be "
                f"{VOICE_RISK_STATUS!r}"
            )
        date_line = re.search(
            r"^Date: (\d{4}-\d{2}-\d{2})$", record, re.MULTILINE
        )
        if date_line is None:
            errors.append(
                "voice risk acceptance: record Date must be an ISO date "
                "(YYYY-MM-DD)"
            )
        else:
            try:
                accepted_date = date.fromisoformat(date_line.group(1))
            except ValueError:
                errors.append(
                    "voice risk acceptance: record Date is not a real calendar date"
                )
            else:
                if accepted_date > datetime.now(tz=UTC).astimezone().date():
                    errors.append(
                        "voice risk acceptance: record Date cannot be in the future"
                    )
        accepted = re.search(r"^Accepted by: (.+)$", record, re.MULTILINE)
        if accepted is None or not accepted.group(1).strip():
            errors.append("voice risk acceptance: record must record Accepted by")
        if "OpenRAIL-M" not in record:
            errors.append(
                "voice risk acceptance: record must name the OpenRAIL-M grant"
            )
        if "not a legal clearance" not in record.lower():
            errors.append(
                "voice risk acceptance: record must say this is not a legal "
                "clearance"
            )
        if "M1" not in record or "F5" not in record:
            errors.append(
                "voice risk acceptance: record must cover the preset families "
                "M1-M5 and F1-F5"
            )
        for revision in VOICE_RISK_REVISIONS:
            if revision not in record:
                errors.append(
                    "voice risk acceptance: record must pin revision "
                    f"{revision}"
                )

    bundled_license = root / BUNDLED_TTS_LICENSE
    if not bundled_license.is_file() or bundled_license.stat().st_size == 0:
        errors.append(
            "voice risk acceptance: missing bundled TTS license text "
            f"{BUNDLED_TTS_LICENSE}"
        )
    else:
        digest = hashlib.sha256(bundled_license.read_bytes()).hexdigest()
        if digest != BUNDLED_TTS_LICENSE_SHA256:
            errors.append(
                "voice risk acceptance: bundled TTS license text "
                f"{BUNDLED_TTS_LICENSE} does not match the pinned license"
            )
    return errors


def evaluate(
    root: Path,
    dependencies: list[Dependency],
    notices_text: str,
) -> list[str]:
    """Runs every in-repository check. Pure: never writes a file."""
    errors: list[str] = []
    errors.extend(_license_expression_errors(dependencies))
    errors.extend(_inventory_errors(notices_text, dependencies))
    errors.extend(_curated_record_errors(notices_text))
    errors.extend(_voice_risk_errors(root, notices_text))
    errors.extend(_distribution_terms_errors(root))
    errors.extend(_license_file_errors(root, dependencies))
    return errors


def scan_cargo(manifest_path: Path, cargo: str = "cargo") -> list[Dependency]:
    """Scans the full Cargo metadata graph, including transitive and dev crates."""
    result = subprocess.run(
        [
            cargo,
            "metadata",
            "--locked",
            "--format-version",
            "1",
            "--manifest-path",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    dependencies: list[Dependency] = []
    for package in metadata["packages"]:
        # Workspace/path members are our own code, not third-party dependencies.
        if package.get("source") is None:
            continue
        license_text = package.get("license")
        if not license_text:
            license_file = package.get("license_file")
            if license_file:
                license_text = f"LicenseRef-file:{license_file}"
            else:
                license_text = ""
        try:
            license_text = canonicalize_expression(license_text)
        except LicenseError:
            pass
        dependencies.append(
            Dependency(
                ecosystem="cargo",
                name=package["name"],
                version=package["version"],
                license=license_text,
            )
        )
    return dependencies


def _metadata_values(metadata: importlib.metadata.PackageMetadata, key: str) -> list[str]:
    values = metadata.get_all(key)
    return [str(value) for value in values or []]


def normalize_name(name: str) -> str:
    """PEP 503 name normalisation shared by Python metadata and uv.lock."""
    return re.sub(r"[-_.]+", "-", name).lower()


def scan_python(site_packages: Path) -> list[Dependency]:
    """Scans installed dist-info metadata, which includes transitive wheels."""
    dependencies: list[Dependency] = []
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = distribution.metadata.get("Name")
        if not name or name.lower() == "verbix":
            continue
        metadata: dict[str, object] = {
            "License-Expression": distribution.metadata.get("License-Expression"),
            "License": distribution.metadata.get("License"),
            "Classifier": _metadata_values(distribution.metadata, "Classifier"),
        }
        try:
            license_text = python_license(metadata)
            license_text = canonicalize_expression(license_text)
        except LicenseError as error:
            license_text = f"UNRESOLVED ({error})"
        dependencies.append(
            Dependency(
                ecosystem="python",
                name=name,
                version=distribution.version,
                license=license_text,
            )
        )
    return dependencies


def find_site_packages(root: Path) -> Path:
    candidates = sorted((root / ".venv" / "lib").glob("python*/site-packages"))
    if not candidates:
        raise FileNotFoundError(
            f"no .venv site-packages under {root}; run 'uv sync --locked' first"
        )
    return candidates[-1]


def lock_packages(lock_path: Path) -> set[tuple[str, str]]:
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    return {
        (normalize_name(package["name"]), package["version"])
        for package in lock.get("package", [])
        if "name" in package and "version" in package
    }


def check_uv_lock_coverage(lock_path: Path, dependencies: list[Dependency]) -> list[str]:
    locked = lock_packages(lock_path)
    errors: list[str] = []
    for dependency in dependencies:
        if dependency.ecosystem != "python":
            continue
        if (normalize_name(dependency.name), dependency.version) not in locked:
            errors.append(
                f"{dependency.key}: installed distribution is not in uv.lock"
            )
    return errors


def verify_extracted_licenses(
    extracted_root: Path,
    *,
    root: Path,
    expected_license_files: list[str],
) -> list[str]:
    """Checks the resource tree inside an extracted AppImage."""
    errors: list[str] = []
    candidates = sorted(extracted_root.rglob("THIRD-PARTY-NOTICES.md"))
    if not candidates:
        return [
            "AppImage does not contain THIRD-PARTY-NOTICES.md",
            "AppImage does not contain DISTRIBUTION-TERMS.md",
            "AppImage does not contain LICENSES/",
        ]
    resource_dir = candidates[0].parent
    notices = resource_dir / "THIRD-PARTY-NOTICES.md"
    terms = resource_dir / "DISTRIBUTION-TERMS.md"
    licenses = resource_dir / "LICENSES"
    if not notices.is_file():
        errors.append("AppImage is missing THIRD-PARTY-NOTICES.md")
    elif notices.read_bytes() != (root / "THIRD-PARTY-NOTICES.md").read_bytes():
        errors.append("AppImage THIRD-PARTY-NOTICES.md does not match the source file")
    if not terms.is_file():
        errors.append("AppImage is missing DISTRIBUTION-TERMS.md")
    elif terms.read_bytes() != (root / "DISTRIBUTION-TERMS.md").read_bytes():
        errors.append("AppImage DISTRIBUTION-TERMS.md does not match the source file")
    if not licenses.is_dir():
        errors.append("AppImage is missing LICENSES/")
        return errors
    for filename in expected_license_files:
        path = licenses / filename
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"AppImage is missing licence text LICENSES/{filename}")
    return errors


def extract_appimage(appimage: Path, destination: Path) -> Path:
    subprocess.run(
        [str(appimage), "--appimage-extract"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    )
    return destination / "squashfs-root"


def appimage_version(tauri_config: Path) -> str:
    return json.loads(tauri_config.read_text(encoding="utf-8"))["version"]


def run_gate(
    root: Path,
    *,
    write: bool = False,
    appimage: Path | None = None,
    cargo: str = "cargo",
    artifact_only: bool = False,
) -> list[str]:
    if artifact_only and appimage is None:
        raise ValueError("artifact_only requires an AppImage")

    dependencies = scan_cargo(root / "src-tauri" / "Cargo.toml", cargo=cargo)
    dependencies.extend(scan_python(find_site_packages(root)))

    if artifact_only:
        if not appimage.is_file():
            return [f"AppImage not found: {appimage}"]
        with tempfile.TemporaryDirectory(prefix="verbix-license-") as tmp:
            extracted = extract_appimage(appimage, Path(tmp))
            return verify_extracted_licenses(
                extracted,
                root=root,
                expected_license_files=required_license_files(dependencies),
            )

    notices_path = root / "THIRD-PARTY-NOTICES.md"
    notices = notices_path.read_text(encoding="utf-8")

    if write:
        updated = update_notices_text(notices, dependencies)
        if updated != notices:
            notices_path.write_text(updated, encoding="utf-8")
            notices = updated

    errors = evaluate(root, dependencies, notices)
    errors.extend(check_uv_lock_coverage(root / "uv.lock", dependencies))

    if appimage is not None:
        if not appimage.is_file():
            errors.append(f"AppImage not found: {appimage}")
        else:
            with tempfile.TemporaryDirectory(prefix="verbix-license-") as tmp:
                extracted = extract_appimage(appimage, Path(tmp))
                errors.extend(
                    verify_extracted_licenses(
                        extracted,
                        root=root,
                        expected_license_files=required_license_files(dependencies),
                    )
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of scripts/)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="regenerate the generated inventory section in THIRD-PARTY-NOTICES.md",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="read-only verification (default)",
    )
    parser.add_argument(
        "--appimage",
        type=Path,
        default=None,
        help="also verify legal files inside this built AppImage",
    )
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help=(
            "with --appimage, verify only the files inside the artifact; skip "
            "the dependency and voice-risk gate"
        ),
    )
    parser.add_argument("--cargo", default="cargo", help="cargo executable")
    args = parser.parse_args(argv)

    if args.artifact_only and args.appimage is None:
        parser.error("--artifact-only requires --appimage")

    root = args.root.resolve()
    try:
        errors = run_gate(
            root,
            write=args.write,
            appimage=args.appimage.resolve() if args.appimage else None,
            cargo=args.cargo,
            artifact_only=args.artifact_only,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError) as error:
        print(f"license gate could not run: {error}", file=sys.stderr)
        return 2

    if errors:
        print(f"license gate FAILED with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if any("voice risk acceptance" in error for error in errors):
            print(
                "The voice risk acceptance record is missing or invalid. The gate "
                "stays red until the dated record and pinned license text are "
                "committed.",
                file=sys.stderr,
            )
        return 1

    print("license gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
