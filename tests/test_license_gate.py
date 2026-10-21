from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_licenses as gate

REPO_ROOT = Path(__file__).resolve().parents[1]


def dependency(
    ecosystem: str,
    name: str,
    version: str,
    license_text: str,
) -> gate.Dependency:
    return gate.Dependency(
        ecosystem=ecosystem,
        name=name,
        version=version,
        license=license_text,
    )


def notices_for(
    dependencies: list[gate.Dependency],
    *,
    risk: str = "ACCEPTED",
    extra_curated: str = "",
) -> str:
    records = "\n".join(f"- {record}: recorded" for record in gate.REQUIRED_CURATED_RECORDS)
    return (
        "# Third-Party Notices\n\n"
        f"{records}\n"
        f"- Preset voice risk acceptance: **{risk}** — {gate.VOICE_RISK_RECORD}\n"
        "- Synthetic audio disclosure: Verbix speaks with a synthetic voice.\n"
        "- Preset identifiers are labels, not names or identities of real people.\n"
        f"{extra_curated}\n\n"
        f"{gate.BEGIN_MARKER}\n"
        f"{gate.render_inventory(dependencies)}"
        f"{gate.END_MARKER}\n"
    )


def write_voice_risk_record(
    root: Path,
    *,
    status: str = "ACCEPTED RESIDUAL RISK",
    date: str = "2026-10-21",
    accepted_by: str = "numairfm (product owner)",
    scope: str = "M1–M5 and F1–F5",
    revisions: tuple[str, str] = gate.VOICE_RISK_REVISIONS,
) -> Path:
    quant_revision, upstream_revision = revisions
    path = root / gate.VOICE_RISK_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Supertonic 3 preset voice risk acceptance\n\n"
        f"Status: {status}\n"
        f"Date: {date}\n"
        f"Accepted by: {accepted_by}\n\n"
        f"Scope: Supertonic 3 preset voice styles {scope}.\n"
        "Model license: BigScience OpenRAIL-M.\n\n"
        "This is a product risk acceptance, not a legal clearance.\n\n"
        "Pinned revisions:\n"
        f"- Quantized package: {quant_revision}\n"
        f"- Upstream model: {upstream_revision}\n",
        encoding="utf-8",
    )
    return path


def write_license_files(root: Path, dependencies: list[gate.Dependency]) -> None:
    licenses = root / "LICENSES"
    licenses.mkdir(parents=True, exist_ok=True)
    for filename in {
        *gate.required_license_files(dependencies),
        *gate.ASSET_LICENSE_FILES.values(),
    }:
        path = licenses / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test text for {filename}\n", encoding="utf-8")

    (licenses / "bigscience-openrail-m.md").write_text(
        (REPO_ROOT / "LICENSES" / "bigscience-openrail-m.md").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    (root / "DISTRIBUTION-TERMS.md").write_text(
        (REPO_ROOT / "DISTRIBUTION-TERMS.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    bundled = root / gate.BUNDLED_TTS_LICENSE
    bundled.parent.mkdir(parents=True, exist_ok=True)
    bundled.write_bytes((REPO_ROOT / gate.BUNDLED_TTS_LICENSE).read_bytes())


def clean_fixture(tmp_path: Path) -> tuple[list[gate.Dependency], str]:
    dependencies = [
        dependency("cargo", "serde", "1.0.0", "MIT OR Apache-2.0"),
        dependency("python", "jinja2", "3.1.6", "BSD-3-Clause"),
    ]
    write_license_files(tmp_path, dependencies)
    write_voice_risk_record(tmp_path)
    return dependencies, notices_for(dependencies)


def test_render_inventory_is_sorted_and_stable():
    dependencies = [
        dependency("python", "jinja2", "3.1.6", "BSD-3-Clause"),
        dependency("cargo", "serde", "1.0.0", "MIT OR Apache-2.0"),
        dependency("cargo", "libc", "0.2.0", "MIT OR Apache-2.0"),
    ]

    first = gate.render_inventory(dependencies)
    second = gate.render_inventory(list(reversed(dependencies)))

    assert first == second
    assert first.splitlines() == sorted(first.splitlines())
    assert "`cargo:libc@0.2.0` — MIT OR Apache-2.0" in first


def test_missing_dependency_fails(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    scanned = [*dependencies, dependency("cargo", "new-crate", "9.9.9", "MIT")]

    errors = gate.evaluate(tmp_path, scanned, notices)

    assert any("cargo:new-crate@9.9.9" in error for error in errors)
    assert any("missing from inventory" in error for error in errors)


def test_extra_inventory_entry_fails(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    notices = notices.replace(
        gate.END_MARKER,
        "- `cargo:stale@0.0.1` — MIT\n" + gate.END_MARKER,
    )

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("not found in scanned dependency set" in error for error in errors)


def test_unresolved_license_fails(tmp_path):
    dependencies = [
        dependency("cargo", "mystery", "1.0.0", "UNRESOLVED"),
    ]
    write_license_files(tmp_path, dependencies)
    notices = notices_for(dependencies)

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("mystery" in error and "UNRESOLVED" in error for error in errors)


@pytest.mark.parametrize("value", ["", "UNKNOWN", "NOASSERTION", "LicenseRef-Unknown"])
def test_missing_or_unknown_license_fails(tmp_path, value):
    dependencies = [dependency("python", "mystery", "1.0.0", value)]
    write_license_files(tmp_path, dependencies)
    notices = notices_for(dependencies)

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("mystery" in error for error in errors)


def test_inventory_drift_is_detected_without_modifying_notices(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    notices_path = tmp_path / "THIRD-PARTY-NOTICES.md"
    notices_path.write_text(notices, encoding="utf-8")
    before = notices_path.read_text(encoding="utf-8")

    changed = [*dependencies, dependency("cargo", "drifted", "2.0.0", "MIT")]
    errors = gate.evaluate(tmp_path, changed, notices_path.read_text(encoding="utf-8"))

    assert any("drifted" in error for error in errors)
    assert notices_path.read_text(encoding="utf-8") == before


def test_clean_pass(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert errors == []


def test_missing_voice_risk_record_fails(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    (tmp_path / gate.VOICE_RISK_RECORD).unlink()

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("voice risk acceptance" in error for error in errors)


def test_voice_risk_record_must_be_accepted_residual_risk(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    write_voice_risk_record(tmp_path, status="CLEARED")

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("voice risk acceptance" in error for error in errors)


def test_voice_risk_record_must_be_dated(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    write_voice_risk_record(tmp_path, date="not-a-date")

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("voice risk acceptance" in error and "Date" in error for error in errors)


def test_voice_risk_record_date_cannot_be_in_the_future(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    write_voice_risk_record(tmp_path, date="2999-01-01")

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any(
        "voice risk acceptance" in error and "future" in error for error in errors
    )


def test_voice_risk_record_must_be_signed(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    write_voice_risk_record(tmp_path, accepted_by="")

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("voice risk acceptance" in error and "Accepted by" in error for error in errors)


def test_voice_risk_record_must_cover_all_presets(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    write_voice_risk_record(tmp_path, scope="M1–M5")

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("voice risk acceptance" in error and "preset" in error for error in errors)


def test_voice_risk_record_must_pin_both_revisions(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    write_voice_risk_record(
        tmp_path,
        revisions=(gate.VOICE_RISK_REVISIONS[0], "0" * 40),
    )

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("voice risk acceptance" in error and "revision" in error for error in errors)


def test_notices_must_reference_the_voice_risk_record(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    notices = notices.replace(gate.VOICE_RISK_RECORD, "docs/legal/other.md")

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("voice risk acceptance" in error for error in errors)


def test_notices_must_mark_the_voice_risk_accepted(tmp_path):
    dependencies, _ = clean_fixture(tmp_path)
    notices = notices_for(dependencies, risk="UNRESOLVED")

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("voice risk acceptance" in error for error in errors)


def test_distribution_terms_must_reproduce_attachment_a(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    terms_path = tmp_path / "DISTRIBUTION-TERMS.md"
    terms = terms_path.read_text(encoding="utf-8")
    terms_path.write_text(
        terms.replace(
            "To provide medical advice and medical results interpretation;",
            "removed;",
        ),
        encoding="utf-8",
    )

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("DISTRIBUTION-TERMS.md" in error and "Attachment A" in error for error in errors)


def test_license_copy_must_reproduce_the_pinned_upstream_text(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    path = tmp_path / "LICENSES" / "bigscience-openrail-m.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "medical results interpretation", "removed", 1
        ),
        encoding="utf-8",
    )

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any(
        "LICENSES/bigscience-openrail-m.md" in error
        and "pinned upstream license text" in error
        for error in errors
    )


def test_bundled_tts_license_text_must_ship(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    (tmp_path / gate.BUNDLED_TTS_LICENSE).unlink()

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any(gate.BUNDLED_TTS_LICENSE in error for error in errors)


def test_bundled_tts_license_text_must_match_the_pinned_license(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    (tmp_path / gate.BUNDLED_TTS_LICENSE).write_text(
        "tampered license text\n", encoding="utf-8"
    )

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any(
        gate.BUNDLED_TTS_LICENSE in error and "does not match" in error
        for error in errors
    )


def test_missing_curated_record_fails(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    notices = notices.replace("- Tauri 2: recorded\n", "")

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("Tauri 2" in error for error in errors)


def test_missing_license_text_fails(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    (tmp_path / "LICENSES" / "mit.md").unlink()

    errors = gate.evaluate(tmp_path, dependencies, notices)

    assert any("mit.md" in error for error in errors)


def test_update_notices_only_replaces_generated_block(tmp_path):
    dependencies, notices = clean_fixture(tmp_path)
    updated_dependencies = [*dependencies, dependency("cargo", "added", "1.0.0", "MIT")]
    write_license_files(tmp_path, updated_dependencies)

    updated = gate.update_notices_text(notices, updated_dependencies)

    assert "cargo:added@1.0.0" in updated
    assert gate.curated_text(updated) == gate.curated_text(notices)
    assert gate.evaluate(tmp_path, updated_dependencies, updated) == []


def test_update_notices_creates_block_when_absent():
    dependencies = [dependency("cargo", "only", "1.0.0", "MIT")]

    updated = gate.update_notices_text("# Third-Party Notices\n", dependencies)

    assert gate.BEGIN_MARKER in updated
    assert gate.END_MARKER in updated
    assert "cargo:only@1.0.0" in updated


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("MIT", "MIT"),
        ("Apache License 2.0", "Apache-2.0"),
        ("Apache licensed, as found in the LICENSE file", "Apache-2.0"),
        ("MIT License\n\nCopyright (c) 2022", "MIT"),
        ("Apache-2.0 AND MIT", "Apache-2.0 AND MIT"),
        ("MIT/Apache-2.0", "MIT OR Apache-2.0"),
        ("MIT OR Apache-2.0", "MIT OR Apache-2.0"),
    ],
)
def test_canonicalize_expression(expression, expected):
    assert gate.canonicalize_expression(expression) == expected


def test_python_metadata_aliases():
    assert (
        gate.python_license(
            {
                "License": "Dual License",
                "Classifier": [
                    "License :: OSI Approved :: BSD License",
                    "License :: OSI Approved :: Apache Software License",
                ],
            }
        )
        == "Apache-2.0 OR BSD-3-Clause"
    )
    assert (
        gate.python_license(
            {
                "Classifier": [
                    "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)"
                ]
            }
        )
        == "LGPL-2.1-or-later"
    )
    assert gate.python_license({"License": "BSD"}) == "BSD-3-Clause"


def test_scan_python_reads_transitive_dist_info(tmp_path):
    site_packages = tmp_path / "site-packages"
    dist_info = site_packages / "transitive_dep-1.2.3.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.4\n"
        "Name: transitive-dep\n"
        "Version: 1.2.3\n"
        "License-Expression: MIT\n",
        encoding="utf-8",
    )

    dependencies = gate.scan_python(site_packages)

    assert dependencies == [
        gate.Dependency("python", "transitive-dep", "1.2.3", "MIT")
    ]


def test_scan_python_excludes_the_project_itself(tmp_path):
    site_packages = tmp_path / "site-packages"
    dist_info = site_packages / "verbix-0.1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: verbix\nVersion: 0.1.0\n",
        encoding="utf-8",
    )

    assert gate.scan_python(site_packages) == []


def test_uv_lock_coverage_rejects_version_drift(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "locked"\n'
        'version = "1.0.0"\n',
        encoding="utf-8",
    )
    dependencies = [dependency("python", "locked", "2.0.0", "MIT")]

    errors = gate.check_uv_lock_coverage(lock, dependencies)

    assert any("locked" in error for error in errors)


def test_uv_lock_coverage_accepts_matching_locked_packages(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "locked"\n'
        'version = "1.0.0"\n',
        encoding="utf-8",
    )
    dependencies = [dependency("python", "locked", "1.0.0", "MIT")]

    assert gate.check_uv_lock_coverage(lock, dependencies) == []


def test_artifact_only_skips_the_release_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "scan_cargo", lambda *args, **kwargs: [])
    monkeypatch.setattr(gate, "scan_python", lambda *args: [])
    monkeypatch.setattr(gate, "find_site_packages", lambda *args: tmp_path)
    monkeypatch.setattr(gate, "extract_appimage", lambda *args: tmp_path)
    seen = {}

    def verify(extracted, *, root, expected_license_files):
        seen["extracted"] = extracted
        seen["expected"] = expected_license_files
        return []

    monkeypatch.setattr(gate, "verify_extracted_licenses", verify)
    appimage = tmp_path / "Verbix.AppImage"
    appimage.write_bytes(b"not an appimage")

    errors = gate.run_gate(tmp_path, appimage=appimage, artifact_only=True)

    assert errors == []
    assert seen == {
        "extracted": tmp_path,
        "expected": sorted(gate.ASSET_LICENSE_FILES.values()),
    }


def test_artifact_only_requires_an_appimage(tmp_path):
    with pytest.raises(ValueError, match="requires an AppImage"):
        gate.run_gate(tmp_path, artifact_only=True)


def test_extracted_appimage_resources_pass(tmp_path):
    (tmp_path / "THIRD-PARTY-NOTICES.md").write_text("notices\n", encoding="utf-8")
    (tmp_path / "DISTRIBUTION-TERMS.md").write_text("terms\n", encoding="utf-8")
    (tmp_path / "LICENSES").mkdir()
    (tmp_path / "LICENSES" / "mit.md").write_text("mit\n", encoding="utf-8")

    extracted = tmp_path / "squashfs-root"
    resources = extracted / "usr" / "lib" / "Verbix" / "licenses"
    resources.mkdir(parents=True)
    for name in ("THIRD-PARTY-NOTICES.md", "DISTRIBUTION-TERMS.md"):
        (resources / name).write_text(
            (tmp_path / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (resources / "LICENSES").mkdir()
    (resources / "LICENSES" / "mit.md").write_text("mit\n", encoding="utf-8")

    errors = gate.verify_extracted_licenses(
        extracted, root=tmp_path, expected_license_files=["mit.md"]
    )

    assert errors == []


def test_extracted_appimage_resources_missing_notice_fails(tmp_path):
    (tmp_path / "THIRD-PARTY-NOTICES.md").write_text("notices\n", encoding="utf-8")
    (tmp_path / "DISTRIBUTION-TERMS.md").write_text("terms\n", encoding="utf-8")
    (tmp_path / "LICENSES").mkdir()
    (tmp_path / "LICENSES" / "mit.md").write_text("mit\n", encoding="utf-8")
    extracted = tmp_path / "squashfs-root"
    (extracted / "usr").mkdir(parents=True)

    errors = gate.verify_extracted_licenses(
        extracted, root=tmp_path, expected_license_files=["mit.md"]
    )

    assert any("THIRD-PARTY-NOTICES.md" in error for error in errors)
    assert any("DISTRIBUTION-TERMS.md" in error for error in errors)
