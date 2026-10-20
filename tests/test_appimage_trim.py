from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_trim_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "trim_appimage.py"
    spec = importlib.util.spec_from_file_location("trim_appimage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trim = _load_trim_module()


def test_portaudio_exports_point_at_the_appdir() -> None:
    lines = "\n".join(trim.portaudio_hook_exports())
    assert 'VERBIX_PORTAUDIO_LIB="$APPDIR/usr/lib/libportaudio.so.2"' in lines
    assert 'LD_LIBRARY_PATH="$APPDIR/usr/lib' in lines


def test_copy_portaudio_libraries_copies_sonames_and_linker_alias(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for soname in trim.PORTAUDIO_LIBRARIES:
        (source / f"{soname}.1.2.3").write_bytes(soname.encode())
    lib_dir = tmp_path / "appdir" / "usr" / "lib"
    lib_dir.mkdir(parents=True)
    actions: list[str] = []

    copied = trim.copy_portaudio_libraries((source,), lib_dir, actions)

    assert {path.name for path in copied} == set(trim.PORTAUDIO_LIBRARIES)
    for soname in trim.PORTAUDIO_LIBRARIES:
        assert (lib_dir / soname).read_bytes() == soname.encode()
    link = lib_dir / "libportaudio.so"
    assert link.is_symlink()
    assert link.readlink() == Path("libportaudio.so.2")


def test_copy_portaudio_libraries_fails_loudly_when_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="cannot bundle"):
        trim.copy_portaudio_libraries(
            (tmp_path / "empty",), tmp_path / "lib", []
        )


def test_patch_appdir_environment_is_idempotent(tmp_path: Path) -> None:
    hook = tmp_path / "apprun-hooks" / "linuxdeploy-plugin-gtk.sh"
    hook.parent.mkdir(parents=True)
    hook.write_text("#!/usr/bin/env bash\nexport APPDIR=test\n", encoding="utf-8")
    actions: list[str] = []

    trim.patch_appdir_environment(tmp_path, actions)
    trim.patch_appdir_environment(tmp_path, actions)

    text = hook.read_text(encoding="utf-8")
    assert text.count(trim.PORTAUDIO_HOOK_MARKER) == 1
    assert "VERBIX_PORTAUDIO_LIB" in text
    assert len(actions) == 1


def test_patch_appdir_environment_requires_the_gtk_hook(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="AppRun GTK hook not found"):
        trim.patch_appdir_environment(tmp_path, [])
