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


def test_copy_portaudio_libraries_copies_dependency_closure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    portaudio = source / "libportaudio.so.2.0.0"
    portaudio.write_bytes(b"portaudio")
    asound = source / "libasound.so.2.1.0"
    asound.write_bytes(b"asound")
    pipewire = source / "libpipewire-0.3.so.0.1"
    pipewire.write_bytes(b"pipewire")

    def fake_dependencies(path: Path) -> list[tuple[str, Path]]:
        if path.name.startswith("libportaudio"):
            return [("libasound.so.2", asound)]
        if path.name.startswith("libasound"):
            return [("libpipewire-0.3.so.0", pipewire)]
        return []

    lib_dir = tmp_path / "appdir" / "usr" / "lib"
    lib_dir.mkdir(parents=True)
    actions: list[str] = []

    copied = trim.copy_portaudio_libraries(
        (source,), lib_dir, actions, dependencies=fake_dependencies
    )

    assert {path.name for path in copied} == {
        "libportaudio.so.2",
        "libasound.so.2",
        "libpipewire-0.3.so.0",
    }
    assert (lib_dir / "libportaudio.so.2").read_bytes() == b"portaudio"
    assert (lib_dir / "libasound.so.2").read_bytes() == b"asound"
    assert (lib_dir / "libpipewire-0.3.so.0").read_bytes() == b"pipewire"
    link = lib_dir / "libportaudio.so"
    assert link.is_symlink()
    assert link.readlink() == Path("libportaudio.so.2")


def test_copy_portaudio_libraries_fails_loudly_when_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="cannot bundle libportaudio.so.2"):
        trim.copy_portaudio_libraries(
            (tmp_path / "empty",), tmp_path / "lib", []
        )


def test_ldd_dependencies_drops_host_libraries(monkeypatch) -> None:
    output = (
        "\tlinux-vdso.so.1 (0x00007fff)\n"
        "\tlibasound.so.2 => /lib/x86_64-linux-gnu/libasound.so.2 (0x00007f)\n"
        "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f)\n"
        "\tlibgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f)\n"
        "\t/lib64/ld-linux-x86-64.so.2 (0x00007f)\n"
    )
    monkeypatch.setattr(
        trim.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": output})(),
    )

    assert trim.ldd_dependencies(Path("libportaudio.so.2")) == [
        ("libasound.so.2", Path("/lib/x86_64-linux-gnu/libasound.so.2"))
    ]


def test_ldd_dependencies_fails_on_a_missing_library(monkeypatch) -> None:
    output = "\tlibjack.so.0 => not found\n"
    monkeypatch.setattr(
        trim.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": output})(),
    )

    with pytest.raises(SystemExit, match="missing dependency"):
        trim.ldd_dependencies(Path("libportaudio.so.2"))


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
