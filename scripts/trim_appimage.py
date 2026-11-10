#!/usr/bin/env python3
"""Trim duplicated runtime files from the Verbix AppDir and repack the AppImage.

The trim is deliberately narrow:

* Python interpreter: keep one real binary and symlink the two aliases.
* libpython: keep one real shared library and symlink the ABI alias.
* Remove build-only Python files: headers, pkg-config, ensurepip, pip, idle,
  tkinter/Tcl/Tk, and tests.
* Remove top-level shared libraries that linuxdeploy copied out of wheel
  package directories when the identical file still exists in the wheel
  (``*.libs`` / ``sherpa_onnx/lib`` / ``transcribe_cpp_native/_native``).

WebKitGTK, GTK, AppIndicator, and every system library stay untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

PYTHON_REMOVALS = (
    "include",
    "share",
    "lib/pkgconfig",
    "bin/pip",
    "bin/pip3",
    "bin/pip3.13",
    "bin/idle3",
    "bin/idle3.13",
    "bin/pydoc3",
    "bin/pydoc3.13",
    "bin/2to3",
    "bin/2to3-3.13",
    "bin/python3-config",
    "bin/python3.13-config",
    "lib/python3.13/ensurepip",
    "lib/python3.13/idlelib",
    "lib/python3.13/lib2to3",
    "lib/python3.13/tkinter",
    "lib/python3.13/turtledemo",
    "lib/python3.13/test",
    "lib/python3.13/config-3.13-x86_64-linux-gnu",
    "lib/python3.13/lib-dynload/_tkinter.cpython-313-x86_64-linux-gnu.so",
    "lib/python3.13/site-packages",
    "lib/tcl9.0",
    "lib/tcl9",
    "lib/tk9.0",
    "lib/itcl4.3.8",
    "lib/thread3.0.6",
    "lib/libtcl9.0.so",
    "lib/libtcl9tk9.0.so",
)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    raise AssertionError("unreachable")


def tree_size(path: Path) -> int:
    total = 0
    for root, _directories, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            try:
                total += file_path.lstat().st_size
            except FileNotFoundError:
                continue
    return total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def remove(path: Path, actions: list[str]) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    actions.append(f"removed {path}")


def symlink_relative(path: Path, target: str, actions: list[str]) -> None:
    if path.is_symlink() and os.readlink(path) == target:
        return
    if path.exists() or path.is_symlink():
        path.unlink()
    path.symlink_to(target)
    actions.append(f"symlinked {path} -> {target}")


def trim_python(python_dir: Path, actions: list[str]) -> None:
    bin_dir = python_dir / "bin"
    real_python = bin_dir / "python3.13"
    if not real_python.is_file() or real_python.is_symlink():
        raise SystemExit(f"expected a real interpreter at {real_python}")
    for alias in ("python3", "python"):
        symlink_relative(bin_dir / alias, "python3.13", actions)

    lib_dir = python_dir / "lib"
    real_libpython = lib_dir / "libpython3.13.so.1.0"
    if real_libpython.is_file():
        symlink_relative(
            lib_dir / "libpython3.13.so", "libpython3.13.so.1.0", actions
        )
    # Linker script, not needed at runtime.
    remove(lib_dir / "libpython3.so", actions)

    for relative in PYTHON_REMOVALS:
        remove(python_dir / relative, actions)


def dedupe_top_level_libraries(appdir: Path, actions: list[str]) -> None:
    lib_dir = appdir / "usr" / "lib"
    site_packages = appdir / "usr" / "lib" / "Verbix" / "voice" / "site-packages"
    if not lib_dir.is_dir() or not site_packages.is_dir():
        return

    originals: dict[tuple[int, str], Path] = {}
    for root, _directories, files in os.walk(site_packages):
        for filename in files:
            path = Path(root) / filename
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            key = (size, sha256(path))
            originals.setdefault(key, path)

    for path in sorted(lib_dir.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        key = (path.stat().st_size, sha256(path))
        original = originals.get(key)
        if original is None:
            continue
        path.unlink()
        actions.append(f"removed duplicate {path} (kept {original})")


PORTAUDIO_ROOT_SONAME = "libportaudio.so.2"
PORTAUDIO_SOURCE_ROOTS = (
    Path("/usr/lib64"),
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/lib"),
)
# The host provides glibc and the loader. Everything else PortAudio links is
# copied so the AppImage does not depend on a host PortAudio/JACK/ALSA stack.
PORTAUDIO_HOST_SONAMES = {
    "libc.so.6",
    "libm.so.6",
    "libgcc_s.so.1",
    "libpthread.so.0",
    "libdl.so.2",
    "librt.so.1",
    "libresolv.so.2",
    "libutil.so.1",
    "ld-linux-x86-64.so.2",
    "linux-vdso.so.1",
}
PORTAUDIO_HOOK_MARKER = "# Verbix: bundled PortAudio runtime"
LDD_LINE = re.compile(r"^\s*(\S+) => (\S+) \(0x")
Dependency = tuple[str, Path]
DependencyResolver = Callable[[Path], list[Dependency]]


def ldd_dependencies(path: Path) -> list[Dependency]:
    """Return the resolved shared-library dependencies reported by ``ldd``."""
    result = subprocess.run(
        ["ldd", str(path)], capture_output=True, text=True, check=True
    )
    dependencies: list[Dependency] = []
    for line in result.stdout.splitlines():
        if "not found" in line:
            raise SystemExit(f"missing dependency for {path}: {line.strip()}")
        match = LDD_LINE.match(line)
        if match is None:
            continue
        soname, resolved = match.groups()
        if soname in PORTAUDIO_HOST_SONAMES or "/ld-linux" in resolved:
            continue
        dependencies.append((soname, Path(resolved)))
    return dependencies



def portaudio_hook_exports() -> tuple[str, ...]:
    """The AppRun hook lines that point the shell and child at bundled PortAudio."""
    return (
        PORTAUDIO_HOOK_MARKER,
        'export LD_LIBRARY_PATH="$APPDIR/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        'export VERBIX_PORTAUDIO_LIB="$APPDIR/usr/lib/libportaudio.so.2"',
    )


def find_soname(root: Path, soname: str) -> Path | None:
    """Find a real library file for ``soname`` under ``root``."""
    for candidate in sorted(root.rglob(f"{soname}*")):
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return None


def copy_portaudio_libraries(
    roots: tuple[Path, ...],
    lib_dir: Path,
    actions: list[str],
    dependencies: DependencyResolver = ldd_dependencies,
) -> list[Path]:
    """Copy PortAudio and its resolved runtime dependency closure into the AppDir.

    Fedora's PortAudio links PipeWire's JACK; Ubuntu's links JACK2 and libdb.
    Resolving ``ldd`` on the build host keeps the bundle specific to the binary
    being shipped instead of assuming one distribution's dependency set.
    """
    source = next(
        (
            found
            for root in roots
            if root.is_dir() and (found := find_soname(root, PORTAUDIO_ROOT_SONAME))
        ),
        None,
    )
    if source is None:
        raise SystemExit(
            f"cannot bundle {PORTAUDIO_ROOT_SONAME}: not found under "
            + ", ".join(str(root) for root in roots)
        )

    copied: list[Path] = []
    pending: list[Dependency] = [(PORTAUDIO_ROOT_SONAME, source)]
    seen: set[str] = set()
    while pending:
        soname, resolved = pending.pop(0)
        if soname in seen:
            continue
        seen.add(soname)
        destination = lib_dir / soname
        if destination.exists():
            actions.append(f"kept existing {destination} (PortAudio dependency)")
            continue
        shutil.copy2(resolved, destination)
        actions.append(f"bundled {destination} (from {resolved})")
        copied.append(destination)
        pending.extend(dependencies(resolved))

    symlink_relative(lib_dir / "libportaudio.so", "libportaudio.so.2", actions)
    return copied


def patch_appdir_environment(appdir: Path, actions: list[str]) -> None:
    """Export the bundled PortAudio path before AppRun.wrapped starts."""
    hook = appdir / "apprun-hooks" / "linuxdeploy-plugin-gtk.sh"
    if not hook.is_file():
        raise SystemExit(f"AppRun GTK hook not found: {hook}")
    text = hook.read_text(encoding="utf-8")
    if PORTAUDIO_HOOK_MARKER in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    text += "\n".join(portaudio_hook_exports()) + "\n"
    hook.write_text(text, encoding="utf-8")
    actions.append(f"exported bundled PortAudio in {hook}")


def find_appimage_plugin(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if env := os.environ.get("APPIMAGE_PLUGIN"):
        candidates.append(Path(env))
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    candidates.extend(
        [
            cache / "tauri" / "linuxdeploy-plugin-appimage.AppImage",
            Path.home() / ".cache" / "tauri" / "linuxdeploy-plugin-appimage.AppImage",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "linuxdeploy-plugin-appimage.AppImage not found; set APPIMAGE_PLUGIN"
    )


def repack(
    appdir: Path,
    output: Path,
    plugin: Path,
    actions: list[str],
) -> None:
    output.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "APPIMAGE_EXTRACT_AND_RUN": "1",
            "ARCH": "x86_64",
            "OUTPUT": str(output),
        }
    )
    subprocess.run(
        [
            str(plugin),
            f"--appdir={appdir}",
        ],
        cwd=appdir.parent,
        env=environment,
        check=True,
    )
    if not output.is_file():
        raise SystemExit(f"AppImage plugin did not produce {output}")
    actions.append(f"repacked {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appdir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="final AppImage path (default: sibling of the AppDir, Verbix_<version>_amd64.AppImage)",
    )
    parser.add_argument("--plugin", type=Path, default=None)
    parser.add_argument(
        "--portaudio-dir",
        type=Path,
        default=None,
        help="directory containing libportaudio.so.2 and its runtime libraries",
    )
    parser.add_argument(
        "--no-repack",
        action="store_true",
        help="trim only; leave the existing AppImage untouched",
    )
    args = parser.parse_args(argv)

    appdir = args.appdir.resolve()
    if not appdir.is_dir():
        raise SystemExit(f"AppDir not found: {appdir}")
    python_dir = appdir / "usr" / "lib" / "Verbix" / "voice" / "python"
    if not python_dir.is_dir():
        raise SystemExit(f"bundled Python not found: {python_dir}")

    before = tree_size(appdir)
    actions: list[str] = []
    trim_python(python_dir, actions)
    dedupe_top_level_libraries(appdir, actions)

    portaudio_dir = args.portaudio_dir
    if portaudio_dir is None and os.environ.get("VERBIX_PORTAUDIO_DIR"):
        portaudio_dir = Path(os.environ["VERBIX_PORTAUDIO_DIR"])
    portaudio_roots = (
        (portaudio_dir,) if portaudio_dir is not None else PORTAUDIO_SOURCE_ROOTS
    )
    copy_portaudio_libraries(
        portaudio_roots, appdir / "usr" / "lib", actions
    )
    patch_appdir_environment(appdir, actions)
    after = tree_size(appdir)

    for action in actions:
        print(action)
    print(f"AppDir: {human_size(before)} -> {human_size(after)} ({after - before:+d} bytes)")

    if not args.no_repack:
        output = args.output
        if output is None:
            output = appdir.parent / "Verbix_0.1.0_amd64.AppImage"
        plugin = find_appimage_plugin(args.plugin)
        repack(appdir, output.resolve(), plugin, actions)
        print(f"AppImage: {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
