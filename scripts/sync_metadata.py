#!/usr/bin/env python3
"""Regenerate every file that derives a value from project.toml.

Run it after editing project.toml:

    python3 scripts/sync_metadata.py

`--check` reports drift and exits non-zero instead of writing, which is what
CI uses to prove quidra.package was not hand-edited.

quidra.package is parsed by an already-released Quidra compiler whose key set
is closed, so only the keys below are ever emitted. Everything else that
project.toml declares - the distribution name, the display name, the ABI
requirement - stays out of the generated file on purpose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toml_subset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROJECT_TOML = ROOT / "project.toml"
PACKAGE = ROOT / "quidra.package"

GENERATED_KEYS = ("name", "version", "repository", "requires.quidra")


def package_pairs() -> list[tuple[str, str]]:
    """The current quidra.package as ordered `key = value` pairs."""
    pairs = []
    text = PACKAGE.read_text(encoding="utf-8")
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise SystemExit(f"quidra.package line {number}: expected 'key = value'")
        key = key.strip()
        if key not in GENERATED_KEYS and not key.startswith("asset."):
            raise SystemExit(
                f"quidra.package line {number}: {key} is not a key the Quidra "
                "package parser accepts"
            )
        pairs.append((key, value.strip()))
    return pairs


def package_fields(project: dict) -> dict:
    """The quidra.package keys owned by project.toml."""
    package = project["package"]
    return {
        "name": package["import"],
        "version": package["version"],
        "repository": package["repository"],
        "requires.quidra": project["requires"]["quidra"],
    }


def render_package(project: dict) -> str:
    owned = package_fields(project)
    pairs = package_pairs()
    present = {key for key, _ in pairs}
    missing = [key for key in GENERATED_KEYS if key not in present]
    if missing:
        raise SystemExit(
            f"quidra.package is missing generated keys: {', '.join(missing)}"
        )
    return "".join(f"{key} = {owned.get(key, value)}\n" for key, value in pairs)


TARGETS = ((PACKAGE, render_package),)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit non-zero instead of writing",
    )
    args = parser.parse_args()

    project = toml_subset.load(PROJECT_TOML)
    stale = []
    for path, render in TARGETS:
        rendered = render(project)
        if rendered == path.read_text(encoding="utf-8"):
            continue
        if args.check:
            stale.append(path.relative_to(ROOT))
        else:
            path.write_text(rendered, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}")

    if stale:
        names = ", ".join(str(path) for path in stale)
        print(
            f"{names} disagree with project.toml.\n"
            "Edit project.toml and run: python3 scripts/sync_metadata.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
