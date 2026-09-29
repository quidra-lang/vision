#!/usr/bin/env python3
"""Regenerate every file that derives a value from project.toml.

Run it after editing project.toml:

    python3 scripts/sync_metadata.py

`--check` reports drift and exits non-zero instead of writing, which is what
CI uses to prove quidra.package was not hand-edited.

quidra.package is parsed by an already-released Quidra compiler whose key set
is closed, so this package emits only the compatibility-safe keys derived from
project.toml. Everything else that project.toml declares - the distribution
name, the display name and the ABI requirement - stays out of the generated
file on purpose.
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


def render_package(project: dict) -> str:
    package = project["package"]
    return "\n".join(
        (
            f"name = {package['import']}",
            f"version = {package['version']}",
            f"repository = {package['repository']}",
            f"requires.quidra = {project['requires']['quidra']}",
        )
    ) + "\n"


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
