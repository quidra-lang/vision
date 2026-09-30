# Development and release workflow

This is the canonical release procedure for the `vision` Quidra package.

## Permanent branches

- `main` is the latest published stable release source.
- `develop` is the long-lived integration branch for the next release.

Routine work goes directly to `develop`.
Do not use `main` for unreleased development. Do not delete and recreate
`develop` after a release.

## Release identity

A release is valid only when these agree:

- `project.toml` version `X.Y.Z`, equal to the coordinated Core/Math version,
  and the `quidra.package` generated from it;
- immutable tag `vX.Y.Z`;
- the exact `main` commit carrying that manifest;
- the declared `requires.quidra` and package dependency ranges admitting
  same-version Core and Math;
- green CI against Core `vX.Y.Z` and Math `vX.Y.Z`.

Branches are never installation identities.

Vision uses the exact same `MAJOR.MINOR.PATCH` version as Core and Math.
It never chooses a release version independently.

## Releasing

When instructed to release vision, release the library, or follow the release
procedure, execute the complete sequence:

1. Fetch the latest remote `develop` and `main` HEADs. Never work from a remembered SHA.
2. Confirm that all intended work is in `develop`.
3. Use the shared first-party release version selected for Core.
4. Update `project.toml`: set that exact shared version, the tested
   `requires.quidra` range, and any `requires.<package>` ranges. Then run
   `quidra package sync .` to regenerate `quidra.package`; never hand-edit it.
   Core owns the metadata parser/generator; package repositories do not keep a
   second TOML implementation.
5. Ensure the release workflow validates against immutable Core and Math tags
   with exactly the Vision package version. Development compatibility CI may
   additionally test the current Quidra `develop` branches.
6. Run/verify all tests and examples on `develop`. Fix failures there.
7. Merge `develop` into `main` while preserving valid history from both branches.
8. Let the release workflow triggered by the `main` push validate the immutable
   same-version Core/Math tags, rerun Vision tests, refuse tag reuse, and create the
   immutable `vX.Y.Z` tag plus GitHub Release on that exact tested `main`
   commit. Do not pre-create or manually retarget the tag.
9. Verify the workflow succeeded and the tag, GitHub Release, and
   `quidra.package` version all match.
10. Return to `develop` and bring back any release-only change if needed.
    Change Vision's version only when the shared Core first-party version
    advances, then run `quidra package sync .` and push it.
11. Continue ordinary work only on `develop`.

Never force-move, delete/recreate, or reuse a published release tag.

## Core-first ordering

Every Vision release follows the same-version Core and Math releases: publish
Core `vX.Y.Z` first, Math `vX.Y.Z` second, then Vision `vX.Y.Z`. The
release workflow checks those exact immutable dependency tags.


## Domain ownership

A primitive belongs to the package whose semantics it represents. Vision keeps
image transforms, codecs, native kernels, SIMD/CUDA/OpenCV integrations and
their third-party dependencies in this repository. Performance is not a reason
to move Vision semantics into Core. Core may provide only generic tensor,
autograd, device/execution, native-extension, and package-build substrate.
