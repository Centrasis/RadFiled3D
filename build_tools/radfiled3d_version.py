"""Dynamic version provider for scikit-build-core.

The package version is taken from the CI tag instead of being hard-coded, in
this order:

1. ``PKG-INFO`` -- present only when an already-released sdist is being built,
   where the version is fixed. This is what makes `pip install RadFiled3D` work
   on platforms with no prebuilt wheel.
2. A CI **tag** variable, which wins over everything else:
   ``CI_COMMIT_TAG`` (GitLab), ``GITHUB_REF=refs/tags/X.Y.Z`` or
   ``GITHUB_REF_NAME`` with ``GITHUB_REF_TYPE=tag`` (GitHub Actions).
3. A generic CI ref name that happens to look like a release
   (``CI_COMMIT_REF_NAME``, the last segment of ``GITHUB_REF``).
4. ``0.0.0`` for a plain local/branch build -- as the previous ``setup.py`` did.

This project's own release builds all start from a git checkout, which carries no
PKG-INFO, so in the CI pipeline the tag always decides the version.

A leading ``v`` is stripped and PEP 440 suffixes (``1.2.3rc1``) are accepted.
This module is loaded by scikit-build-core via the top-level
``[[tool.dynamic-metadata]]`` table in pyproject.toml.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["dynamic_metadata"]


# "X.Y.Z" with an optional leading "v" and an optional PEP 440 suffix
# (1.2.3rc1, 1.2.3.post1, 1.2.3.dev4, ...). Anything else is a non-release build.
_VERSION_RE = re.compile(
    r"^v?(\d+\.\d+\.\d+(?:[.-]?(?:a|b|c|rc|alpha|beta|post|rev|r|dev)\.?\d*)?)$",
    re.IGNORECASE,
)


def _tag_candidates() -> list[str]:
    """Version candidates in priority order.

    A CI *tag* variable always wins over a branch/ref name, so that a release
    build is never mislabelled 0.0.0 just because the CI also exports a branch
    variable. GitHub exposes the tag either as ``GITHUB_REF=refs/tags/X.Y.Z`` or
    as ``GITHUB_REF_NAME`` together with ``GITHUB_REF_TYPE=tag``.
    """
    env = os.environ
    candidates: list[str] = []

    # 1. Explicit tag variables.
    if env.get("CI_COMMIT_TAG"):  # GitLab CI tag
        candidates.append(env["CI_COMMIT_TAG"])
    github_ref = env.get("GITHUB_REF", "")
    if github_ref.startswith("refs/tags/"):  # GitHub Actions tag ref
        candidates.append(github_ref[len("refs/tags/") :])
    if env.get("GITHUB_REF_TYPE") == "tag" and env.get("GITHUB_REF_NAME"):
        candidates.append(env["GITHUB_REF_NAME"])

    # 2. Generic ref names, which may or may not describe a release.
    if env.get("CI_COMMIT_REF_NAME"):  # GitLab CI branch or tag
        candidates.append(env["CI_COMMIT_REF_NAME"])
    if github_ref:
        candidates.append(github_ref.rsplit("/", 1)[-1])

    return candidates


def _sdist_version() -> str | None:
    """The version recorded in PKG-INFO when building from an unpacked sdist.

    Without this, a source install off PyPI resolves to 0.0.0 on the user's
    machine (no CI tag variable there), pip sees a version inconsistent with the
    one it requested and discards the sdist:

        Requested RadFiled3D==X.Y.Z ... has inconsistent version:
        expected 'X.Y.Z', but metadata has '0.0.0'

    which makes `pip install RadFiled3D` fail on every platform that has no
    prebuilt wheel instead of compiling the module on the fly.
    """
    candidates = [Path.cwd() / "PKG-INFO", Path(__file__).resolve().parent.parent / "PKG-INFO"]
    for pkg_info in candidates:
        try:
            content = pkg_info.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # PKG-INFO is a simple "Header: value" block terminated by a blank line.
        for line in content.splitlines():
            if not line.strip():
                break
            name, separator, value = line.partition(":")
            if separator and name.strip().lower() == "version":
                match = _VERSION_RE.match(value.strip())
                if match is not None:
                    return match.group(1)
    return None


def _resolve_version() -> str:
    # PKG-INFO exists only when building an already-released sdist, whose version
    # is fixed and must not be relabelled -- a *foreign* tag would otherwise win
    # and pip would reject the very sdist it asked for:
    #
    #   pip install RadFiled3D==1.4.0   # inside a CI checked out at tag v2.3.0
    #   -> has inconsistent version: expected '1.4.0', but metadata has '2.3.0'
    #
    # This does not weaken tag handling for this project's own pipeline: every
    # build it runs (sdist, cibuildwheel, `pip install .`) starts from a git
    # checkout, which has no PKG-INFO, so the CI tag governs there.
    sdist_version = _sdist_version()
    if sdist_version is not None:
        return sdist_version

    for candidate in _tag_candidates():
        match = _VERSION_RE.match(candidate.strip())
        if match is not None:
            return match.group(1)
    return "0.0.0"


def dynamic_metadata(
    settings: Mapping[str, Any] | None = None,
    project: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """dynamic-metadata 0.3 provider hook.

    ``settings`` is the ``[[tool.dynamic-metadata]]`` entry minus ``provider``
    (so it carries ``field = "version"``); the return value is the metadata
    fragment merged into the project table.
    """
    field = (settings or {}).get("field", "version")
    if field != "version":
        msg = f"This provider only supports the 'version' field, got {field!r}"
        raise ValueError(msg)
    return {"version": _resolve_version()}
