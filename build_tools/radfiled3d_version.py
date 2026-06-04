"""Dynamic version provider for scikit-build-core.

Reproduces the historical RadFiled3D versioning rule so that the package version
is taken from the CI tag / git ref instead of being hard-coded:

* GitHub Actions:  ``GITHUB_REF`` (e.g. ``refs/tags/1.2.3`` -> ``1.2.3``)
* GitLab CI:       ``CI_COMMIT_TAG`` then ``CI_COMMIT_REF_NAME``

If none of those describe a release (e.g. a plain branch push), the version
falls back to ``0.0.0`` -- identical to the behaviour of the previous
``setup.py``. This module is loaded by scikit-build-core via
``[tool.scikit-build.metadata.version]`` (``provider`` / ``provider-path``).
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

__all__ = ["dynamic_metadata"]


def _resolve_version() -> str:
    version = "0.0.0"
    if os.environ.get("CI_COMMIT_TAG"):  # GitLab CI tag
        version = os.environ["CI_COMMIT_TAG"]
    elif os.environ.get("CI_COMMIT_REF_NAME"):  # GitLab CI branch/ref
        version = os.environ["CI_COMMIT_REF_NAME"]
    elif os.environ.get("GITHUB_REF"):  # GitHub Actions ref, e.g. refs/tags/1.2.3
        version = os.environ["GITHUB_REF"].split("/")[-1]

    # Only accept PEP 440-ish "X.Y.Z" tags; anything else is a non-release build.
    if re.match(r"\d+\.\d+\.\d+", version) is None:
        version = "0.0.0"
    return version


def dynamic_metadata(field: str, settings: Mapping[str, Any] | None = None) -> str:
    if field != "version":
        msg = f"This provider only supports the 'version' field, got {field!r}"
        raise ValueError(msg)
    return _resolve_version()
