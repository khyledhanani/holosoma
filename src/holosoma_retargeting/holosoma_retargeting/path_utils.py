"""Portable path helpers for the holosoma_retargeting package."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = PACKAGE_ROOT.parent

_LEGACY_PACKAGE_PREFIX = ("src", "holosoma_retargeting", "holosoma_retargeting")


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _relative_variants(path: Path) -> list[Path]:
    variants = [path]
    parts = path.parts
    prefix_len = len(_LEGACY_PACKAGE_PREFIX)
    if len(parts) >= prefix_len and parts[:prefix_len] == _LEGACY_PACKAGE_PREFIX:
        stripped_parts = parts[prefix_len:]
        if stripped_parts:
            variants.append(Path(*stripped_parts))
    return _dedupe_paths(variants)


def _candidate_paths(path: Path) -> list[Path]:
    if path.is_absolute():
        return [path]

    candidates: list[Path] = []
    for rel in _relative_variants(path):
        candidates.append((Path.cwd() / rel).resolve())
        candidates.append((PACKAGE_ROOT / rel).resolve())
        candidates.append((BUNDLE_ROOT / rel).resolve())
    return _dedupe_paths(candidates)


def resolve_portable_path(path_value: str | Path, *, prefer_bundle: bool = False, must_exist: bool = False) -> Path:
    """Resolve a path against cwd, package root, and bundle root.

    This keeps old relative paths working after moving the package directory.
    """
    raw_path = Path(path_value)
    candidates = _candidate_paths(raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback_rel = _relative_variants(raw_path)[-1]
    fallback_base = BUNDLE_ROOT if prefer_bundle else PACKAGE_ROOT
    fallback = (fallback_base / fallback_rel).resolve()

    if must_exist:
        probed = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            f"Could not resolve existing path for {path_value!r}. Probed:\n{probed}"
        )

    return fallback
