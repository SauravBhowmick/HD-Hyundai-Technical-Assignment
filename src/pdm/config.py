"""Configuration loader.

The pipeline is driven by a single YAML config (``configs/default.yaml``).
``load_config`` returns a plain dict so we don't impose a Pydantic schema on
every internal call, but we expose typed accessors for the bits that matter.

Config discovery: we search ``$PDM_CONFIG`` first, then ``configs/default.yaml``
under the current working directory and its ancestors, then under the source
tree (useful when the package is installed editable). This means the project
works both in the dev checkout (``pip install -e .``) and once installed into
``site-packages`` (Docker image, wheel install, etc.) as long as you ``cd``
into a directory that contains ``configs/default.yaml``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_SRC_ROOT_GUESS = Path(__file__).resolve().parents[2]


def _find_default_config() -> Path:
    env = os.environ.get("PDM_CONFIG")
    if env:
        return Path(env)

    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        candidate = parent / "configs" / "default.yaml"
        if candidate.is_file():
            return candidate

    src_candidate = _SRC_ROOT_GUESS / "configs" / "default.yaml"
    if src_candidate.is_file():
        return src_candidate

    return cwd / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file. Falls back to ``configs/default.yaml``."""
    cfg_path = Path(path) if path else _find_default_config()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(cfg_path)
    cfg["_repo_root"] = str(cfg_path.resolve().parent.parent)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path from ``cfg['paths'][key]`` relative to the repo root."""
    rel = cfg["paths"][key]
    p = Path(rel)
    if p.is_absolute():
        return p
    return Path(cfg["_repo_root"]) / p


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create the standard output directories if they're missing."""
    for key in ("artifacts_dir", "plots_dir", "mlruns_dir"):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
