from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import argparse

try:
    import yaml  # type: ignore
except ModuleNotFoundError as _yaml_error:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]
else:
    _yaml_error = None


CONFIG_DIR = Path(__file__).resolve().parent


def load_config(
    name: str = "default_simulation",
    *,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Load a YAML configuration and apply optional dotted-path overrides.

    Examples
    --------
    >>> load_config(overrides=['simulation.dt=0.05', 'synapse.plasticity.U=0.1'])
    """
    config_path = _resolve_config_path(name)
    if yaml is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "PyYAML is required to load configuration files. "
            "Install it via 'pip install pyyaml'."
        ) from _yaml_error
    with config_path.open(encoding='utf-8') as handle:
        base_config = yaml.safe_load(handle) or {}

    if not overrides:
        return base_config

    override_dict = parse_overrides(overrides)
    return deep_update(base_config, override_dict)


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """
    Convert key=value strings into a nested dictionary.

    The keys use dotted-path notation, e.g. 'simulation.dt'.
    Values are parsed via YAML to re-use its type coercion rules.
    """
    root: dict[str, Any] = {}
    for raw in pairs:
        if '=' not in raw:
            raise ValueError(f"Override '{raw}' is missing '='.")
        key_path, raw_value = raw.split('=', 1)
        target = root
        *parents, leaf = key_path.split('.')
        for segment in parents:
            target = target.setdefault(segment, {})
            if not isinstance(target, dict):
                raise ValueError(f"Cannot override non-dict path '{key_path}'.")
        target[leaf] = yaml.safe_load(raw_value)
    return root


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge *updates* into *base*, returning a new dictionary.
    """
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.suffix:
        return candidate if candidate.is_file() else CONFIG_DIR / candidate
    return CONFIG_DIR / f'{name}.yaml'


def add_override_arguments(
    parser: argparse.ArgumentParser,
    *,
    config_option: str = "--config",
    overwrite_option: str = "--overwrite",
    config_default: str = "default_simulation",
    overwrite_metavar: str = "path=value",
) -> None:
    """
    Register standard configuration CLI arguments on an argparse parser.

    Parameters
    ----------
    parser:
        Existing ArgumentParser to extend.
    config_option:
        Name of the CLI flag for selecting the base config (default: --config).
    overwrite_option:
        Name of the CLI flag for overrides (default: --overwrite). A short '-O'
        alias is also added automatically.
    config_default:
        Default config name passed to :func:`load_config` when the flag is omitted.
    overwrite_metavar:
        Display name for overwrite arguments in help text.
    """
    parser.add_argument(
        config_option,
        default=config_default,
        help="Config name or path (defaults to '%(default)s').",
    )
    parser.add_argument(
        "-O",
        overwrite_option,
        action="append",
        default=[],
        metavar=overwrite_metavar,
        help="Override a config value using dotted-path notation (may be repeated).",
    )


def load_from_args(
    args: argparse.Namespace,
    *,
    config_attr: str = "config",
    overwrite_attr: str = "overwrite",
    default_config: str = "default_simulation",
) -> dict[str, Any]:
    """
    Convenience wrapper that calls :func:`load_config` using parsed CLI arguments.
    """
    config_name = getattr(args, config_attr, default_config) or default_config
    overrides = getattr(args, overwrite_attr, None)
    return load_config(config_name, overrides=overrides)


__all__ = [
    'load_config',
    'parse_overrides',
    'deep_update',
    'CONFIG_DIR',
    'add_override_arguments',
    'load_from_args',
]
