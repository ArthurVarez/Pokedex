import operator
import typing as t
from functools import reduce
from pathlib import Path

import yaml


Default = object()

_CONFIG: t.Dict[str, t.Any] = {}


def load(config_path: Path):
    global _CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Unknown config file {config_path}")

    with open(config_path) as fd:
        _CONFIG = yaml.safe_load(fd)


def get(*keys, default: t.Any = Default, cls: t.Type = None) -> t.Any:
    try:
        value = reduce(operator.getitem, keys, _CONFIG)
        if cls is not None:
            return cls(value)
        return value
    except KeyError:
        if default is not Default:
            return default
        raise KeyError(f"Unknown sequence of config keys {keys}")


def has(*keys) -> bool:
    try:
        reduce(operator.getitem, keys, _CONFIG)
        return True
    except KeyError:
        return False
