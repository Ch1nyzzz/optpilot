"""Skill registry — maps FM group IDs to Skill classes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optpilot.skills.base import BaseSkill

_SKILL_REGISTRY: dict[str, type[BaseSkill]] = {}


def register_skill(cls: type[BaseSkill]) -> type[BaseSkill]:
    """Decorator: register a Skill class by its FM_GROUP."""
    _SKILL_REGISTRY[cls.FM_GROUP] = cls
    return cls


def get_skill(fm_group: str) -> BaseSkill:
    """Instantiate the skill for the given FM group."""
    cls = _SKILL_REGISTRY.get(fm_group)
    if cls is None:
        raise KeyError(f"No skill registered for FM group {fm_group}")
    return cls()


def load_evolved_skill(fm_group: str, path: Path) -> BaseSkill:
    """Dynamically load an evolved skill from a Python file."""
    spec = importlib.util.spec_from_file_location(f"evolved_skill_{fm_group}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load skill from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the registered skill class for this group
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and hasattr(attr, "FM_GROUP")
            and attr.FM_GROUP == fm_group
        ):
            _SKILL_REGISTRY[fm_group] = attr
            return attr()

    raise ImportError(f"No skill with FM_GROUP={fm_group} found in {path}")
