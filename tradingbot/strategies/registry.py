"""TIIM's strategy library.

Built-in strategies live in this package. Your own strategies go in the `my_strategies/` folder at
the top of the project: any .py file there that defines a Strategy subclass is picked up
automatically the next time TIIM starts (see my_strategies/README.md).

Switch strategies off without deleting them in data/strategies.json:
    {"disabled": ["news_fade", "earnings_runup"]}
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
from pathlib import Path

from ..config import ROOT
from .base import Strategy

log = logging.getLogger(__name__)

BUILTIN_MODULES = [
    "tradingbot.strategies.core",
    "tradingbot.strategies.news",
    "tradingbot.strategies.earnings",
]
USER_DIR = ROOT / "my_strategies"


def _classes_in(module) -> list[type[Strategy]]:
    out = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module.__name__
                and not obj.experimental and not inspect.isabstract(obj) and obj.name != "base"):
            out.append(obj)
    return out


def discover(user_dir: Path | None = None) -> tuple[list[type[Strategy]], list[str]]:
    """All strategy classes, built-in first, then yours. Returns (classes, problems)."""
    classes: list[type[Strategy]] = []
    problems: list[str] = []
    for name in BUILTIN_MODULES:
        classes += _classes_in(importlib.import_module(name))
    folder = user_dir or USER_DIR
    if folder.exists():
        for path in sorted(folder.glob("*.py")):
            if path.name.startswith(("_", "TEMPLATE")):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"my_strategies.{path.stem}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                found = _classes_in(mod)
                for cls in found:
                    if cls.author == "TIIM":
                        cls.author = "you"
                classes += found
                if not found:
                    problems.append(f"{path.name}: no Strategy class found")
            except Exception as e:  # a broken user file must never stop TIIM
                problems.append(f"{path.name}: {type(e).__name__}: {e}")
    seen, unique = set(), []
    for cls in classes:
        if cls.name in seen:
            problems.append(f"duplicate strategy name '{cls.name}' ({cls.__module__}) - skipped")
            continue
        seen.add(cls.name)
        unique.append(cls)
    return unique, problems


def disabled(data_dir: Path) -> set[str]:
    path = data_dir / "strategies.json"
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()).get("disabled", []))
    except ValueError:
        log.warning("data/strategies.json is not valid JSON - ignoring it")
        return set()


def load_library(data_dir: Path, user_dir: Path | None = None, journal=None) -> list[Strategy]:
    classes, problems = discover(user_dir)
    off = disabled(data_dir)
    lib = []
    for cls in classes:
        if cls.name in off:
            continue
        try:
            lib.append(cls())
        except Exception as e:
            problems.append(f"{cls.name}: could not start ({e})")
    for p in problems:
        log.warning("strategy library: %s", p)
        if journal is not None:
            journal.event("error", f"strategy library: {p}")
    return lib
