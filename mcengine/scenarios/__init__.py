"""Return generators. ``generate(cfg)`` dispatches on ``cfg.generator``."""

from __future__ import annotations

from ..config import SimConfig
from .base import ScenarioSet


def generate(cfg: SimConfig) -> ScenarioSet:
    if cfg.generator == "bootstrap":
        from .bootstrap import generate_bootstrap

        return generate_bootstrap(cfg)
    from .parametric import generate_parametric

    return generate_parametric(cfg)


__all__ = ["ScenarioSet", "generate"]
