"""Data structures for the paper-aligned OmegaPRM generation baseline."""
from dataclasses import dataclass, field
import math
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str = "stop"
    token_count: int = 0


class Completer(Protocol):
    def generate(self, problem: str, prefix: str, n: int, seed: int) -> list[Completion]: ...
    def count_tokens(self, text: str) -> int: ...
    def split_midpoint(self, text: str) -> tuple[str, str] | None: ...


@dataclass
class Rollout:
    text: str
    finish_reason: str
    token_count: int
    correct: bool
    extracted_answer: str | None
    reason: str
    unknown_mapped_to_incorrect: bool = False
    visited: bool = False


@dataclass
class State:
    id: int
    prefix: str
    mc: float
    visits: int = 0
    rollouts: list[Rollout] = field(default_factory=list)
    mc_source: str = "monte_carlo"
    sampling_seed: int | None = None
    in_tree: bool = False


@dataclass(frozen=True)
class SearchConfig:
    filter_rollouts: int = 32
    rollouts: int = 8
    searches: int = 100
    target_parts: int = 16
    alpha: float = 0.5
    beta: float = 0.9
    length_scale: float = 500.0
    c_puct: float = 0.125
    # Optional engineering controls, disabled in the default search algorithm.
    max_calls: int | None = None
    unknown_policy: str = "incorrect"
    incomplete_policy: str = "error"

    def __post_init__(self):
        if self.filter_rollouts < 2 or self.rollouts < 2 or self.searches < 0 or self.target_parts < 1:
            raise ValueError("require filter_rollouts/rollouts >= 2, searches >= 0, target_parts >= 1")
        if self.max_calls is not None and self.max_calls < 1:
            raise ValueError("max_calls must be positive or None")
        if not (0 < self.alpha <= 1 and 0 < self.beta <= 1):
            raise ValueError("alpha and beta must be in (0,1]")
        if not math.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("length_scale must be finite and positive")
        if not math.isfinite(self.c_puct) or self.c_puct < 0:
            raise ValueError("c_puct must be finite and nonnegative")
        if self.unknown_policy not in {"incorrect", "error"}:
            raise ValueError("unknown_policy must be incorrect or error")
        if self.incomplete_policy not in {"incorrect", "error"}:
            raise ValueError("incomplete_policy must be incorrect or error")
