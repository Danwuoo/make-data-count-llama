from __future__ import annotations

"""Generate semantic-preserving prompt perturbations."""

from dataclasses import dataclass
from typing import List
import random


@dataclass
class GeneratorConfig:
    """Configuration for :class:`PerturbationGenerator`."""

    num_variants: int = 5
    seed: int | None = None


class PerturbationGenerator:
    """Create simple variations of a prompt while preserving intent."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def generate(self, prompt: str) -> List[str]:
        """Return ``num_variants`` perturbations of ``prompt``."""
        variants: List[str] = []
        for i in range(self.config.num_variants):
            strategy = i % 4
            if strategy == 0:  # add polite prefix
                variants.append(f"Please {prompt}")
            elif strategy == 1:  # append courtesy suffix
                variants.append(f"{prompt}, thank you")
            elif strategy == 2:  # insert filler words
                variants.append(f"Um, {prompt}")
            else:  # reorder simple clauses if possible
                parts = prompt.split(",")
                if len(parts) > 1:
                    random.shuffle(parts)
                    variants.append(",".join(p.strip() for p in parts))
                else:
                    variants.append(prompt)
        return variants
