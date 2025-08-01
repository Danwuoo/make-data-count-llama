from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from peft import PeftModel


@dataclass
class LoRAModelManager:
    """Utility to save and load LoRA adapter weights."""

    save_dir: Path

    def save(
        self,
        model: PeftModel,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self.save_dir / version
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        if metadata is not None:
            with open(path / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
        return path

    def load(self, base_model, version: str) -> PeftModel:
        path = self.save_dir / version
        return PeftModel.from_pretrained(base_model, path)
