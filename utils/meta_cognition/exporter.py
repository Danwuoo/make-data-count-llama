from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

from .schema import ContrastivePromptPair, PromptPairItem


class PairExporter:
    """Export prompt pairs to various file formats."""

    def to_jsonl(
        self,
        items: Iterable[Union[PromptPairItem, ContrastivePromptPair]],
        path: str,
    ) -> None:
        path_obj = Path(path)
        with path_obj.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
