from __future__ import annotations

from typing import List

from .error_storage import ErrorStorageManager
from .schema import ErrorRecord, ErrorType


class ErrorQueryAPI:
    """Convenient accessors for error records."""

    def __init__(self, storage: ErrorStorageManager | None = None) -> None:
        self.storage = storage or ErrorStorageManager()

    def load(self) -> List[ErrorRecord]:
        return self.storage.load()

    def filter_by_category(self, error_type: ErrorType) -> List[ErrorRecord]:
        return self.storage.load(error_type)
