from __future__ import annotations

from typing import Optional

from .error_classifier import ErrorClassifier
from .error_storage import ErrorStorageManager
from .schema import ErrorRecord, ErrorType


class ErrorLogger:
    """High level interface for logging errors to disk."""

    def __init__(
        self,
        classifier: Optional[ErrorClassifier] = None,
        storage: Optional[ErrorStorageManager] = None,
    ) -> None:
        self.classifier = classifier or ErrorClassifier()
        self.storage = storage or ErrorStorageManager()

    def log(self, record: ErrorRecord) -> ErrorRecord:
        """Classify ``record`` if needed and persist it."""

        if record.error_type == ErrorType.OTHER:
            record.error_type = self.classifier.classify(record)
        self.storage.append(record)
        return record

    def load_errors(self, error_type: Optional[ErrorType] = None):
        """Load errors using underlying storage manager."""

        return self.storage.load(error_type)
