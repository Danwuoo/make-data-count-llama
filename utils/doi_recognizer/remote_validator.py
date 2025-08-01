"""Optional remote validation for identifiers."""

from __future__ import annotations

class RemoteValidator:
    """Validate identifiers using remote APIs (placeholder)."""

    def validate(self, id_value: str, id_type: str) -> bool:
        # In production this would query Crossref or Entrez.
        # Here we simply accept all identifiers.
        return True
