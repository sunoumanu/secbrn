"""Entity resolution subsystem (Stage 6)."""

from secbrn.resolve.resolver import EntityResolver, MergeDecision, normalize_name, string_similarity

__all__ = ["EntityResolver", "MergeDecision", "normalize_name", "string_similarity"]
