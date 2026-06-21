"""SecBrn — local graph second brain for LLMs.

Public API:

    from secbrn import Brain
    brain = Brain.from_env()
    brain.ingest("./my-notes/")
    print(brain.ask("How does retrieval relate to rerankers?").text)
"""

from secbrn.config import Settings, get_settings
from secbrn.models import Document, Chunk
from secbrn.pipeline import Brain

__all__ = ["Brain", "Settings", "get_settings", "Document", "Chunk"]
__version__ = "0.1.0"
