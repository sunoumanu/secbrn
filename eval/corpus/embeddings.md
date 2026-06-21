# Embedding Models

An embedding model maps text to a dense vector so semantically similar text lands nearby.
Dimension and model are coupled: nomic-embed-text emits 768 dimensions, while
mxbai-embed-large and bge-large emit 1024. The vector index dimension must match the
model exactly, or search silently breaks.

## Local embeddings

[[Ollama]] serves embedding models locally, keeping data on the machine. Swapping
embedding models changes retrieval quality the most, but requires re-embedding the whole
corpus and recreating the index. Embeddings power [[Vector Search]] and feed the
similarity step of [[Entity Resolution]].
