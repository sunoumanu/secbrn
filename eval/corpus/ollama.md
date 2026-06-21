# Ollama

Ollama runs language and embedding models locally with a single command, exposing an HTTP
API. It keeps data on the machine, which is the point of a local-first second brain.

## Models and roles

Ollama serves both the [[Embedding Models|embedding model]] (such as nomic-embed-text) and
the generation model (such as llama3.1 or qwen2.5). Extraction can use a stronger, slower
model than answering because it runs offline at ingest time. LlamaIndex uses Ollama for
local inference. vLLM is an alternative to Ollama when GPU throughput matters.
