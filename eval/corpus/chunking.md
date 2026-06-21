# Chunking Strategies

Chunking splits documents into passages before [[Embeddings]] are computed. Chunk size is
a tuned trade-off: chunks that are too large dilute the embedding and bury the relevant
sentence; chunks that are too small fragment ideas and break relationships needed for
extraction.

## Structure-aware splitting

Splitting on natural boundaries — Markdown headings, PDF pages, transcript turns — beats
blind fixed-width windows because it keeps coherent ideas together. Overlap of around
10–15% preserves context across boundaries. Poor chunking hurts both [[Vector Search]]
recall and the [[Knowledge Graphs]] built during extraction.
