# Vector-Search-RAG


We run vector search in two stages.

- 1- Offline (indexing): we convert all documents into vectors (arrays of numbers) and store them in an index.
- 2- Online (querying): we convert the user's query into a vector with the same model, then find the closest document vectors by similarity.