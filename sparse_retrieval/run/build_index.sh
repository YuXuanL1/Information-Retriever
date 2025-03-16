## without  stemming
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/collection \
  --index indexes/collection \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

## with stemming
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/collection_stemmed \
  --index indexes/collection_stemmed \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
