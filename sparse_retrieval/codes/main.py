import argparse
from search import *
from util import *
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/collection_stemmed", type=str, help="Path to the index directory")
    parser.add_argument("--query", default="./data/topics.401-440.txt", type=str, help="Path to the query file")
    parser.add_argument("--method", default="bm25", type=str, 
                      choices=["bm25", "MLE_smoothed", "jelinek_mercer"], 
                      help="Retrieval method")
    parser.add_argument("--k", default=1000, type=int, help="Number of top documents to retrieve")
    parser.add_argument("--output", default='runs/output.run', type=str, help="Output file for results")
    parser.add_argument("--lambda_param", default=0.8, type=float, help="Lambda parameter for Jelinek-Mercer smoothing")
    args = parser.parse_args()

    # Initialize LuceneSearcher
    searcher = LuceneSearcher(args.index)
    
    # Read queries
    query = read_title(args.query)
    
    # Select method
    if args.method == "bm25":
        searcher.set_bm25(k1=2, b=0.75)
        bm25(searcher, query, args)
    
    elif args.method == "MLE_smoothed":
        corpus_stats = compute_corpus_stats(searcher)
        print("Performing MLE smoothed search...")
        mle_smoothed_search(query, searcher, corpus_stats, args.output, args.index)
    
    elif args.method == "jelinek_mercer":
        corpus_stats = compute_corpus_stats(searcher)
        print("Performing Jelinek-Mercer smoothed search...")
        jelinek_mercer_search(query, searcher, corpus_stats, args.output, args, args.lambda_param)
