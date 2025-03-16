from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import os
from collections import Counter
import math

# bm25
def bm25(searcher, query, args):
    # 根據 index 的路徑設置 IndexID
    IndexID = 0 if isinstance(args.index, str) and "collection_stemmed" in args.index else 1
    
    output = open(args.output, 'w')
    print(f'Do {args.method} search...')
    
    for qid, qtext in tqdm(query.items()):
        hits = searcher.search(qtext, k=args.k)
        for i in range(len(hits)):
            if hits[i].score != 0:  # 過濾掉分數為零的文檔
                output.write(f'{qid} Q{IndexID} {hits[i].docid.upper()} {i+1} {hits[i].score:.5f} {args.method}\n')
            

#########################################################
# MLE smoothed with laplace
def compute_corpus_stats(searcher):
    """
    Compute corpus statistics:
    - t: total terms in corpus
    - k: number of unique terms
    - term_frequencies: Counter of term frequencies in corpus
    """
    corpus_stats = {
        "t": 0,  # total terms in corpus
        "k": 0,  # number of unique terms
        "term_frequencies": Counter()  # cf for each term
    }

    print("Counting terms in corpus...")
    for doc_id in tqdm(range(searcher.num_docs)):
        doc = searcher.doc(doc_id)
        if not doc:
            continue
            
        doc_text = doc.raw()
        if not doc_text:
            continue

        # Tokenize the document text
        terms = doc_text.split()
        corpus_stats["t"] += len(terms)  # Update total terms
        corpus_stats["term_frequencies"].update(terms)  # Update term frequencies

    # Calculate number of unique terms (k)
    corpus_stats["k"] = len(corpus_stats["term_frequencies"])
    
    return corpus_stats

def calculate_smoothed_probability(mi, n, t, k, cf):
    """
    Calculate smoothed probability using the formula:
    ρi = (mi + 1)/(n + t/k) + ((t-k)/k)/(n + t/k) * (cf/t)
    
    Parameters:
    - mi: frequency of term i in document
    - n: document length (total terms)
    - t: total terms in corpus
    - k: number of unique terms in corpus
    - cf: frequency of term in corpus
    """
    denominator = n + (t/k)
    if denominator == 0:
        return 0
        
    first_term = (mi + 1) / denominator
    second_term = ((t-k)/k) / denominator * (cf/t if t > 0 else 0)
    return first_term + second_term

def query_likelihood_smoothed(doc_id, query_terms, corpus_stats, searcher):
    """
    Calculate Query Likelihood using specified smoothing formula.
    """
    doc = searcher.doc(doc_id)
    if not doc:
        return float('-inf')

    doc_text = doc.raw()
    if not doc_text:
        return float('-inf')

    # Calculate document statistics
    terms = doc_text.split()
    term_freqs = Counter(terms)
    n = len(terms)  # document length
    
    # Get corpus statistics
    t = corpus_stats["t"]  # total terms in corpus
    k = corpus_stats["k"]  # unique terms in corpus
    
    # Calculate total score
    score = 0
    for term in query_terms:
        mi = term_freqs.get(term, 0)  # frequency in document
        cf = corpus_stats["term_frequencies"].get(term, 0)  # frequency in corpus
        
        prob = calculate_smoothed_probability(mi, n, t, k, cf)
        if prob > 0:
            score += math.log(prob)  # Use log for numerical stability

    return score

def mle_smoothed_search(query, searcher, corpus_stats, output_file, args):
    # 根據 index 的路徑設置 IndexID
    IndexID = 0 if isinstance(args.index, str) and "collection_stemmed" in args.index else 1
    
    with open(output_file, 'w') as output:
        print("Processing queries...")
        for qid, qtext in tqdm(query.items()):
            query_terms = qtext.split()
            
            # First get initial hits
            hits = searcher.search(qtext, k=1000)
            
            # Calculate scores using our smoothing formula
            scores = []
            for hit in hits:
                score = query_likelihood_smoothed(hit.docid, query_terms, corpus_stats, searcher)
                if score != 0:  # 只保留分數大於零的項目
                    scores.append((hit.docid, score))

            # 排序並寫入結果
            scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(scores, 1):
                output.write(f'{qid} Q{IndexID} {docid.upper()} {rank} {score:.5f} MLE_smoothed\n')


#############################################################
# JM smoothing
def jelinek_mercer_smoothing(query_terms, doc_id, corpus_stats, searcher, lambda_param):
    """
    計算 Jelinek-Mercer 平滑的語言模型分數。
    
    Parameters:
    - query_terms: 詞彙列表
    - doc_id: 文檔 ID
    - corpus_stats: 語料庫統計數據（包括 t 和 cf）
    - searcher: 檢索器
    - lambda_param: 平滑參數 \(\lambda\)
    """
    doc = searcher.doc(doc_id)
    if not doc:
        return float('-inf')

    doc_text = doc.raw()
    if not doc_text:
        return float('-inf')

    # 文檔統計
    terms = doc_text.split()
    term_freqs = Counter(terms)
    n = len(terms)  # 文檔長度
    
    # 語料庫統計
    t = corpus_stats["t"]  # 語料庫中的總詞數
    cf_dict = corpus_stats["term_frequencies"]  # 每個詞的頻率

    # 計算查詢詞的總分數
    score = 0
    for term in query_terms:
        mi = term_freqs.get(term, 0)  # 詞在文檔中的頻率
        cf = cf_dict.get(term, 0)  # 詞在語料庫中的頻率
        
        # 估計 P(w|D) 和 P(w|C)
        P_w_D = mi / n if n > 0 else 0
        P_w_C = cf / t if t > 0 else 0
        
        # Jelinek-Mercer 平滑
        prob = lambda_param * P_w_D + (1 - lambda_param) * P_w_C
        if prob > 0:
            score += math.log(prob)  # 對數計算數值穩定性
    
    return score

def jelinek_mercer_search(query, searcher, corpus_stats, output_file, args, lambda_param):
    # 根據 index 的路徑設置 IndexID
    IndexID = 0 if isinstance(args.index, str) and "collection_stemmed" in args.index else 1
    
    with open(output_file, 'w') as output:
        print("Processing queries with Jelinek-Mercer smoothing...")
        for qid, qtext in tqdm(query.items()):
            query_terms = qtext.split()
            
            # 初步檢索
            hits = searcher.search(qtext, k=1000)
            
            # 計算 Jelinek-Mercer 分數
            scores = []
            for hit in hits:
                score = jelinek_mercer_smoothing(query_terms, hit.docid, corpus_stats, searcher, lambda_param)
                if score != 0:  # 只保留分數大於零的項目
                    scores.append((hit.docid, score))

            # 排序並寫入結果
            scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(scores, 1):
                output.write(f'{qid} Q{IndexID} {docid.upper()} {rank} {score:.5f} Jelinek_Mercer\n')
