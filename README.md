# Building IR systems based on the Pyserini Project

## Introduction
In this project I will implement several different retrieval methods, i.e. algorithms that given a user's request (query) and a corpus of documents assign a score to each document according to its relevance to the query. Some of these retrieval methods will be the implementation of the basic retrieval models (e.g. TF-IDF, BM25, Language Models with different Smoothing). In this case, I will need the toolkits of [Pyserini Project](https://github.com/castorini/pyserini), which includes search engines, browser toolbars, text analysis tools, and data resources that support research and development of information retrieval and text mining.
![螢幕擷取畫面 2025-03-16 171226](https://github.com/user-attachments/assets/ad1068a8-22fc-4a69-8b42-d38d5da5b134)


## Document Corpus
[WT2g data](https://drive.google.com/file/d/1DHOHw6wwVF5ZiQ-BSYgn98UEoYOY5Pcf/view) (The collection contains Web documents, with being a 2GB corpus.) ([Here](http://ir.dcs.gla.ac.uk/test_collections/wt10g.html) you can find details about the corpus (WT10g instead), in case you are interested.) 

For the corpus, I build two indexes (1) with stemming (2) without stemming. For stemming, I use the porter (Porter) stemmer. 

## Queries
[Here](https://wm5.nccu.edu.tw/base/10001/course/10028223/content/proj02/topics.401-440.txt) is a set of 40 TREC queries for the corpus, with the standard TREC format having topic title, description and narrative. Documents from the corpus have been judged with respect to their relevance to these queries by NIST assessors.

## Part1: Ranking Functions
Run the set of queries against the WT2g collection, return a ranked list of documents (the top 1000) in a particular format, and the evaluate the ranked lists.

Implement the following variations of a retrieval system and evaluate the result:

- OKAPI BM25
  - I use for the weights OKAPI BM25 TF x IDF where OKAPI BM25 TF = tf/(tf + 0.5 + 1.5 * doclen / avgdoclen). For queries, OKAPI BM25 TF can also be computed in the same way, just use the length of the query to replace doclen.

  - Also note that the definition of OKAPI BM25 TF is tf / tf + k1((1 - b) + b * doclen / avgdoclen). In the above formula, I set k1 = 2 and b = 0.75, to end up with: tf / (tf + 0.5 + 1.5 * doclen / avgdoclen).

- Language modeling, maximum likelihood estimates with Laplace smoothing and estimated probability from corpus, query likelihood.

  - For model estimation use maximum-likelihood with Laplace smoothing and estimated probability from corpus. Use formula (for term i)
![laplace_and_pwc](https://github.com/user-attachments/assets/363273e1-1ff3-4f22-bd91-fadb7db27dad)

     where m = term frequency, n=number of terms in document (doc length) , k=number of unique terms in corpus, t=total terms in corpus, and P(w|C) is the estimated probability from corpus (background probability = cf / terms in the corpus).

- Language modeling, Jelinek-Mercer smoothing using the corpus, 0.8 of the weight attached to the background probability, query likelihood.

  - The formula for Jelinek-Mercer smoothing is,
![jelenik-mercer](https://github.com/user-attachments/assets/4c7ac6f3-b8e7-4fe3-ada6-7a11c1348b6c)

    where P(w|D) is the estimated probability from document (max likelihood = m_i/n) and P(w|C) is the estimated probability from corpus (background probability = cf / terms in the corpus).

## Part2: Learning to Rank
Utilize the retrieval results for the 40 queries in part 1, along with their respective relevance answers, to train a model. Subsequently, evaluate its performance on another set of 10 queries.

- In this part, I ensemble these three scores to improve the ranking.(i.e., Use above three scores as features to train a model.)

- Take 40 queries as training data and the remaining 10 quries as testing data.

- I use machine learning (XGBoost) & deep learning (pytorch) models on this work to improve the model performance.

## Evaluation
**Part 1** :
Run the 40 queries and return at top 1,000 documents for each query. Not returning documents with score equal to zero. If there are only N<1000 documents with non-zero scores then only return these N documents. Save the 40 ranked lists of documents in a single file.

**Part 2** :
Run the test 10 queries and return at top 1,000 documents for each query. Not returning documents with score equal to zero. If there are only N<1000 documents with non-zero scores then only return these N documents. Save the 10 ranked lists of documents in a single file.

To evaluate a single run (i.e. a single file containing 40,000 or 10,000 lines or less), first download the qrel files [qrels.401-440](https://wm5.nccu.edu.tw/base/10001/course/10028223/content/proj02/qrels.401-440.txt) [qrels.441-450](https://wm5.nccu.edu.tw/base/10001/course/10028223/content/proj02/qrels.441-450.txt) you can find the qrel file for the WT2g corpus. Then, you can use the evaluation tool in Pyserini Toolkit or you can download the script of [trec_eval](https://wm5.nccu.edu.tw/base/10001/course/10028223/content/proj02/trec_eval.pl) in demo repo and run:

``` perl trec_eval.pl [-q] qrel_file results_file ```

## Report
For more detail analysis and results you can take a look at this [report](https://github.com/user-attachments/files/19270306/WSM_project2.1.pdf).

