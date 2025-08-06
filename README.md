# Building IR systems based on the Pyserini Project

## Introduction
In this project I implement several different retrieval methods, i.e. algorithms that given a user's request (query) and a corpus of documents assign a score to each document according to its relevance to the query. Some of these retrieval methods will be the implementation of the basic retrieval models (e.g. TF-IDF, BM25, Language Models with different Smoothing). In this case, I use the toolkits of [Pyserini Project](https://github.com/castorini/pyserini), which includes search engines, browser toolbars, text analysis tools, and data resources that support research and development of information retrieval and text mining.
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

## Results
### Part1: 3 (ranking functions) * 2 (with & without stemming) = 6 files
![螢幕擷取畫面 2025-03-16 172257](https://github.com/user-attachments/assets/9fa818d7-2b88-480a-8e50-ee05999c6178)
- In the low recall region, the curves of BM25, BM25 stemmed, and
Jelinek-Mercer are significantly higher than the others, performing better. As
recall increases, the precision differences become smaller.
- Jelinek-Mercer shows stable performance in the high recall region, with a
slower decline. There is a noticeable performance difference between
Jelinek-Mercer and Jelinek-Mercer stemmed.
- MLE with Laplace Smoothing performs poorly regardless of stemming.

#### Discussion
1. Impact of Stemming
    - Loss of specificity: Stemming might merge terms with different meanings and cause
    the precision to be lower. This is evident in all ranking functions with stemming,
    where the precision is lower across various K values due to possibly overgeneralized
    matching.
    - Small performance gains: In many cases (e.g., Jelinek-Mercer Stemmed vs.
    Non-stemmed), stemming offers little improvement, suggesting it may not always be
    beneficial for complex tasks.

2. Compare between three ranking functions
    - OKAPI BM25: BM25 effectively normalizes term frequency, often outperforming
    the other models due to its handling of term saturation and document length
    normalization.
    - MLE with Laplace Smoothing: Performs poorly in recall-precision averages and
    precision at K. The likely reason is that MLE assumes uniform distribution, which
    doesn't account for variability in query or document distributions. While
    computationally simpler, this simplicity may limit its retrieval effectiveness.
    - Jelinek-Mercer Smoothing: Performs better than MLE with Laplace Smoothing in
    all metrics, offering a balanced approach by interpolating between document-level
    and collection-level probabilities. This flexibility makes it effective in noisy or sparse
    datasets.

#### Conclusion
When it comes to optimizing smoothing, prefer Jelinek-Mercer over MLE with Laplace
Smoothing for smoother and more adaptive query modeling. Jelinek-Mercer smooths
probability distributions effectively, but BM25 remains more robust for ranking relevance in
diverse datasets.

-------------------------------------

### Part2: 3 (ranking functions) + 1 (learning to rank) = 4 files
![螢幕擷取畫面 2025-03-16 172141](https://github.com/user-attachments/assets/69472360-58af-4d34-8c54-26fd135da7d7)
- The curves for BM25 and LTR_model2 (deep learning) are significantly
higher than other models, showing better performance, especially in the low
recall regions.
- MLE Smoothed has weaker overall performance, especially at low recall rates,
where it performs noticeably worse than other models.
- As recall increases, the precision differences between the models become
smaller.

#### Discussion
1. Model Comparison: Deep Learning vs XGBoost
  In this project, the deep learning model outperformed XGBoost, mainly due to the following reasons:
    - Complex Pattern Learning: Deep models can capture non-linear, high-dimensional relationships better than tree-based models, making them more suitable for unstructured text data.
    - Automatic Feature Representation: Unlike XGBoost, which relies on hand-crafted features, deep learning learns hierarchical representations directly from raw data, improving relevance prediction.
    - Scalability: Deep models scale well with large datasets using techniques like SGD and dropout, while XGBoost requires careful tuning and may overfit on complex data.
    - Regularization & Generalization: Built-in techniques such as dropout and batch normalization help deep models avoid overfitting more robustly than traditional boosting.
    - Hyperparameter Tuning: While XGBoost can perform well with optimal settings, deep learning models often benefit more from flexible and automated tuning strategies.

#### Conclusion
Deep learning models excel in semantic understanding and contextual modeling, which are essential for IR tasks. XGBoost remains effective for structured data with well-designed features but is less suited for capturing nuanced text relationships.

For more detail analysis and results you can take a look at this [report](https://github.com/user-attachments/files/19270306/WSM_project2.1.pdf).

