./clean.sh

# 前置作業
python3 -m venv myenv
source myenv/bin/activate
pip install pyserini

#############################################################################
# We first convert WT2G files into the jsonl format required by pyserini.   #
# No need this step when Using TrecwebCollection instead of JsonCollection. #
#############################################################################
python sparse_retrieval/codes/jsonl.py
python sparse_retrieval/codes/jsonl_stemmed.py

##################################################################
# Secondly, we can build index for our WT2G corpus(247491 docs). #
# Use TrecwebCollection to build WT2G corpus(246772).            #
##################################################################
./codes/build_index.sh
# ./codes/build_trecweb_index.sh


##########################################################
# Then, search and store result in the trec_eval format. #
##########################################################
python sparse_retrieval/codes/main.py --index indexes/collection --query ./data/topics.401-440.txt --method bm25 --output runs/bm25_1.run
python sparse_retrieval/codes/main.py --index indexes/collection --query ./data/topics.401-440.txt --method MLE_smoothed --output runs/MLE_smoothed_1.run
python sparse_retrieval/codes/main.py --index indexes/collection --query ./data/topics.401-440.txt --method jelinek_mercer --output runs/jelinek_mercer_1.run

python sparse_retrieval/codes/main.py --index indexes/collection_stemmed --query ./data/topics.401-440.txt --method bm25 --output runs/bm25_stemmed_1.run
python sparse_retrieval/codes/main.py --index indexes/collection_stemmed --query ./data/topics.401-440.txt --method MLE_smoothed --output runs/MLE_smoothed_stemmed_1.run
python sparse_retrieval/codes/main.py --index indexes/collection_stemmed --query ./data/topics.401-440.txt --method jelinek_mercer --output runs/jelinek_mercer_stemmed_1.run

python sparse_retrieval/codes/main.py --index indexes/collection --query ./data/topics.441-450.txt --method bm25 --output runs/bm25_test_1.run
python sparse_retrieval/codes/main.py --index indexes/collection --query ./data/topics.441-450.txt --method MLE_smoothed --output runs/MLE_smoothed_test_1.run
python sparse_retrieval/codes/main.py --index indexes/collection --query ./data/topics.441-450.txt --method jelinek_mercer --output runs/jelinek_mercer_test_1.run


##############################################
# Merge 3 ranking scores and relevance label. #
##############################################
python3 sparse_retrieval/codes/merge.py

##############################################
# Train learning to rank model and predict relevance. #
##############################################
python3 sparse_retrieval/codes/LTR_model_1.py
python3 sparse_retrieval/codes/LTR_model_2.py


##############################
# Lastly, do the evaluation. #
##############################
# Part1: 3 (ranking functions) * 2 (with & without stemming) = 6 files
perl sparse_retrieval/trec_eval.pl ./data/qrels.401-440.txt runs/bm25_1.run
perl sparse_retrieval/trec_eval.pl ./data/qrels.401-440.txt runs/bm25_stemmed_1.run

perl sparse_retrieval/trec_eval.pl ./data/qrels.401-440.txt runs/MLE_smoothed_1.run
perl sparse_retrieval/trec_eval.pl ./data/qrels.401-440.txt runs/MLE_smoothed_stemmed_1.run

perl sparse_retrieval/trec_eval.pl ./data/qrels.401-440.txt runs/jelinek_mercer_1.run
perl sparse_retrieval/trec_eval.pl ./data/qrels.401-440.txt runs/jelinek_mercer_stemmed_1.run

# Part2: 3 (ranking functions) + 1 (learning to rank) = 4 files
perl sparse_retrieval/trec_eval.pl ./data/qrels.441-450.txt runs/bm25_test_1.run

perl sparse_retrieval/trec_eval.pl ./data/qrels.441-450.txt runs/MLE_smoothed_test_1.run

perl sparse_retrieval/trec_eval.pl ./data/qrels.441-450.txt runs/jelinek_mercer_test_1.run

perl sparse_retrieval/trec_eval.pl ./data/qrels.441-450.txt runs/ltr_test_1.run
perl sparse_retrieval/trec_eval.pl ./data/qrels.441-450.txt runs/ltr_test_2.run

