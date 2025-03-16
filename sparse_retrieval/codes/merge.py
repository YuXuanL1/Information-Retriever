import pandas as pd

# train data

# # 讀取檢索結果文件
# def read_run_file(file_path, score_column_name):
#     """
#     讀取 .run 文件並提取必要欄位
#     """
#     cols = ['query_id', 'IndexID', 'doc_id', 'rank', score_column_name, 'method']
#     return pd.read_csv(file_path, delim_whitespace=True, header=None, names=cols)[['query_id', 'doc_id', score_column_name]]

# # 讀取 qrels 文件
# def read_qrels_file(file_path):
#     """
#     讀取 qrels 文件並提取必要欄位
#     """
#     cols = ['query_id', 'IndexID', 'doc_id', 'relevance_label']
#     return pd.read_csv(file_path, delim_whitespace=True, header=None, names=cols)

# # 載入資料
# bm25_data = read_run_file('runs/bm25_1.run', 'bm25_score')
# mle_data = read_run_file('runs/MLE_smoothed_1.run', 'mle_score')
# jm_data = read_run_file('runs/jelinek_mercer_1.run', 'jm_score')
# qrels_data = read_qrels_file('data/qrels.401-440.txt')

# # 合併三種分數
# merged_scores = bm25_data.merge(mle_data, on=['query_id', 'doc_id'], how='outer')
# merged_scores = merged_scores.merge(jm_data, on=['query_id', 'doc_id'], how='outer')

# # 合併相關性標籤
# final_table = merged_scores.merge(qrels_data, on=['query_id', 'doc_id'], how='left')

# # 檢查是否有缺失值並填補
# final_table = final_table.fillna({'bm25_score': 0, 'mle_score': 0, 'jm_score': 0, 'relevance_label': 0})

# # Drop 'Q0' column if it exists
# if 'IndexID' in final_table.columns:
#     final_table = final_table.drop(columns=['IndexID'])

# # 儲存為 CSV 文件
# final_table.to_csv('train_data_table.csv', index=False)
# print(final_table.head())

#####################################################################
#test data
# 讀取檢索結果
bm25_results = pd.read_csv('runs/bm25_test_1.run', sep=' ', names=['query_id', 'IndexID', 'doc_id', 'rank', 'bm25_score', 'method'])
mle_results = pd.read_csv('runs/MLE_smoothed_test_1.run', sep=' ', names=['query_id', 'IndexID', 'doc_id', 'rank', 'mle_score', 'method'])
jm_results = pd.read_csv('runs/jelinek_mercer_test_1.run', sep=' ', names=['query_id', 'IndexID', 'doc_id', 'rank', 'jm_score', 'method'])

# 合併檢索結果
merged_results = bm25_results[['query_id', 'doc_id', 'bm25_score']].merge(
    mle_results[['query_id', 'doc_id', 'mle_score']], on=['query_id', 'doc_id']
).merge(
    jm_results[['query_id', 'doc_id', 'jm_score']], on=['query_id', 'doc_id']
)

# 讀取相關性標籤
qrels = pd.read_csv('data/qrels.441-450.txt', sep=' ', names=['query_id', 'zero', 'doc_id', 'relevance_label'])

# 合併相關性標籤
final_data = merged_results.merge(qrels[['query_id', 'doc_id', 'relevance_label']], on=['query_id', 'doc_id'], how='left')

# 將結果保存為 CSV
final_data.to_csv('test_data_table.csv', index=False)
