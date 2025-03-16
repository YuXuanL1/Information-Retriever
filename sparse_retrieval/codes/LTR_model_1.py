import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# 加載數據
train_data = pd.read_csv('train_data_table.csv')  # 假設包含 query_id, doc_id, bm25_score, mle_score, jm_score, relevance_label
test_data = pd.read_csv('test_data_table.csv')

X_train = train_data[['bm25_score', 'mle_score', 'jm_score']]
y_train = train_data['relevance_label']
X_test = test_data[['bm25_score', 'mle_score', 'jm_score']]
y_test = test_data['relevance_label']

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練 XGBoost 模型
model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# 進行預測
y_pred = model.predict_proba(X_test_scaled)[:, 1]


# 添加 predict_relevance 欄位
merged_results = test_data
merged_results['predict_relevance'] = y_pred

# 將結果保存為 CSV
# merged_results.to_csv('predicted_test_data_table.csv', index=False)

# 按 query-id 和分數降序排序
merged_results = merged_results.sort_values(by=['query_id', 'predict_relevance'], ascending=[True, False])

# 為每個 query-id 添加排名
merged_results['rank'] = merged_results.groupby('query_id').cumcount() + 1

# 過濾掉分數為 0 的文檔
merged_results = merged_results[merged_results['predict_relevance'] != 0]

# 確保每個 query-id 最多返回 1000 個文檔
merged_results = merged_results[merged_results['rank'] <= 1000]

# 指定輸出文件名
output_file = 'runs/ltr_test_1.run'

# 生成 TREC 格式內容
with open(output_file, 'w') as f:
    for _, row in merged_results.iterrows():
        # 按格式輸出
        f.write(f"{row['query_id']} Q0 {row['doc_id']} {row['rank']} {row['predict_relevance']:.5f} Exp\n")

