import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# 加載數據
train_data = pd.read_csv('train_data_table.csv')  # 假設包含 query_id, doc_id, bm25_score, mle_score, jm_score, relevance_label
test_data = pd.read_csv('test_data_table.csv')

X_train = train_data[['bm25_score', 'mle_score', 'jm_score']]
y_train = train_data['relevance_label']
X_test = test_data[['bm25_score', 'mle_score', 'jm_score']]
y_test = test_data['relevance_label']

# 標準化數據
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 轉換為 PyTorch Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# 定義模型
class RelevanceModel(nn.Module):
    def __init__(self):
        super(RelevanceModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 避免過擬合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# 初始化模型、損失函數和優化器
model = RelevanceModel()

class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
weights = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.BCELoss(weight=weights[y_train_tensor.long()])
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 訓練模型
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 測試模型
model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_tensor).numpy().flatten()  # 模型輸出為概率
    y_pred = (y_pred_proba > 0.5).astype(int)  # 將概率轉換為二進制分類標籤

# 添加 predict_relevance 欄位
merged_results = test_data.copy()
merged_results['predict_relevance'] = y_pred_proba  # 使用概率進行排序

# 按 query_id 和 predict_relevance 降序排序
merged_results = merged_results.sort_values(by=['query_id', 'predict_relevance'], ascending=[True, False])

# 為每個 query_id 添加排名
merged_results['rank'] = merged_results.groupby('query_id').cumcount() + 1

# 過濾掉分數為 0 的文檔
merged_results = merged_results[merged_results['predict_relevance'] != 0]

# 確保每個 query_id 最多返回 1000 個文檔
merged_results = merged_results[merged_results['rank'] <= 1000]

# 指定輸出文件名
output_file = 'runs/ltr_test_2.run'

# 生成 TREC 格式內容
with open(output_file, 'w') as f:
    for _, row in merged_results.iterrows():
        # 按格式輸出
        f.write(f"{row['query_id']} Q0 {row['doc_id']} {row['rank']} {row['predict_relevance']:.5f} Exp\n")

# 保存為 CSV（如果需要查看結果）
merged_results.to_csv('predicted_test_data_table_2.csv', index=False)

print(f"Results saved to {output_file} and predicted_test_data_table.csv")


