import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

# 設定資料夾路徑
data_dir = "/mnt/c/Users/6yx/Downloads/WSM_project2/data/WT2G"
output_dir = "/mnt/c/Users/6yx/Downloads/WSM_project2/stemmed_output"

# 初始化 PorterStemmer 和 Counter
stemmer = PorterStemmer()
stemmed_terms_counter = Counter()

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

# 遍歷所有子資料夾和檔案
for root, dirs, files in os.walk(data_dir):
    for file in files:
            file_path = os.path.join(root, file)
            # 讀取檔案內容時忽略無法解碼的字符
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                # 分詞
                tokens = word_tokenize(content)

                # 詞幹提取
                stemmed_tokens = [stemmer.stem(word) for word in tokens]

                # 更新詞幹計數
                stemmed_terms_counter.update(stemmed_tokens)

                # 儲存處理後的文檔
                relative_path = os.path.relpath(file_path, data_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(' '.join(stemmed_tokens))

# 計算 unique_terms 的個數
unique_terms_count = len(stemmed_terms_counter)
# print(f"詞幹處理後的 unique_terms 個數：{unique_terms_count}")
