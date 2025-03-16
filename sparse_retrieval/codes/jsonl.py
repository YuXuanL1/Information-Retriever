import os
import json
import re
from tqdm import tqdm

WT2G_path = "/mnt/c/Users/6yx/Downloads/WSM_project2/data/WT2G"

# [WT01, WT02, ...]
corpus_dir = os.listdir(WT2G_path)

print("Converting WT2G files into jsonl...")
for WTs in tqdm(corpus_dir):
    # [B01, B02, ...]
    corpus_files = os.listdir(os.path.join(WT2G_path, WTs))
    
    for Bs in corpus_files:
        # read WT2G corpus
        with open(os.path.join(WT2G_path, WTs, Bs), 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            # [WT01-B01-1, WT01-B01-2, ...]
            pattern = '<DOCNO>(.*?)</DOCNO>'
            docid = re.findall(pattern, ''.join(lines))

            index = 0
            documents = []
            contents = ''
            for line in lines:
                if '<DOC>' in line:
                    contents = ''
                    continue
                contents = contents + ' ' + line
                if '</DOC>' in line:
                    contents = re.sub(r' {2,}', ' ', contents)
                    documents.append({'id': docid[index], 'contents': contents})
                    index += 1

        output_dir = 'data/collection'
        os.makedirs(output_dir, exist_ok=True)

        # write jsonl file
        with open('data/collection/collection.jsonl', 'a', encoding='utf-8') as f:
            for d in documents:
                f.write(json.dumps(d) + '\n')
