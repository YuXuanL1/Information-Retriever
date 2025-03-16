import os
import json
import re
from tqdm import tqdm

# Define file paths
WT2G_path = "/mnt/c/Users/6yx/Downloads/WSM_project2/data/stemmed_output"
output_dir = 'data/collection_stemmed'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'collection_stemmed.jsonl')

# Clear output file
with open(output_file, 'w', encoding='utf-8') as f:
    pass

print("Converting WT2G files into jsonl...")

# Process each WT directory
for WTs in tqdm(os.listdir(WT2G_path)):
    corpus_files = os.listdir(os.path.join(WT2G_path, WTs))
    
    for Bs in corpus_files:
        # Read WT2G file
        with open(os.path.join(WT2G_path, WTs, Bs), 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Split into individual documents
        documents = []
        doc_parts = content.split('< doc >')
        
        for part in doc_parts[1:]:  # Skip first empty part
            try:
                # Extract document number
                docno_match = re.search(r'< docno >(.*?)< /docno >', part)
                if not docno_match:
                    continue
                doc_id = docno_match.group(1).strip()
                
                # Remove document header section
                content_part = part.split('< /dochdr >')[1] if '< /dochdr >' in part else part
                
                # Clean up content
                # Remove HTML tags
                content_cleaned = re.sub(r'<[^>]+>', ' ', content_part)
                # Remove document end tag
                content_cleaned = content_cleaned.split('< /doc >')[0]
                # Clean extra whitespace
                content_cleaned = ' '.join(content_cleaned.split())
                
                if doc_id and content_cleaned:
                    documents.append({
                        'id': doc_id,
                        'contents': content_cleaned
                    })
            
            except Exception as e:
                print(f"Error processing document in {WTs}/{Bs}: {str(e)}")
                continue
        
        # Write results to JSONL
        if documents:
            with open(output_file, 'a', encoding='utf-8') as f_out:
                for doc in documents:
                    f_out.write(json.dumps(doc) + '\n')