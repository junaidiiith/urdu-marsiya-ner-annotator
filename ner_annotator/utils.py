import os
import pandas as pd
import hashlib
import json



def get_all_files(dataset_dir: str):
    """
    Get all files in the dataset directory.
    """
    if os.path.exists(f"{dataset_dir}/status.csv"):
        df = pd.read_csv(f"{dataset_dir}/status.csv")
        return {r['name']: dict(r) for _, r in df.iterrows()}
        
    
    all_files = dict()
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".pdf.json.txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    file_name = file.replace('-', ' ').replace('.pdf.json.txt', '')
                    all_files[file_name] = {
                        "name": file_name,
                        "content": content,
                        "path": file_path,
                        "tagged": False,
                    }
    # Save the status to a CSV file
    
    df = pd.DataFrame(list(all_files.values()))
    df.to_csv(f"{dataset_dir}/status.csv", index=False)
                    
    return all_files


def update_file_status(file_path: str):
    """
    Update the file status to tagged.
    """
    df = pd.read_csv(f"{file_path}/status.csv")
    df.loc[df['path'] == file_path, 'tagged'] = True
    df.to_csv(f"{file_path}/status.csv", index=False)




def calculate_hash(text: str) -> str:
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex


def get_llm_configs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    llm_configs = json.load(open(f"{current_dir}/llms.json"))
    return llm_configs