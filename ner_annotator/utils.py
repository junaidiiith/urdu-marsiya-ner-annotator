import os
import pandas as pd
import hashlib
import json

from ner_annotator.constants import UPLOAD_DIR



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


def save_file_data(data):
    text_hash = calculate_hash(data['text'])
    with open(f"{UPLOAD_DIR}/{text_hash}.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Test hash:", text_hash)
    print("File saved successfully.")
    
    
def save_ner_tags(text, ner_tags):
    text_hash = calculate_hash(text)
    with open(f"{UPLOAD_DIR}/{text_hash}.json", "r") as f:
        data = json.load(f)
        data["tagged_elements"] = ner_tags
        data["tagged"] = True
    
    with open(f"{UPLOAD_DIR}/{text_hash}.json", "w") as f:
        json.dump(data, f, indent=4)
    
    return data


def save_text_with_hash(text: str):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    text_hash = calculate_hash(text)
    if not os.path.exists(f"{UPLOAD_DIR}/{text_hash}.json"):
        with open(f"{UPLOAD_DIR}/{text_hash}.json", "w") as f:
            data = {
                "text": text,
                "tagged": False,
            }
            json.dump(data, f, indent=4)
        print("Text saved successfully.")
        return data
    else:
        print("Text already exists.")
        with open(f"{UPLOAD_DIR}/{text_hash}.json", "r") as f:
            data = json.load(f)
        print("Data loaded successfully.")
    return data