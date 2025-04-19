from collections import Counter
import io
import os
import pandas as pd
import hashlib
import json

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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


def save_file_data(text, data):
    text_hash = calculate_hash(text)
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


def save_llm_judgement(text: str, judgement_data: str):
    text_hash = calculate_hash(text)
    with open(f"{UPLOAD_DIR}/{text_hash}.json", "r") as f:
        data = json.load(f)
        data["llm_judgement"] = judgement_data
    
    with open(f"{UPLOAD_DIR}/{text_hash}.json", "w") as f:
        json.dump(data, f, indent=4)
    
    return data


def format_llm_response(response: str):
    try:
        response = json.loads(response)
    except json.JSONDecodeError:
        import ast
        try:
            response = ast.literal_eval(response)
            return response
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
    return response


def get_ner_tags_excel(file_name, tagged_elements):
    """
    Convert tagged elements to a DataFrame.
    """
    ner_tags = []
    for item in tagged_elements:
        if "entity_status" in item:
            user_verified = item["entity_status"].get("user_verified", False)
            for entity, status in item["entity_status"].items():
                # print(entity, status)
                if entity == 'user_verified':
                    continue
                ner_tags.append({
                    "Sentence": item['original'],
                    "NER Tagged": item["tagged"],
                    "English": item['english'],
                    "Entity": entity,
                    "LLM-NER-Tag": status["tag"],
                    "Reviewed": user_verified,
                    "Correct": user_verified and status['user_updated'] is None,
                    "User Corrected": status["user_updated"] if status['user_updated'] is not None else "NA",
                })
                
    df = pd.DataFrame(ner_tags)
    df.insert(0, "File Name", file_name.replace('.pdf.json.txt', '.txt'))
    return encode_df(df, "NER Tags")



def get_llm_judgment_excel(file_name, llm_judgement):
    """
    Convert LLM judgement to a DataFrame.
    """
    llm_judgement = [
        {
            "Entity": item["entity"],
            "Original": item["original"],
            "Correct": item["correct"],
            "Alternative": item["alternative"] if item["correct"] else "NA",
            "LLM": item["model"],
        }
        for item in llm_judgement
    ]
    df = pd.DataFrame(llm_judgement)
    df.insert(0, "File Name", file_name.replace('.pdf.json.txt', '.txt'))
    return encode_df(df, "LLM Judgement")


def encode_df(df: pd.DataFrame, sheet_name: str):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)  # reset pointer to the start
    return output


def get_stats(tagged_data_elements):
    def get_true_verified(e):
        return e['tag'] if e['user_updated'] is None else e['user_updated']
    
    def get_predicted_verified(e):
        return e['tag']
    
    entity_status = [d['entity_status'] for d in tagged_data_elements]
    total_entities = [v for es in entity_status for k, v in es.items() if k != 'user_verified']
    per_category_count = dict(Counter([t['user_updated']  if t['user_updated'] else t['tag'] for t in total_entities]))
    # remove none counts
    per_category_count = {k: v for k, v in per_category_count.items() if k is not None}

    total_verified = sum([
            [v for k, v in es.items() if k != 'user_verified'] 
            for es in entity_status if es.get('user_verified', False)
        ], []
    )
    y_t = [get_true_verified(v) for v in total_verified]
    y_p = [get_predicted_verified(v) for v in total_verified]
    print(f"  Accuracy:  {balanced_accuracy_score(y_t, y_p):.3f}")
    print(f"  Precision: {precision_score(y_t, y_p, zero_division=0, labels=list(per_category_count.keys())):.3f}")
    print(f"  Recall:    {recall_score(y_t, y_p, zero_division=0, labels=list(per_category_count.keys())):.3f}")
    print(f"  F1-score:  {f1_score(y_t, y_p, zero_division=0, labels=list(per_category_count.keys())):.3f}\n")

    
    # print("entity status: ", entity_status)
    return {
        'total_entities': len(total_entities),
        'per_category_count': per_category_count,
        'total_verified': len(total_verified),
        **get_classification_metrics(y_t, y_p, list(per_category_count.keys())),
    }


def get_classification_metrics(y_true, y_pred, labels):
    if len(labels) == 0 or len(y_true) == 0 or len(y_pred) == 0:
        return {
            "micro_scores": None,
            "macro_scores": None,
            "weighted_scores": None,
            "df_per_type": None,
        }
    print(f"  Labels: {labels}")
    print(f"  True:   {y_true}")
    print(f"  Pred:   {y_pred}")
    
    
# ─── overall scores ──────────────────────────────────────────────────────────
    overall = {
        'micro': {
            'precision': precision_score(y_true, y_pred, labels=labels, average='micro', zero_division=0),
            'recall':    recall_score(y_true, y_pred, labels=labels, average='micro', zero_division=0),
            'f1':        f1_score(y_true, y_pred, labels=labels, average='micro', zero_division=0),
        },
        'macro': {
            'precision': precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
            'recall':    recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
            'f1':        f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        },
        'weighted': {
            'precision': precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
            'recall':    recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
            'f1':        f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        },
    }

        # ─── per‐entity‐type scores ────────────────────────────────────────────────────
    # average=None returns an array in label order
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls    = recall_score(   y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s        = f1_score(       y_true, y_pred, labels=labels, average=None, zero_division=0)
    supports   = pd.Series(y_true).value_counts().reindex(labels, fill_value=0)

    df_per_type = pd.DataFrame({
        'precision': precisions,
        'recall':    recalls,
        'f1-score':  f1s,
        'support':   supports.values
    }, index=labels)

    return {
        "micro_scores": overall['micro'],
        "macro_scores": overall['macro'],
        "weighted_scores": overall['weighted'],   
        "df_per_type": df_per_type,
    }
    


def get_llm_judgment_stats(responses_data, threshold=None):
    entity_predictions = dict()
    for response in responses_data:
        for model_name, model_response in response.items():
            for prediction in model_response['predictions']:
                entity = prediction['entity']
                if entity not in entity_predictions:
                    entity_predictions[entity] = []
                entity_predictions[entity].append({**prediction, 'model': model_name})
    
        
        # Function to compute overall accuracy for each entity
    def compute_entity_accuracy(entity_data, threshold=threshold, filters=None):
        new_data = entity_data.copy()
        if filters:
            new_data = [item for item in new_data if all(item.get(k) == v for k, v in filters.items())]
            
        total = len(new_data)
        if total == 0:
            return None
        correct = sum(1 for item in new_data if item['correct'])
        avg_correct = correct / total if total > 0 else None
        if threshold is not None:
            return avg_correct >= threshold
        return avg_correct
        
    # Function to compute overall accuracy (average of accuracy of each entity)
    def compute_overall_accuracy(data, threshold=None, filters=None):
        total_accuracies = list(compute_entity_accuracy(entity_data, threshold, filters=filters) for entity_data in data.values())
        # print(total_accuracies)
        total_accuracies = [accuracy for accuracy in total_accuracies if accuracy is not None]
        # print(total_accuracies)
        return sum(total_accuracies) / len(total_accuracies)


    # Function to compute accuracy of each entity per model
    def compute_entity_accuracy_per_model(data, threshold=None):
        models = {dp['model'] for ed in data.values() for dp in ed}
        # Calculate the average correct per model and entity
        entity_accuracy_per_model = {
            model: compute_overall_accuracy(data, threshold=threshold, filters={'model': model})
            for model in models
        }
        
        return entity_accuracy_per_model

    # Function to compute accuracy of entities of each entity type
    def compute_accuracy_per_entity_type(data, threshold=None):
        entity_types = {dp['tag'] for ed in data.values() for dp in ed}
        
        # Calculate the average correct per entity type
        accuracy_per_type = {
            entity_type: compute_overall_accuracy(
                data, 
                threshold=threshold, 
                filters={'tag': entity_type}
            )
            for entity_type in entity_types
        }
        
        return accuracy_per_type

    # Function to compute accuracy of entities across different models for each entity type
    def compute_accuracy_per_type_per_model(data, threshold=None):
        entity_types = {dp['tag'] for ed in data.values() for dp in ed}
        models = {dp['model'] for ed in data.values() for dp in ed}
        accuracy_per_type_per_model = {tag_type: dict() for tag_type in entity_types}
        for tag_type in entity_types:
            for model in models:
                accuracy_per_type_per_model[tag_type][model] = compute_overall_accuracy(
                    data, 
                    threshold=threshold, 
                    filters={'tag': tag_type, 'model': model}
            )
        return accuracy_per_type_per_model

    return {
        "overall_accuracy": compute_overall_accuracy(entity_predictions, threshold),
        "model_accuracy": compute_entity_accuracy_per_model(entity_predictions, threshold),
        "entity_type_accuracy": compute_accuracy_per_entity_type(entity_predictions, threshold),
        "model_entity_type_accuracy": compute_accuracy_per_type_per_model(entity_predictions, threshold),    
    }