import time
import pandas as pd
import streamlit as st
import re

from ner_annotator.utils import (
    get_llm_judgment_excel,
    get_llm_judgment_stats, 
    get_ner_tags_excel,
    encode_df,
    get_stats
)


# Helper Functions
def extract_entities(tagged_text):
    pattern = r"<(.*?)>(.*?)</\1>"
    return [(match[0], match[1]) for match in re.findall(pattern, tagged_text)]


def add_entity_status():
    start_time = time.time()
    data = get_current_data()
    if not data:
        st.error("No data found. Please upload a file or start a new session.")
        return False
    
    if 'tagged_elements' not in data or not data['tagged_elements']:
        st.error("No tagged elements found in the data. ")
        return False
    
    for i, item in enumerate(data['tagged_elements']):
        if 'entity_status' not in item:
            entities = extract_entities(item["tagged"])
            entities_status = {
                entity: {
                    'entity': entity,
                    'tag': tag,
                    'user_updated': None,
                }
                for tag, entity in entities    
            }
            data['tagged_elements'][i]['entity_status'] = entities_status
            
    set_text_session_data(**data)
    print("Added entity status in ", time.time() - start_time, "seconds.")
    return True


def get_current_text_hash():
    
    if 'current_hash' not in st.session_state:
        st.error("You have not set the text to work with yet. Please go to the upload page.")
        return 
    
    return st.session_state['current_hash']

def get_current_data(text_hash=None):
    current_hash = get_current_text_hash() if not text_hash else text_hash
    if current_hash is None:
        st.error(f"No data found for {text_hash}. Please upload a file or start a new session.")
        return 
    return st.session_state[current_hash]


def init_session_state(text, text_hash, filename):
    st.session_state[text_hash] = dict()
    st.session_state[text_hash] = {
        "filename": filename,
        'text': text,
        'tagged_elements': [],
        'tagged': False,
        'llm_judgement': [],
    }
    
    st.session_state['all_hashes'] = st.session_state.get('all_hashes', [])
    if text_hash not in st.session_state['all_hashes']:
        st.session_state['all_hashes'].append(text_hash)
    st.session_state['current_hash'] = text_hash


def set_text_session_data(**kwargs):
    current_hash = get_current_text_hash()
    current_data = st.session_state[current_hash]
    current_data.update(kwargs)
    st.session_state[current_hash] = current_data
    

# st.cache_data(max_entries=10)
def download_ner_tags_data(text_hash):
    current_data = get_current_data(text_hash=text_hash) 
    file_name = current_data.get('filename')   
    if 'tagged_elements' not in current_data:
        st.error("No NER tags data found.")
        return None
    ner_tags_excel = get_ner_tags_excel(file_name, current_data['tagged_elements'])
    return ner_tags_excel


st.cache_data(max_entries=10)
def download_llm_judgement_data(text_hash):
    current_data = get_current_data(text_hash=text_hash)
    file_name = current_data.get('filename')
    if 'llm_judgement' not in current_data:
        st.error("No LLM judgement data found.")
        print("No LLM judgement data found.")
        return None

    llm_judgement_excel = get_llm_judgment_excel(file_name, current_data['llm_judgement'])
    return llm_judgement_excel


def get_combined_data(data_fn, hashes):
    all_excels = [data_fn(text_hash=text_hash) for text_hash in hashes]
    all_excels = [excel for excel in all_excels if excel is not None]
    combined_excel = pd.concat([pd.read_excel(df) for df in all_excels], ignore_index=True, axis=0)
    print("Combined DataFrame shape:", combined_excel.shape)
    return encode_df(combined_excel, "Combined NER Tags")


def download_all_ner_tags_data():
    all_hashes = st.session_state.get('all_hashes', [])
    if not all_hashes:
        st.error("No NER tags data found.")
        return None
    print("All hashes:", all_hashes)
    return get_combined_data(download_ner_tags_data, all_hashes)


def download_all_llm_judgement_data():
    all_hashes = st.session_state.get('all_hashes', [])
    if not all_hashes:
        st.error("No LLM judgement data found.")
        return None
    return get_combined_data(download_llm_judgement_data, all_hashes)


def get_current_file_review_stats():
    current_data = get_current_data()
    return get_stats(current_data['tagged_elements'])


def get_all_files_review_stats():
    all_hashes = st.session_state.get('all_hashes', [])
    if not all_hashes:
        st.error("No NER tags data found.")
        return None
    
    all_data = sum([
        get_current_data(text_hash=text_hash)['tagged_elements'] 
        for text_hash in all_hashes], []
    )
    return get_stats(all_data)
        

@st.cache_data(max_entries=10)
def get_judgment_stats(data, threshold):
    return get_llm_judgment_stats(data, threshold=threshold)