import streamlit as st
import re

from ner_annotator.utils import (
    get_llm_judgment_excel, 
    get_ner_tags_excel
)


# Helper Functions
def extract_entities(tagged_text):
    pattern = r"<(.*?)>(.*?)</\1>"
    return [(match[0], match[1]) for match in re.findall(pattern, tagged_text)]


def add_entity_status():
    data = get_current_data()
    if not data:
        st.error("No data found. Please upload a file or start a new session.")
        return False
    
    if 'tagged_elements' not in data:
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
    return True


def set_current_text_hash(text_hash):
    st.session_state['current_hash'] = text_hash

def get_current_text_hash():
    
    if 'current_hash' not in st.session_state:
        st.error("You have not set the text to work with yet. Please go to the upload page.")
        return 
    
    return st.session_state['current_hash']

def get_current_data():
    current_hash = get_current_text_hash()
    if current_hash is None:
        st.error("No data found. Please upload a file or start a new session.")
        return 
    return st.session_state[current_hash]

def init_session_state(text, text_hash):
    st.session_state[text_hash] = dict()
    st.session_state[text_hash] = {
        'text': text,
        'tagged_elements': [],
        'tagged': False,
        'llm_judgement': [],
    }
    

def set_text_session_data(**kwargs):
    current_hash = get_current_text_hash()
    current_data = st.session_state[current_hash]
    current_data.update(kwargs)
    st.session_state[current_hash] = current_data
    

def download_ner_tags_data():
    current_data = get_current_data()
    ner_tags_excel = get_ner_tags_excel(current_data['tagged_elements'])
    return ner_tags_excel


def download_llm_judgement_data():
    current_data = get_current_data()
    if 'llm_judgement' in current_data:
        llm_judgement_excel = get_llm_judgment_excel(current_data['llm_judgement'])
        return llm_judgement_excel
    else:
        st.error("No LLM judgement data found.")
        return None
    