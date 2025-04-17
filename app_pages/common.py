import streamlit as st
import re


# Initial data
def get_data():
    if 'data' not in st.session_state:
        st.error("No data found. Please upload a file or start a new session.")
        return None
    
    return st.session_state['data']


# Helper Functions
def extract_entities(tagged_text):
    pattern = r"<(.*?)>(.*?)</\1>"
    return [(match[0], match[1]) for match in re.findall(pattern, tagged_text)]


def add_entity_status():
    data = get_data()
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
            
    st.session_state['data'] = data
    return True