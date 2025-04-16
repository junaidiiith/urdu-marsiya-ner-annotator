import json
import streamlit as st
import re

from ner_annotator.constants import UPLOAD_DIR
from ner_annotator.stats import get_stats
from ner_annotator.utils import calculate_hash



# Color map for entity types
TAG_COLORS = {
    "PERSON": "#AED6F1",       # Light Blue
    "LOCATION": "#A9DFBF",     # Light Green
    "DATA": "#FCF3CF",         # Light Yellow
    "TIME": "#FADBD8",         # Light Pink
    "ORGANIZATION": "#F9E79F", # Light Orange
    "DESGINATION": "#D5DBDB",  # Light Gray
    "NUMBER": "#D7BDE2"        # Light Purple
}

# Entity tags
TAGS = list(TAG_COLORS.keys())


# Initial data
def get_data():
    if 'data' not in st.session_state:
        st.error("No data found. Please upload a file or start a new session.")
    
    return st.session_state['data']


def add_entity_status():
    data = get_data()
    if 'tagged_elements' not in data:
        st.error("No tagged elements found in the data. ")
        return
    
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


def get_current_line():
    tagged_data = get_data()['tagged_elements']
    return tagged_data[st.session_state.get('current_line')-1]

def set_current_line(line_number, content):
    st.session_state['data']['tagged_elements'][line_number-1]['tagged'] = content

def get_current_entities_status():
    return get_current_line()['entity_status']

def set_current_entities_status(entities_status):
    current_line = get_current_line()
    current_line['entity_status'] = entities_status
    st.session_state['data']['tagged_elements'][st.session_state.get('current_line')-1] = current_line

def set_new_entity_tag_current_entities_status(entity, new_tag):
    current_entities_status = get_current_entities_status()
    if entity in current_entities_status:
        current_entities_status[entity].update({
            'user_updated': new_tag,
        })
        set_current_entities_status(current_entities_status)
    else:
        st.error(f"Entity '{entity}' not found in the current line.")


# Helper Functions
def extract_entities(tagged_text):
    pattern = r"<(.*?)>(.*?)</\1>"
    return [(match[0], match[1]) for match in re.findall(pattern, tagged_text)]

def render_tagged_text():
    display_text: str = get_current_line()["tagged"]
    entities = extract_entities(display_text)
    for tag, text in entities:
        color = TAG_COLORS.get(tag.upper(), "#E5E7E9")  # Default light grey
        # print("Tag:", tag, "Text:", text)
        replacement = f'<span style="background-color: {color}; padding: 2px;" title="{tag}">{text}</span>'
        display_text = display_text.replace(f"<{tag}>{text}</{tag}>", replacement)
    st.markdown(display_text, unsafe_allow_html=True)


def set_new_tag(entity, old_tag, new_tag):
    tagged_text: str = get_current_line()["tagged"]
    pattern = f"<{old_tag}>{entity}</{old_tag}>" if old_tag else entity
    replacement = f"<{new_tag}>{entity}</{new_tag}>"
    set_current_line(
        st.session_state['current_line'], 
        tagged_text.replace(pattern, replacement)
    )
    set_new_entity_tag_current_entities_status(entity, new_tag)
    st.success(f"Tagged '{entity}' as {new_tag}")
    st.rerun()
    
    
def save_file_data():
    text_hash = calculate_hash(get_data()['text'])
    with open(f"{UPLOAD_DIR}/{text_hash}.json", "w") as f:
        json.dump(get_data(), f, indent=4)
    # print(get_data()['tagged_elements'][0].keys())
    print("File saved successfully.")
    

def save_tags():
    current_entity_status = get_current_entities_status()
    current_entity_status.update({
        'user_verified': True,
    })
    set_current_entities_status(current_entity_status)
    print("Tags at line: ", st.session_state['current_line'], "saved successfully.")
    
    
def manual_tagging():
    st.subheader("Tag Untagged Words")
    words = re.sub(r"<.*?>", "", get_current_line()["original"]).split()

    selected_words = st.multiselect("Select words to tag", words, placeholder="Select words from dropdown")
    new_tag_type = st.selectbox("Select tag", TAGS)
    if st.button("Add Tag"):
        if selected_words:
            selected_text = " ".join(selected_words)
            entities = extract_entities(get_current_line()["tagged"])
            if any(selected_text in entity for _, entity in entities):
                st.warning("This phrase is already tagged.")
            elif selected_text in get_current_line()["original"]:
                set_new_tag(selected_text, None, new_tag_type)
            else:
                st.error("Selected words are not part of the original text.")



def tags_review():
        # Review tagged entities
    entities = extract_entities(get_current_line()["tagged"])
    cols = st.columns([6, 1])
    with cols[0]:
        st.subheader("Review Existing Tags")
    with cols[1]:
        st.button("Save", key="save_tags", on_click=save_tags)
    
    if not entities:
        st.write("No tagged entities found in this line.")
        return
    entities = extract_entities(get_current_line()["tagged"])
    for i, (tag, entity) in enumerate(entities):
        col1, col2, col3 = st.columns([3, 2, 3])
        with col1:
            st.write(f"**{entity}**")
        with col2:
            correct = st.radio(f"Is '{entity}' correctly tagged as {tag}?", ["Yes", "No"], key=f"correct_{i}")
        with col3:
            if correct == "No":
                new_tag = st.selectbox(f"Correct tag for '{entity}'", TAGS, key=f"newtag_{i}")
                if st.button("Update Tag", key=f"btn_update_{i}"):
                    set_new_tag(entity, tag, new_tag)

def show_file_statistics():
    stats = get_stats(get_data())

    # Header with button floated right
    header_col, _, btn_col = st.columns([4, 1, 1])
    with header_col:
        st.subheader("📊 File Statistics")
    with btn_col:
        st.button("✉️ Send Email")

    st.markdown("---")

    # Top‐line metrics
    m1, m2 = st.columns(2)
    m1.metric("Total Entities", stats["total_entities"])
    m2.metric("Verified Entities", stats["total_verified"])

    st.markdown("**Entities per Category**")

    # One metric per category
    cat_cols = st.columns(len(stats["per_category_count"]))
    for col, (tag, cnt) in zip(cat_cols, stats["per_category_count"].items()):
        col.metric(tag.capitalize(), cnt)
# Initialize session state
def main():
    
    add_entity_status()
    st.set_page_config(page_title="Marsiya NER Reviewing", layout="centered")
    st.title("📜 LLM-based NER Manual Review")

    # Display Legend
    st.subheader("Named Entity Categories")
    legend_html = ""
    for tag, color in TAG_COLORS.items():
        legend_html += f'<span style="background-color: {color}; padding: 4px; margin:4px; border-radius:4px;">{tag}</span> '
    st.markdown(legend_html, unsafe_allow_html=True)
    st.markdown("---")
    max_lines = len(get_data()['tagged_elements'])
    st.number_input(f"Select Line Number / {max_lines}", min_value=1, max_value=max_lines, value=1, key="current_line")
    # cols = st.columns([1, 2, 1])
    # with cols[0]:
    #     st.number_input("Select Line Number", min_value=1, max_value=len(get_data()['tagged_elements']), value=1, key="current_line")
    # with cols[2]:
    #     st.button("Save Annotations", key="save_changes", on_click=save_file_data)
    
    # Display original and translated text
    cols = st.columns([1, 2, 1])
    with cols[0]:
        st.subheader("Original Text")
    with cols[-1]:
        st.button("Save Annotations", key="save_changes", on_click=save_file_data)
    st.write(get_current_line()["original"])

    st.subheader("English Translation")
    st.write(get_current_line()["english"])


    # Display tagged text
    st.subheader("Tagged Text")
    render_tagged_text()
    st.markdown("---")
    tags_review()
    manual_tagging()
    st.markdown("---")
    show_file_statistics()

main()