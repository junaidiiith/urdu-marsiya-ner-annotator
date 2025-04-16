import json
import os
import streamlit as st
from ner_annotator.utils import (
    calculate_hash, 
    get_all_files, 
    get_llm_configs
)
from ner_annotator.constants import UPLOAD_DIR, DATASET_DIR
from ner_annotator.llm_tagger import get_ner_tags


def start_ner_tagging(text):
    if text:
        model_id = st.session_state.get("selected_model_id")
        chunk_size = st.session_state.get("chunk_size")
        with st.spinner("LLM-based NER Tagging...Will take a while for large texts."):
            ner_tags = get_ner_tags(text, model_id=model_id, chunk_size=chunk_size)
        print("Total NER Tags:", len(ner_tags))
        set_tagged_result(text, ner_tags)


def add_text_if_not_exists(text):
    """
    Process the text for NER tagging.
    This function should be replaced with the actual NER processing logic.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    text_hash = calculate_hash(text)
    if not os.path.exists(f"{UPLOAD_DIR}/{text_hash}.json"):
        with open(f"{UPLOAD_DIR}/{text_hash}.json", "w") as f:
            data = {
                "text": text,
                "tagged": False,
            }
            json.dump(data, f, indent=4)
        st.session_state['data'] = data
        st.success("Text added for processing.")
    else:
        st.warning("This text has already been processed.")
        with open(f"{UPLOAD_DIR}/{text_hash}.json", "r") as f:
            data = json.load(f)
            st.session_state['data'] = data
            st.success("Text already exists. You can proceed to tagging.")


def set_tagged_result(text, ner_tags):
    text_hash = calculate_hash(text)
    with open(f"{UPLOAD_DIR}/{text_hash}.json", "r") as f:
        data = json.load(f)
        data["tagged_elements"] = ner_tags
        data["tagged"] = True
        with open(f"{UPLOAD_DIR}/{text_hash}.json", "w") as f:
            json.dump(data, f, indent=4)
    
    st.session_state['data'] = data
    st.success("NER tagging completed. Now you can move to reviewing the results.")


def select_ner_config():
    with st.expander("Select LLM Configurations", expanded=True):
        llm_configs = get_llm_configs()
        providers = list(llm_configs.keys())
        selected_provider = st.selectbox("Select LLM Provider", providers)
        default_model_id = llm_configs[selected_provider]["default"]
        model_ids = llm_configs[selected_provider]["models"]
        model_id_to_name = {m['name']: m['model_id'] for m in model_ids}
        default_model_idx = list(model_id_to_name.values()).index(default_model_id)
        
        selected_model = st.selectbox("Select Model", list(model_id_to_name.keys()), key="model_id", index=default_model_idx)
        selected_model_id = f"{selected_provider}/{model_id_to_name[selected_model]}"
        st.session_state['selected_model_id'] = selected_model_id
        st.number_input("Chunk Size (for large texts)", min_value=1, max_value=100, value=40, step=5, key="chunk_size")


st.set_page_config(page_title="Marsiya NER", layout="centered")
st.title("📜 LLM-based Marsiya Named Entity Tagging")

select_ner_config()

st.markdown("### Choose your input method:")

# Tabs for three options
tab1, tab2, tab3 = st.tabs(["📁 Upload File", "✍️ Paste Text", "🔍 Search Existing"])

with tab1:
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Content", value=text, height=300)
        
with tab2:
    pasted_text = st.text_area("Paste Marsiya Text Below:", height=300)
    if pasted_text:
        st.success("Text received.")

with tab3:
    st.markdown("Search from existing marsiyas:")
    tagged_filter = st.toggle("🔖 Show only tagged files", value=False)
    all_marsiya_files: dict = st.session_state.get("all_marsiya_files", get_all_files(f"{DATASET_DIR}"))
    # Filter based on 'tagged' flag
    filtered_files = [
        name for name, meta in all_marsiya_files.items()
        if tagged_filter and meta["tagged"] or not tagged_filter
    ]

    selected_file = st.selectbox("Select a marsiya", options=filtered_files)

    if selected_file:
        # In a real app, this would load content from a file or DB
        st.info(f"Selected Marsiya: {selected_file} {'(Tagged)' if all_marsiya_files[selected_file]['tagged'] else ''}")
        content = f"{selected_file}\nContent: {all_marsiya_files[selected_file]['content']}"
        st.text_area("Marsiya Content", value=content, height=300)


if st.button("Start LLM Tagging"):
    final_text = ""
    if uploaded_file:
        final_text = text
    elif pasted_text:
        final_text = pasted_text
    elif selected_file:
        final_text = content

    if final_text:
        st.success("Ready to process this text.")
        add_text_if_not_exists(final_text)
        start_ner_tagging(final_text)
    else:
        st.warning("Please provide some text first.")
