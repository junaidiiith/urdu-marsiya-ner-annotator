import streamlit as st
from ner_annotator.utils import (
    get_all_files, 
    get_llm_configs,
    save_ner_tags,
    save_text_with_hash
)
from ner_annotator.constants import DATASET_DIR
from ner_annotator.llm_tagger import get_ner_tags
from stqdm import stqdm


text_states = {
    1: "File received. Please wait until the contents show in the text box. Loading...",
    2: "File content loaded successfully.",
    3: "Text received."
}


def set_tagged_result(text, ner_tags):
    st.session_state['data'] = save_ner_tags(text, ner_tags)
    st.success("NER tagging completed. Now you can move to reviewing the results.")


def start_ner_tagging(text):
    if text:
        model_id = st.session_state.get("selected_model_id")
        chunk_size = st.session_state.get("chunk_size")
        with st.spinner("LLM-based NER Tagging...Will take a while for large texts."):
            show_message(message="Tagging in progress...")
            ner_tags = get_ner_tags(text, model_id=model_id, chunk_size=chunk_size, tqdm=stqdm)
        print("Total NER Tags:", len(ner_tags))
        set_tagged_result(text, ner_tags)


def add_text_if_not_exists(text):
    """
    Process the text for NER tagging.
    This function should be replaced with the actual NER processing logic.
    """
    st.session_state['data'] = save_text_with_hash(text)
    data = st.session_state['data']
    st.session_state['data'] = data
    if 'tagged' in data and data['tagged']:
        st.success("Tags already exists. You can now proceed to reviewing tags.")
        return False
    
    return True


def select_ner_config():
    with st.expander("Select LLM Configurations", expanded=False):
        llm_configs = get_llm_configs()
        providers = list(llm_configs.keys())
        selected_provider = st.selectbox("Select LLM Provider", providers)
        prefix = llm_configs[selected_provider]["prefix"]
        default_model_id = llm_configs[selected_provider]["default"]
        model_ids = llm_configs[selected_provider]["models"]
        model_ids = list({m['model_id'] for m in model_ids})
        default_model_idx = model_ids.index(default_model_id)
        
        selected_model = st.selectbox("Select Model", model_ids, key="model_id", index=default_model_idx)
        selected_model_id = f"{prefix}/{selected_model}"
        st.session_state['selected_model_id'] = selected_model_id
        st.number_input("Chunk Size (for large texts)", min_value=1, max_value=100, value=40, step=5, key="chunk_size")


def initiate_ner_tagging(text):
    if not text:
        st.warning("Please provide some text first.")
        return
    print("Starting NER tagging on...")
    print(text[:50])
    print("Total length:", len(text))
    if add_text_if_not_exists(text):
        start_ner_tagging(text)

def show_message(message, message_type="info"):
    if message_type == "info":
        st.info(message)
    elif message_type == "warning":
        st.warning(message)
    elif message_type == "error":
        st.error(message)
    elif message_type == "success":
        st.success(message)
    else:
        st.write(message)

def main():

    st.title("📜 LLM-based Marsiya Named Entity Tagging")

    select_ner_config()

    st.markdown("### Choose your input method:")

    # Tabs for three options
    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "✍️ Paste Text", "🔍 Search Existing"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=["txt"]
        )
        if uploaded_file:
            # wrap the slow part in a spinner
            with st.spinner("Loading file…"):
                text = uploaded_file.read().decode("utf-8")

            # now show the text
            st.text_area(
                "Uploaded Content",
                value=text,
                height=300,
                key="uploaded_file_text"
            )
            if st.session_state.get("uploaded_file_text"):
                show_message(message=text_states[2], message_type="success")
                # if st.button("🖋️ Tag this file", key="tag_file"):
                #     initiate_ner_tagging(text)
            else:
                show_message(message=text_states[1])
                
    with tab2:
        pasted_text = st.text_area("Paste Marsiya Text Below:", height=300)
        if pasted_text:
            show_message(message=text_states[3], message_type="success")
            if st.button("🖋️ Tag this file", key="tag_pasted_text"):
                initiate_ner_tagging(pasted_text)
            

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
            # if st.button("🖋️ Tag this file", key="tag_existing_file"):
            #     initiate_ner_tagging(all_marsiya_files[selected_file]['content'])

main()