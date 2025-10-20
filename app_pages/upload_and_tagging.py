import streamlit as st
from app_pages.common import init_session_state, set_text_session_data
from ner_annotator.utils import (
    get_all_files,
    get_llm_configs,
    save_ner_tags,
    save_text_with_hash,
    calculate_hash
)
from ner_annotator.constants import DATASET_DIR
from ner_annotator.llm_tagger import NERMode, get_ner_tags
from stqdm import stqdm
import time

text_states = {
    1: "File received. Please wait until the contents show in the text box. Loading...",
    2: "File content loaded successfully.",
    3: "Text received.",
}

def get_file_suffix():
    """
    Get the file suffix based on the selected model and NER mode.
    """
    model_id = st.session_state.get("selected_model_id").replace("/", "_").replace(".", "_")
    ner_mode = st.session_state.get("ner_mode", NERMode.MARSIYA_ADVANCED.value)
    return f"_{model_id}_{ner_mode}.json"


def set_tagged_result(text, ner_tags, suffix=None):
    set_text_session_data(**save_ner_tags(text, ner_tags, suffix=suffix))
    st.success("NER tagging completed. Now you can move to reviewing the results.")


def start_ner_tagging(text):
    print("Starting NER tagging...1234567890")
    if text:
        model_id = st.session_state.get("selected_model_id")
        chunk_size = st.session_state.get("chunk_size")
        st.text(f"Using model: {model_id}")
        print("Prefix: ", get_file_suffix())
        with st.spinner("LLM-based NER Tagging...Will take a while for large texts."):
            show_message(message="Tagging in progress...")
            ner_tags = get_ner_tags(
                text, 
                mode=NERMode.MARSIYA_ADVANCED,
                model_id=model_id, 
                chunk_size=chunk_size, 
                tqdm=stqdm
            )
        print("Total NER Tags:", len(ner_tags))
        set_tagged_result(text, ner_tags, suffix=get_file_suffix())


def add_text_if_not_exists(text):
    """
    Process the text for NER tagging.
    This function should be replaced with the actual NER processing logic.
    """
    print(f"Prefix: {get_file_suffix()}")
    data = save_text_with_hash(text, suffix=get_file_suffix())
    current_hash = calculate_hash(text)
    print("Setting current hash:", current_hash)
    st.session_state["current_hash"] = current_hash
    set_text_session_data(**data)
    # print("Data: ", data)
    if "tagged" in data and data["tagged"]:
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
        model_ids = list({m["model_id"] for m in model_ids})
        default_model_idx = model_ids.index(default_model_id)
        # import os
        # print("API Key:", os.environ.get("OPENAI_API_KEY", "Not Set"))

        selected_model = st.selectbox(
            "Select Model", model_ids, key="model_id", index=default_model_idx
        )
        selected_model_id = f"{prefix}/{selected_model}"
        st.session_state["selected_model_id"] = selected_model_id
        st.number_input(
            "Chunk Size (for large texts)",
            min_value=1,
            max_value=100,
            value=40,
            step=5,
            key="chunk_size",
        )
        st.selectbox(
            "NER Mode",
            options=[mode.value for mode in NERMode],
            index=0,
            key="ner_mode",
            help="Select the NER mode to use for tagging.",
        )


def initiate_ner_tagging(text):
    if not text:
        st.warning("Please provide some text first.")
        return
    print("Starting NER tagging on...")
    print(text[:50])
    print("Total length:", len(text))
    # set_current_hash(text)

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


def set_current_hash(key, filename):
    print("Setting current hash for key:", key)
    print("Filename:", filename)
    text = st.session_state.get(key)
    if text:
        text_hash = calculate_hash(text)
        print("Text hash: ", text_hash)
        st.session_state["current_hash"] = text_hash
        init_session_state(text, text_hash, filename)


def main():
    start_time = time.time()
    st.title("📜 LLM-based Marsiya Named Entity Tagging")

    select_ner_config()
    print("Configurations loaded in", time.time() - start_time, "seconds.")
    st.markdown("### Choose your input method:")

    # Tabs for three options
    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "✍️ Paste Text", "🔍 Search Existing"])

    with tab1:
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file:
            # wrap the slow part in a spinner
            with st.spinner("Loading file…"):
                text = uploaded_file.read().decode("utf-8")

            # now show the text
            st.text_area(
                "Uploaded Content",
                value=text,
                height=300,
                on_change=set_current_hash,
                key="uploaded_file_text",
                kwargs={"key": "uploaded_file_text", "filename": uploaded_file.name},
            )
            if st.session_state.get("uploaded_file_text"):
                show_message(message=text_states[2], message_type="success")
                if st.button("🖋️ Tag this file", key="tag_file"):
                    initiate_ner_tagging(st.session_state["uploaded_file_text"])
            else:
                show_message(message=text_states[1])

    with tab2:
        pasted_text = st.text_area(
            "Paste Marsiya Text Below and then press cmd+Enter:",
            height=300,
            on_change=set_current_hash,
            key="pasted_text",
            kwargs={"key": "pasted_text", "filename": "Pasted Text"},
        )
        if pasted_text:
            show_message(message=text_states[3], message_type="success")
            if st.button("🖋️ Tag this file", key="tag_pasted_text"):
                initiate_ner_tagging(pasted_text)

    with tab3:
        st.markdown("Search from existing marsiyas:")
        tagged_filter = st.toggle("🔖 Show only tagged files", value=False)
        all_marsiya_files: dict = st.session_state.get(
            "all_marsiya_files", get_all_files(f"{DATASET_DIR}")
        )
        # Filter based on 'tagged' flag
        filtered_files = [
            name
            for name, meta in all_marsiya_files.items()
            if tagged_filter and meta["tagged"] or not tagged_filter
        ]

        selected_file = st.selectbox("Select a marsiya", options=filtered_files)

        if selected_file:
            # In a real app, this would load content from a file or DB
            
            st.info(
                f"Selected Marsiya: {selected_file} {'(Tagged)' 
                if all_marsiya_files[selected_file]['tagged'] else ''}"
            )
            content = f"{selected_file}\nContent: {all_marsiya_files[selected_file]['content']}"
            content = f"{all_marsiya_files[selected_file]['content']}"
            st.text_area(
                "Marsiya Content",
                value=content,
                height=300,
                on_change=set_current_hash,
                key="existing_file_text",
                kwargs={"key": "existing_file_text", "filename": selected_file},
            )
            set_current_hash("existing_file_text", selected_file)
            if st.button("🖋️ Tag this file", key="tag_existing_file"):
                print("Initiating NER tagging for selected file:", selected_file)
                content = st.session_state.get("existing_file_text")
                initiate_ner_tagging(content)


    print("File upload and tagging completed in", time.time() - start_time, "seconds.")


main()
