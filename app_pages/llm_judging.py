import time
import streamlit as st
import pandas as pd

from app_pages.common import (
    download_all_llm_judgement_data,
    download_llm_judgement_data,
    get_current_data, 
    add_entity_status,
    get_judgment_stats,
    set_text_session_data
)


from ner_annotator.llm_judge import run_evaluation
from ner_annotator.utils import save_llm_judgement
from settings import SUPPORTED_LLM_JUDGE_MODELS
from stqdm import stqdm


def has_judgment_data():
    # set_text_session_data(**{'llm_judgement': st.session_state['evaluated_data']})
    data = get_current_data()
    # print("Judgment data:", data)
    # print(st.session_state['evaluated_data'])
    return all(c in data and data[c] for c in ['tagged_elements', 'llm_judgement'])


def set_judgment_configuration():
    with st.expander("Judgment Configuration - Select LLMs, Advanced Settings", expanded=False):
        st.markdown("**Judgment Configuration**")
        st.markdown("Configure the LLM-based judgment settings.")
        
        st.markdown("**Model Selection**")
        st.markdown("Select the models you want to compare. The selected models will be used for evaluation.")
        
        st.markdown("**Available Models**")
        for model in SUPPORTED_LLM_JUDGE_MODELS:
            if model[1]:
                print(model[0])
                st.checkbox(model[0], value=True, key=model[0])
        
        selected_models = [model[0] for model in SUPPORTED_LLM_JUDGE_MODELS if st.session_state.get(model[0], False) and model[1]]
        st.session_state['selected_models'] = selected_models

        st.number_input(
            "Sentence Chunk Size",
            min_value=1,
            value=15,
            help="Number of sentences to process in each chunk.",
            key="sentence_chunk_size"
        )
        st.number_input(
            "Context Size",
            min_value=1,
            max_value=10,
            value=2,
            help="Number of before and after sentences to include as context for each sentence.",
            key="context_size"
        )
        st.number_input(
            "Judgment Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Threshold for judgment. If the prediction by these many LLMs out of all judges is judged 'Correct', the prediction is considered correct.",
            key="judgment_threshold"
        )

def evaluate_models():
    
    # if has_judgment_data():
    #     st.success("Judgment data already exists. You can view the results.")
    #     st.balloons()
    #     return
    if st.button("Run LLM-As-A-Judge Evaluation"):
        tagged_data = get_current_data()['tagged_elements']
        selected_models = st.session_state.get('selected_models')
        sentence_chunk_size = st.session_state.get('sentence_chunk_size')
        context_size = st.session_state.get('context_size')
        with st.spinner("Evaluating..."):
            if not selected_models:
                st.warning("Please select at least one model to evaluate.")
                return
            st.session_state['evaluated_data'] = run_evaluation(
                tagged_data, selected_models, 
                sentence_chunk_size, context_size,
                tqdm=stqdm
            )
            st.success("Evaluation completed!")
            save_llm_judgement(get_current_data()['text'], st.session_state['evaluated_data'])
            set_text_session_data(**{'llm_judgement': st.session_state['evaluated_data']})
            st.balloons()
            st.rerun()


def show_results():
    st.subheader("Evaluation Results")

    threshold = st.session_state.get('judgment_threshold')
    results = get_judgment_stats(st.session_state['evaluated_data'], threshold)
    overall_acc = results['overall_accuracy']
    model_acc = results['model_accuracy']
    tag_acc = results['entity_type_accuracy']
    tag_model_acc = results['model_entity_type_accuracy']
    
    
    # Top‑level metrics
    st.metric(f"Accuracy Entities ≥ {threshold:.0%}", f"{overall_acc:.2f}%")
    st.markdown("---")

    # 4) Per‑model accuracy
    st.subheader("2. Accuracy per Model")
    df_mod = pd.DataFrame.from_dict(model_acc, orient="index", columns=["Accuracy"])
    df_mod = df_mod.sort_values("Accuracy", ascending=False)
    st.dataframe(df_mod.style.format("{:.2%}"), use_container_width=True)
    st.bar_chart(df_mod["Accuracy"], use_container_width=True)

    st.markdown("---")

    # 5) Per‑tag accuracy
    st.subheader("3. Accuracy per Entity Type (Tag)")
    df_tag = pd.DataFrame.from_dict(tag_acc, orient="index", columns=["Accuracy"])
    df_tag = df_tag.sort_values("Accuracy", ascending=False)
    st.dataframe(df_tag.style.format("{:.2%}"), use_container_width=True)
    st.bar_chart(df_tag["Accuracy"], use_container_width=True)

    st.markdown("---")

    # 6) Per‑tag‑per‑model accuracy
    st.subheader("4. Accuracy per Type per Model")
    df_tm = pd.DataFrame(tag_model_acc)
    st.dataframe(df_tm.style.format("{:.2%}"), use_container_width=True)


def download_data():
    def prepare_download(key):
        st.session_state[key] = download_llm_judgement_data(st.session_state["current_hash"])

    def prepare_download_all(key):
        st.session_state[key] = download_all_llm_judgement_data()
        
    def unset_download_data(key):
        if key in st.session_state:
            del st.session_state[key]
        else:
            st.warning("No LLM Judgement data to clear.")

    
    st.markdown("**Download LLM Judgment Results**")
    st.markdown(
        "Click the button to generate the file. After the file is generated, you can download it."
    )
    cols = st.columns([6, 6])
    with cols[0]:
        st.markdown(
            "Download the LLM Judgment for current file"
        )
        judgment_data_key = "judgment_data_key"
        if judgment_data_key not in st.session_state:
            st.button(
                label="Generate LLM Judgment Data",
                on_click=prepare_download,
                kwargs={"key": judgment_data_key},
                help="Click to build the Excel file before downloading"
            )
        else:
            st.download_button(
                label=":arrow_down: Download LLM Judgment Data",
                data=st.session_state[judgment_data_key],
                file_name="llm_judgment_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet",
                key="download_llm_judgement",
                help="Your Excel file is ready—click to save it locally!",
                on_click=unset_download_data,
                kwargs={"key": judgment_data_key},
            )
    
    with cols[-1]:
        judgment_data_key_all = "judgment_data_key_all"
        st.markdown(
            "Download the results of all files"
        )
        if judgment_data_key_all not in st.session_state:
            st.button(
                label="Generate Data for All Files",
                on_click=prepare_download_all,
                kwargs={"key": judgment_data_key_all},
                help="Click to build the Excel file before downloading"
            )
        else:
            st.download_button(
                label=":arrow_down: Download All Judgment Data",
                data=st.session_state[judgment_data_key_all],
                file_name="all_judgment_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet",
                key="download_all_llm_judgement",
                help="Your Excel file is ready—click to save it locally!",
                on_click=unset_download_data,
                kwargs={"key": judgment_data_key_all},
            )
    


def main():
    start_time = time.time()
    st.title("📝 Urdu NER LLM-As-A-Judge")
    st.markdown("""
    This page allows you to evaluate the performance of different LLMs on Urdu NER tasks.
    You can select the models you want to compare and run the evaluation.
    The evaluation results will be displayed below.
    """)
    st.markdown("**Note:** You can select or deselect models to include in the comparison.")
    st.markdown("---")
    
    if add_entity_status():
        
        set_judgment_configuration()

        if selected_models := st.session_state.get('selected_models'):
            st.markdown("**Selected Models**")
            for model in selected_models:
                st.markdown(f"- {model}")
            st.markdown("**Note:** You can select or deselect models to include in the comparison.")
            st.markdown("---")
        
        if has_judgment_data():
            show_results()
            st.markdown("---")
            download_data()
        else:
            evaluate_models()
        print("Judgment page loaded in", time.time() - start_time, "seconds.")

main()