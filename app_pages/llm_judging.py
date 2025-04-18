import streamlit as st
import pandas as pd
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)

from app_pages.common import (
    download_llm_judgement_data,
    get_current_data, 
    add_entity_status,
    set_text_session_data
)

import numpy as np
import matplotlib.pyplot as plt


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

    # --- 1) Build dataframe of all predictions ---
    rows = st.session_state['evaluated_data']
    df = pd.DataFrame(rows)

    # --- 2) Show total entities count ---
    total_entities = len(df)
    st.metric("Total Entities Evaluated", total_entities)

    # --- 3) Compute metrics per model ---
    model_metrics = {}
    for model_name, grp in df.groupby("model"):
        y_true = [1] * len(grp)
        y_pred = grp["correct"].tolist()
        model_metrics[model_name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    # --- 4) Display each model’s metrics ---
    st.subheader("Model Metrics")
    for model_name, m in model_metrics.items():
        st.markdown(f"**{model_name}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",    f"{m['accuracy']:.2%}")
        c2.metric("Precision",   f"{m['precision']:.2%}")
        c3.metric("Recall",      f"{m['recall']:.2%}")
        c4.metric("F1 Score",    f"{m['f1']:.2%}")
        st.markdown("---")

    # bar‑chart of model metrics
    df_model = pd.DataFrame(model_metrics).T
    models    = df_model.index.tolist()
    metrics_n = df_model.columns.tolist()
    x = np.arange(len(metrics_n))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, model in enumerate(models):
        vals = df_model.loc[model].values
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels(metrics_n)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)

    # --- 5) Compute metrics per tag type ---
    tag_metrics = {}
    for tag, grp in df.groupby("original"):
        y_true = [1] * len(grp)
        y_pred = grp["correct"].tolist()
        tag_metrics[tag] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    # --- 6) Display tag‑level metrics ---
    st.subheader("Metrics by Tag Type")
    for tag, m in tag_metrics.items():
        st.markdown(f"**{tag}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",    f"{m['accuracy']:.2%}")
        c2.metric("Precision",   f"{m['precision']:.2%}")
        c3.metric("Recall",      f"{m['recall']:.2%}")
        c4.metric("F1 Score",    f"{m['f1']:.2%}")
        st.markdown("---")


def main():

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
            st.markdown("**Download Evaluation Results**")
            st.markdown("Download the evaluation results as an Excel file.")
            st.download_button(
                ":arrow_down: Download Evaluation Results",
                data=download_llm_judgement_data(),
                file_name="llm_judgement.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_llm_judgement"
            )
        else:
            evaluate_models()

main()