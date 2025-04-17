import streamlit as st
import pandas as pd
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)

from app_pages.common import (
    get_data, 
    add_entity_status
)

import numpy as np
import matplotlib.pyplot as plt


from ner_annotator.llm_judge import run_evaluation
from ner_annotator.utils import save_llm_judgement
from settings import SUPPORTED_LLM_JUDGE_MODELS
from stqdm import stqdm


def has_judgment_data():
    data = get_data()
    return all(c in data for c in ['tagged_elements', 'llm_judgement'])


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
    
    if st.session_state.get('evaluated_data'):
        st.balloons()
        return 
    
    if has_judgment_data():
        st.session_state['evaluated_data'] = get_data()['llm_judgement']
        st.success("Judgment data already exists. You can view the results.")
        st.balloons()
        return
        
    if st.button("Run LLM-As-A-Judge Evaluation"):
        tagged_data = get_data()['tagged_elements']
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
            save_llm_judgement(get_data()['text'], st.session_state['evaluated_data'])
            st.balloons()
            st.rerun()


def show_results():
    st.subheader("Evaluation Results")
    rows = st.session_state['evaluated_data']
    df = pd.DataFrame(rows)

    # --- 3) Compute metrics per model ---
    metrics = {}
    for model_name, grp in df.groupby("model"):
        y_true = [1] * len(grp)            # assume every prediction SHOULD be correct
        y_pred = grp["correct"].tolist()  # 1 if model was correct, 0 otherwise

        # avoid division-by-zero if no positive predictions
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        acc       = accuracy_score(y_true, y_pred)

        metrics[model_name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # --- 4) Display each model’s metrics ---
    st.subheader("Model Metrics")
    for model_name, m in metrics.items():
        st.markdown(f"**{model_name}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",    f"{m['accuracy']:.2%}")
        c2.metric("Precision",   f"{m['precision']:.2%}")
        c3.metric("Recall",      f"{m['recall']:.2%}")
        c4.metric("F1 Score",    f"{m['f1']:.2%}")
        st.markdown("---")


    st.subheader("Judgment Comparison Across Different LLMs")

    df_metrics = pd.DataFrame(metrics).T
    models    = df_metrics.index.tolist()
    metrics_n = df_metrics.columns.tolist()
    n_models  = len(models)
    n_metrics = len(metrics_n)

    x     = np.arange(n_metrics)
    width = 0.8 / n_models

    # make the figure smaller: e.g. 6" wide by 3.5" tall
    fig, ax = plt.subplots(figsize=(6, 3.5))

    for i, model in enumerate(models):
        vals = df_metrics.loc[model].values
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics_n)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    st.pyplot(fig)
    


def main():

    if add_entity_status():
        st.title("📝 Urdu NER LLM-As-A-Judge")
        st.markdown("This page uses different LLMs as Judges to evaluate LLM-based on Urdu NER tagging. ")
        set_judgment_configuration()

        if selected_models := st.session_state.get('selected_models'):
            st.markdown("**Selected Models**")
            for model in selected_models:
                st.markdown(f"- {model}")
            st.markdown("**Note:** You can select or deselect models to include in the comparison.")
            st.markdown("---")
        
        if st.session_state.get('evaluated_data', None):
            show_results()
        else:
            evaluate_models()

        st.markdown("---")
main()