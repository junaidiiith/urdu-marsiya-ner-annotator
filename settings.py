import streamlit as st


auth_file = 'authentication.yaml'
chroma_data_dir = 'chroma_db_data'
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = st.secrets["OPENAI_LLM"]

CHUNK_SIZE = 4096
CHUNK_OVERLAP = 128

dataset_dir = 'dataset'
MARSIYA_DATASET_DIR = f'{dataset_dir}/marsiya-all'
MAX_CONCURRENT_REQUESTS = 5