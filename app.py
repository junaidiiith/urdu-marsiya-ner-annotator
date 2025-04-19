import time
import streamlit as st
from ner_annotator.auth import authenticate
from settings import auth_file


def run_pages(pages):
    pg = st.navigation(pages)
    pg.run()

start_time = time.time()
authentication_status = st.session_state.get('authentication_status', None)
authenticate(auth_file=auth_file)
print("Authentication completed in", time.time() - start_time, "seconds.")

if authentication_status:
    username = st.session_state['username']
    name = st.session_state['name']  
    roles = st.session_state['roles']

    pages = [
        st.Page("app_pages/upload_and_tagging.py", title="New Upload and NER Tagging"),
        st.Page("app_pages/reviewing.py", title="LLM NER Tags Reviewing"),
        st.Page("app_pages/llm_judging.py", title="LLM-As-A-Judge")
    ]

    run_pages(pages)
    print("Pages loaded successfully.")
    print("Pages loaded in", time.time() - start_time, "seconds.")
    