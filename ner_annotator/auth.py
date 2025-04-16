import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import LoginError
import yaml
from yaml.loader import SafeLoader
import streamlit as st


def add_authentication(auth_file):
    with open(auth_file) as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    st.session_state['credentials_config'] = config

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        auto_hash=False
    )
    return authenticator, config


def authenticate(auth_file):
    try:
        authenticator, auth_config = add_authentication(auth_file)
        authenticator.login()
    except LoginError as e:
        st.error(e)

    if st.session_state['authentication_status'] is False:
        st.error('Username/password is incorrect')
    elif st.session_state['authentication_status'] is None:
        st.warning('Please enter your username and password')
    elif st.session_state['authentication_status']:
        authenticator.logout(location='sidebar')
        if st.session_state['authentication_status']:
            name = st.session_state['name']
            st.sidebar.title(f'Welcome {name}!')
        else:
            print("Re-running...")
            st.cache_data.clear()
            st.rerun()

    return auth_config
