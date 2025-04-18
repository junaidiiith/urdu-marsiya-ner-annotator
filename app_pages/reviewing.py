import streamlit as st
import re
from app_pages.common import (
    download_ner_tags_data,
    extract_entities, 
    get_current_data, 
    add_entity_status, 
    get_current_text_hash
)

from ner_annotator.stats import get_stats
from ner_annotator.utils import save_file_data


# Color map for entity types
TAG_COLORS = {
    "PERSON": "#AED6F1",  # Light Blue
    "LOCATION": "#A9DFBF",  # Light Green
    "DATE": "#FCF3CF",  # Light Yellow
    "TIME": "#FADBD8",  # Light Pink
    "ORGANIZATION": "#F9E79F",  # Light Orange
    "DESGINATION": "#D5DBDB",  # Light Gray
    "NUMBER": "#D7BDE2",  # Light Purple
}

# Entity tags
TAGS = list(TAG_COLORS.keys())

st.markdown(
    """
    <style>
    /* target only our Clear button by wrapping it in .delete-btn-container */
    .delete-btn-container > button {
        background-color: #e74c3c !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5em 1em !important;
        font-size: 1em !important;
    }
    .delete-btn-container > button:hover {
        background-color: #c0392b !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def get_current_line():
    tagged_data = get_current_data()["tagged_elements"]
    return tagged_data[st.session_state.get("current_line") - 1]


def set_current_line(line_number, content):
    current_hash = get_current_text_hash()
    st.session_state[current_hash]["tagged_elements"][line_number - 1]["tagged"] = content


def get_current_entities_status():
    return get_current_line()["entity_status"]


def set_current_entities_status(entities_status):
    current_line = get_current_line()
    current_line["entity_status"] = entities_status
    current_hash = get_current_text_hash()
    
    st.session_state[current_hash]["tagged_elements"][
        st.session_state.get("current_line") - 1
    ] = current_line


def set_new_entity_tag_current_entities_status(entity, new_tag):
    current_entities_status = get_current_entities_status()
    if entity in current_entities_status:
        current_entities_status[entity].update(
            {
                "user_updated": new_tag,
            }
        )
        set_current_entities_status(current_entities_status)
    else:
        current_entities_status[entity] = {
            "entity": entity,
            "tag": None,
            "user_updated": new_tag,
        }


def render_tagged_text():
    display_text: str = get_current_line()["tagged"]
    entities = extract_entities(display_text)
    for tag, text in entities:
        color = TAG_COLORS.get(tag.upper(), "#E5E7E9")  # Default light grey
        replacement = f'<span style="background-color: {color}; padding: 2px;" title="{tag}">{text}</span>'
        display_text = display_text.replace(f"<{tag}>{text}</{tag}>", replacement)
    st.markdown(display_text, unsafe_allow_html=True)


def set_new_tag(entity, old_tag, new_tag):
    tagged_text: str = get_current_line()["tagged"]
    pattern = f"<{old_tag}>{entity}</{old_tag}>" if old_tag else entity
    replacement = f"<{new_tag}>{entity}</{new_tag}>"
    set_current_line(
        st.session_state["current_line"], tagged_text.replace(pattern, replacement)
    )
    set_new_entity_tag_current_entities_status(entity, new_tag)
    save_tags()
    st.success(f"Tagged '{entity}' as {new_tag}")
    st.rerun()


def save_all_data():
    data = get_current_data()
    text = data["text"]
    save_file_data(text, data)


def save_tags():
    current_entity_status = get_current_entities_status()
    current_entity_status.update(
        {
            "user_verified": True,
        }
    )
    set_current_entities_status(current_entity_status)
    print("Tags at line: ", st.session_state["current_line"], "saved successfully.")
    print("Current Entity Status: ", current_entity_status)
    print("Tagged elements: ", get_current_data()['tagged_elements'][st.session_state["current_line"] - 1])


def remove_newly_added_tag(entity):
    current_entities_status = get_current_entities_status()
    if entity in current_entities_status:
        tag = current_entities_status[entity]["user_updated"]
        del current_entities_status[entity]
        tagged_text = get_current_line()["tagged"]
        pattern = f"<{tag}>{entity}</{tag}>"
        replacement = entity
        print("Ts", tagged_text)
        print("tag", tag)
        set_current_line(
            st.session_state["current_line"], 
            tagged_text.replace(pattern, replacement)
        )
        set_current_entities_status(current_entities_status)
        
        save_tags()
        st.rerun()
        # st.success(f"Removed tag for '{entity}'")
    else:
        st.error(f"No tag found for '{entity}'")
    

def manual_tagging():
    st.subheader("Tag Untagged Words")
    words = re.sub(r"<.*?>", "", get_current_line()["original"]).split()

    # print("Current tagged line: ", get_current_line()["tagged"])
    st.multiselect(
        "Select words to tag", 
        words, 
        placeholder="Select words from dropdown",
        key='manual_tagging_words'
    )
    st.selectbox("Select tag", TAGS, key="new_tag_type")
    if st.button("Add Tag"):
        if st.session_state["manual_tagging_words"]:
            selected_text = " ".join(st.session_state["manual_tagging_words"])
            entities = extract_entities(get_current_line()["tagged"])
            if any(selected_text in entity for _, entity in entities):
                st.warning("This phrase is already tagged.")
            elif selected_text in get_current_line()["original"]:
                new_tag_type = st.session_state["new_tag_type"]
                set_new_tag(selected_text, None, new_tag_type)
                st.session_state["manual_tagging_words"] = []
                st.session_state["new_tag_type"] = None
            else:
                st.error("Selected words are not part of the original text.")


def tags_review():
    # Review tagged entities
    current_entity_status = get_current_entities_status()
    entities = [e for e in current_entity_status if e != 'user_verified']
    cols = st.columns([6, 1])
    with cols[0]:
        st.subheader("Review Existing Tags")
    with cols[1]:
        st.button("Save", key="save_tags", on_click=save_tags)

    if not entities:
        st.write("No tagged entities found in this line.")
        return
    for i, entity in enumerate(entities):
        correct = None
        tag = current_entity_status[entity]["tag"]
        col1, col2, col3 = st.columns([3, 2, 3])
        with col1:
            st.write(f"**{entity}**")
        with col2:
            if current_entity_status[entity]["tag"] is not None:
                tag = current_entity_status[entity]["tag"]
                correct = st.radio(
                    f"Is '{entity}' correctly tagged as {tag}?",
                    ["Yes", "No"],
                    key=f"correct_{i}",
                )
            else:
                ### Show Option to delete the tag
                st.markdown('<div class="delete-btn-container">', unsafe_allow_html=True)
                if st.button("🗑️", key=f"clear_{entity}"):
                    remove_newly_added_tag(entity)
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col3:
            if correct == "No":
                new_tag = st.selectbox(
                    f"Correct tag for '{entity}'", TAGS, key=f"newtag_{i}"
                )
                if st.button("Update Tag", key=f"btn_update_{i}"):
                    set_new_tag(entity, tag, new_tag)


def show_file_statistics():
    stats = get_stats(get_current_data())
    

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
    st.title("📜 LLM-based NER Manual Review")
    if add_entity_status():
        st.subheader("Named Entity Categories")
        legend_html = ""
        for tag, color in TAG_COLORS.items():
            legend_html += f'<span style="background-color: {color}; padding: 4px; margin:4px; border-radius:4px;">{tag}</span> '
        st.markdown(legend_html, unsafe_allow_html=True)
        st.markdown("---")
        max_lines = len(get_current_data()["tagged_elements"])
        st.number_input(
            f"Select Line Number / {max_lines}",
            min_value=1,
            max_value=max_lines,
            value=1,
            key="current_line",
        )
        cols = st.columns([1, 2, 1])
        with cols[0]:
            st.subheader("Original Text")
        with cols[-1]:
            st.button("Save Annotations", key="save_changes", on_click=save_all_data)
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
        
        st.markdown("**Download Review Results**")
        st.markdown("Download the review results as an Excel file.")
        st.download_button(
            ":arrow_down: Download Review Data",
            data=download_ner_tags_data(),
            file_name=f"{st.session_state["selected_model_id"]}_ner_tagged_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_llm_judgement"
        )


main()
