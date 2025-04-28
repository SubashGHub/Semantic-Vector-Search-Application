import time

import requests
import streamlit as st
from pinecone_embed import PineconeOperations

db_pin = PineconeOperations()


def search_vectors(query):
    index_name = "java-q-a"
    result = db_pin.query_index(index_name, query)
    return result


# Title
st.title("Semantic Vector Search")

# Banner Image
st.image("https://assets.zilliz.com/Group_13360_96bc801980.png", width=300)

# Description
st.markdown("""
- This Demo allows up to 5 queries using semantic vector search.  
- Type your question below and view the top relevant result.  
- This app will fetch related java answer, which is pre-loaded in the vector database.    
- Ask Questions from the list given.
""")
st.write('---')

# Initialize session state to store interactions
if "interactions" not in st.session_state:
    st.session_state.interactions = []

# Custom styles for buttons
st.markdown("""
<style>
/* Reset button (Red) */
div.stButton > button {
    width: 100%;
   
    color: white;
    border: none;
    margin-top: 10px;
}

/* Style for the 'Reset' button (assigned type="secondary") */
div.stButton > button[kind="secondary"] {
    background-color: red !important; /* Important overrides Streamlit's default */
}

/* Optional: Style for Reset button hover state */
div.stButton > button[kind="secondary"]:hover {
    background-color: darkred !important;
    color: white !important; /* Ensure text color stays white */
    border-color: darkred !important; /* Match border color */
}

/* Optional: Style for Reset button focus state */
div.stButton > button[kind="secondary"]:focus:not(:active) {
    background-color: darkred !important;
    color: white !important;
    border-color: darkred !important;
}


/* Style for the 'Search' button (assigned type="primary") */
div.stButton > button[kind="primary"] {
    background-color: #1e88e5 !important; /* Important overrides Streamlit's default */
}

/* Optional: Style for Search button hover state */
div.stButton > button[kind="primary"]:hover {
    background-color: #1E90FF !important;
    color: white !important; /* Ensure text color stays white */
    border-color: darkblue !important; /* Match border color */
}

/* Optional: Style for Search button focus state */
div.stButton > button[kind="primary"]:focus:not(:active) {
    background-color: #0077b6 !important;
    color: white !important;
    border-color: darkblue !important;
}
</style>
""", unsafe_allow_html=True)

# Reset button
rst = st.button("Reset", type='secondary')

if rst:
    st.success("Reset Done.")
    st.session_state.interactions = []
    time.sleep(1)
    st.rerun()

# input box
if len(st.session_state.interactions) < 5:
    query = st.text_input("Ask the Java Questions here:", key=f"query_{len(st.session_state.interactions)}")
    st.button("Search", type='primary')
    # st.toast("Searching...")

    if query:
        # Show loader spinner while "searching"
        with st.spinner("Searching..."):
            result = search_vectors(query)

        # Store the query and result
        st.session_state.interactions.append({
            "query": query,
            "response": result
        })

        # Clear the input (force rerun)
        st.rerun()

# Display past interactions
if st.session_state.interactions:
    st.subheader("Your Queries and Results")

    for idx, interaction in enumerate(st.session_state.interactions, 1):
        with st.expander(f"{idx} . {interaction['query']}"):
            st.markdown(
                f"<span style='color: #2e7d32; font-weight: bold;'>→ Response:</span> "
                f"<span style='color: white;'>{interaction['response']}</span>",
                unsafe_allow_html=True
            )

# If limit reached
if len(st.session_state.interactions) >= 5:
    st.warning("You have reached the maximum of 5 queries.\n Please click on Reset Button.")

