import streamlit as st
import requests
from api_utils import upload_document, list_documents, delete_document

def display_sidebar():
    # Sidebar: Model Selection
    model_options = ["gpt-4o", "gpt-4o-mini"]
    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    # Sidebar: Upload Document
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "html"])
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully with ID {upload_response['file_id']}.")
                    st.session_state.documents = list_documents()  # Refresh the list after upload

    # Sidebar: List Documents
    st.sidebar.header("Uploaded Documents")
    
    if st.sidebar.button("🔄 Sync Now (if uploaded to folder)"):
        try:
            res = requests.post("http://localhost:8000/sync-now", timeout=30)
            res.raise_for_status()
            data = res.json()

            st.sidebar.success(f"Synced folder: {data['watch_dir']}")
            #stats = data.get("stats", {})
            #st.sidebar.write("**Stats:**")
            #st.sidebar.json(stats)  # pretty-print dict
        except Exception as e:
            st.sidebar.error(f"Sync failed: {e}")

    # Initialize document list if not present
    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    documents = st.session_state.documents
    with st.sidebar.expander("List Documents", expanded=False):
        if documents:
            for doc in documents:
                st.text(
                    f"{doc['filename']} (ID: {doc['id']}, Uploaded: {doc['upload_timestamp']})"
                )
            if st.button("Refresh Document List"):
                with st.spinner("Refreshing..."):
                    st.session_state.documents = list_documents()
        else:
            st.info("No documents available.")
        
        # Delete Document
        with st.sidebar.expander("Delete Document", expanded=False):
            selected_file_id = st.selectbox("Select a document to delete", options=[doc['id'] for doc in documents], format_func=lambda x: next(doc['filename'] for doc in documents if doc['id'] == x))
            if st.button("Delete Selected Document"):
                with st.spinner("Deleting..."):
                    delete_response = delete_document(selected_file_id)
                    if delete_response:
                        st.sidebar.success(f"Document with ID {selected_file_id} deleted successfully.")
                        st.session_state.documents = list_documents()  # Refresh the list after deletion
                    else:
                        st.sidebar.error(f"Failed to delete document with ID {selected_file_id}.")
