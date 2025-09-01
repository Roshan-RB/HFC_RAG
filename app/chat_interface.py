# chat_interface.py
import streamlit as st
from api_utils import get_api_response
import os

def _render_sources(sources: list[dict]):
    if not sources:
        return
    st.markdown("**Sources (cited in answer)**")
    for s in sources:
        with st.container(border=True):
            head = f"[S{s.get('rank','?')}] **{s.get('file_name','?')}**"
            fid = s.get("file_id")
            if fid is not None:
                head += f" (file_id `{fid}`)"
            st.markdown(head)

            bits = []
            if s.get("block_type"): bits.append(f"type: `{s['block_type']}`")
            if s.get("block_id"):   bits.append(f"block: `{s['block_id']}`")
            if s.get("page") is not None: bits.append(f"page: `{s['page']}`")
            if bits: st.caption(" · ".join(bits))

            prev = s.get("original_text_preview")
            if prev: st.write(f"> {prev}")

            # Show assets only for the right block type
            btype = s.get("block_type")
            img_path = s.get("image_path")
            if btype == "figure" and img_path and os.path.exists(img_path):
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

            if btype == "table":
                assets = []
                if s.get("table_path"): assets.append(f"html: `{s['table_path']}`")
                if s.get("table_csv_path"): assets.append(f"csv: `{s['table_csv_path']}`")
                if assets: st.code(" | ".join(assets), language="text")

            # Full original chunk
            full_text = s.get("text")
            if full_text:
                with st.expander("Original chunk"):
                    st.code(full_text)

def display_chat_interface():
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Render sources for assistant messages if present
            if message["role"] == "assistant" and message.get("sources"):
                _render_sources(message["sources"])

    if prompt := st.chat_input("Query:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)

            if response:
                st.session_state.session_id = response.get('session_id')
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response.get('sources', [])  # store alongside content
                })

                with st.chat_message("assistant"):
                    st.markdown(response['answer'])
                    _render_sources(response.get('sources', []))

                    with st.expander("Details"):
                        st.subheader("Generated Answer")
                        st.code(response['answer'])
                        st.subheader("Model Used")
                        st.code(response['model'])
                        st.subheader("Session ID")
                        st.code(response['session_id'])
            else:
                st.error("Failed to get a response from the API. Please try again.")
