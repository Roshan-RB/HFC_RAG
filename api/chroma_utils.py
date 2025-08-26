from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
# NEW imports
import asyncio
from docling_pipeline import (
    convert_with_docling,
    docling_to_blocks,
    smart_chunk,
    enrich_blocks_with_llm,
    export_for_embedding,
)

# --- one-time setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY,
            model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

def _docling_prepare(file_path: str):
    """
    Docling -> blocks -> smart chunks -> LLM enrichment (captions/summaries) -> docs/metas
    (All synchronous to avoid event-loop conflicts.)
    """
    doc = convert_with_docling(file_path)
    blocks = docling_to_blocks(doc, file_path)
    chunks = smart_chunk(blocks, max_chars=1500)
    enriched = enrich_blocks_with_llm(chunks, model="gpt-4o-mini")
    return export_for_embedding(enriched)  # -> (docs, metas)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """
    Use Docling for ALL supported docs (pdf, docx, html, pptx).
    Store rich metadata so your retriever can render tables/figures later.
    """
    try:
        documents, metadatas = _docling_prepare(file_path)

        # Attach file_id to each metadata
        for m in metadatas:
            m["file_id"] = file_id

        vectorstore.add_texts(texts=documents, metadatas=metadatas)
        return True
    except Exception as e:
        print(f"Error indexing document with Docling: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False
