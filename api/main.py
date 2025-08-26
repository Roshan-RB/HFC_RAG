from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
from folder_sync import sync_once, FolderWatcher
logging.basicConfig(filename='app.log', level=logging.INFO)
app = FastAPI()

# NEW: default to your local docs folder
WATCH_DIR = os.getenv("WATCH_DIR", "./docs")
ENABLE_WATCH = os.getenv("ENABLE_WATCH", "true").lower() == "true"
_watcher = None

@app.on_event("startup")
def on_startup():
    # First reconcile everything on disk with DB + Chroma
    stats = sync_once(WATCH_DIR)
    logging.info(f"[startup sync] {stats}")

    # Then start live watcher (created/deleted)
    if ENABLE_WATCH:
        try:
            global _watcher
            _watcher = FolderWatcher(WATCH_DIR)
            _watcher.start()
            logging.info("Folder watcher started")
        except Exception as e:
            logging.exception(f"Failed to start folder watcher: {e}")

@app.on_event("shutdown")
def on_shutdown():
    global _watcher
    if _watcher:
        _watcher.stop()
        _watcher = None

# Optional: manual sync endpoint (handy for a Streamlit button)
@app.post("/sync-now")
def sync_now():
    stats = sync_once(WATCH_DIR)
    return {"watch_dir": WATCH_DIR, "stats": stats}

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())

    

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']
    
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

from fastapi import UploadFile, File, HTTPException
import os
import shutil

@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)
        
        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
    
# --- DEBUG ROUTES (inspect Chroma contents) ---
from typing import Optional
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from chroma_utils import vectorstore


@app.get("/debug/chroma/summary")
def debug_chroma_summary(file_id: Optional[int] = None):
    """
    Count chunks per block_type (text/table/figure/heading/summary) for a file (or all files).
    """
    where_base = {}
    if file_id is not None:
        where_base["file_id"] = file_id

    coll = vectorstore._collection  # underlying Chroma collection
    out = {}
    for t in ["text", "heading", "table", "figure", "summary"]:
        where = dict(where_base)
        where["block_type"] = t
        res = coll.get(where=where, include=[])
        out[t] = len(res.get("ids", []))
    return out

