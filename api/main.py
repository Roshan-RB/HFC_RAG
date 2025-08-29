from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
from docling_utils import docling_to_markdown, DoclingProcessingError
from chroma_utils import index_markdown_content

logging.basicConfig(filename='app.log', level=logging.INFO)
app = FastAPI()


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
    allowed_extensions = [".pdf", ".docx", ".html"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}",
        )

    # Save upload temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # DB record
    file_id = insert_document_record(file.filename)
    original_filename = file.filename

    try:
        logging.info(f"Starting Docling conversion for file_id={file_id}, path={temp_file_path}")

        # 1) Run Docling → enriched Markdown
        md_text, _assets = docling_to_markdown(temp_file_path)
        logging.info(f"Docling conversion done. Markdown length={len(md_text)} chars")
        # 2) Index Markdown into Chroma
        ok = index_markdown_content(
            file_id=file_id,
            filename=original_filename,
            markdown_text=md_text,
        )
        logging.info(f"Indexing complete. Success={ok}")
        
        if not ok:
            # roll back DB record if desired
            delete_document_record(file_id)
            raise RuntimeError("Failed to index Docling Markdown")

        logging.info(f"Docling ingestion + indexing complete for file_id={file_id}")
        return {"status": "success", "file_id": file_id}

    except DoclingProcessingError as e:
        logging.exception(f"Docling ingestion failed for file_id={file_id}: {e}")
        # Optional fallback to legacy text loader
        ok = index_document_to_chroma(temp_file_path, file_id=file_id)
        if not ok:
            delete_document_record(file_id)
            raise HTTPException(
                status_code=500,
                detail="Failed to index document (Docling + fallback failed).",
            )
        return {"status": "success", "file_id": file_id}

    except Exception as e:
        logging.exception(
            f"Unexpected error during Docling ingestion for file_id={file_id}: {e}"
        )
        delete_document_record(file_id)
        raise HTTPException(
            status_code=500, detail=f"Docling ingestion failed: {str(e)}"
        )

    finally:
        # Always clean temp file
        try:
            os.remove(temp_file_path)
        except Exception:
            pass


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
