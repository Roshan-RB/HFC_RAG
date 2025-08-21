import os
from pathlib import Path
from typing import Dict, Set, List

from db_utils import (
    insert_document_record_with_path,
    get_document_by_path,
    get_all_documents_with_path,
    delete_document_record,
)
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma

SUPPORTED_EXTS = {".pdf", ".docx", ".html"}

def _iter_supported_files(root: str) -> List[str]:
    root = os.path.abspath(root)
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED_EXTS:
                out.append(os.path.join(dirpath, f))
    return out

def sync_once(watch_dir: str) -> dict:
    """
    Reconcile by absolute path:
      - For each FS path not in DB: insert (get new file_id) + index to Chroma
      - For each DB row whose path no longer exists: delete from Chroma + DB
    """
    stats = {"added": 0, "deleted": 0, "skipped": 0, "errors": 0}
    os.makedirs(watch_dir, exist_ok=True)

    fs_paths = set(map(os.path.abspath, _iter_supported_files(watch_dir)))
    db_rows = get_all_documents_with_path()
    db_paths = {r["path"] for r in db_rows if r.get("path")}

    # Add new files
    to_add = fs_paths - db_paths
    for p in sorted(to_add):
        try:
            file_id = insert_document_record_with_path(os.path.basename(p), p)
            ok = index_document_to_chroma(p, file_id)
            if ok:
                stats["added"] += 1
            else:
                # rollback DB if indexing failed
                delete_document_record(file_id)
                stats["errors"] += 1
        except Exception:
            stats["errors"] += 1

    # Remove files that disappeared from disk
    path_to_row = {r["path"]: r for r in db_rows if r.get("path")}
    to_delete = db_paths - fs_paths
    for p in sorted(to_delete):
        try:
            row = path_to_row[p]
            fid = int(row["id"])
            delete_doc_from_chroma(fid)
            delete_document_record(fid)
            stats["deleted"] += 1
        except Exception:
            stats["errors"] += 1

    stats["skipped"] = len(fs_paths & db_paths)
    return stats


# --------- Live watcher (created/deleted only) ----------
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class _Handler(FileSystemEventHandler):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def on_created(self, event):
        if event.is_directory:
            return
        ext = Path(event.src_path).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            return
        try:
            abs_path = os.path.abspath(event.src_path)
            if get_document_by_path(abs_path) is not None:
                return  # already known
            file_id = insert_document_record_with_path(os.path.basename(abs_path), abs_path)
            index_document_to_chroma(abs_path, file_id)
        except Exception:
            # optional: log
            pass

    def on_deleted(self, event):
        if event.is_directory:
            return
        try:
            abs_path = os.path.abspath(event.src_path)
            row = get_document_by_path(abs_path)
            if row:
                fid = int(row["id"])
                delete_doc_from_chroma(fid)
                delete_document_record(fid)
        except Exception:
            pass

class FolderWatcher:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.observer = None
        self.handler = _Handler(self.root)

    def start(self):
        if self.observer:
            return
        os.makedirs(self.root, exist_ok=True)
        self.observer = Observer()
        self.observer.schedule(self.handler, self.root, recursive=True)
        self.observer.start()

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2)
            self.observer = None
