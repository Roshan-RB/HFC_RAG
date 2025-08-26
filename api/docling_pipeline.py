# docling_pipeline.py
import os
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Docling (v2.x)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# Optional type names (avoid hard dep breakage)
try:
    from docling_core.types.doc import PictureItem, TableItem
except Exception:  # fallbacks if API surface changes
    PictureItem, TableItem = object, object

# OpenAI (for captions/table summaries) via LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


@dataclass
class EnrichedBlock:
    text: str
    block_type: str             # "text" | "table" | "figure" | "heading" | "summary"
    page: int
    source_path: str
    extra: Dict[str, Any]


def _make_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(model=model)


def _guess_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".png": "image/png",
    }.get(ext, "image/png")


def _b64_image(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def caption_image_with_openai(image_path: str, llm: ChatOpenAI) -> Optional[str]:
    b64 = _b64_image(image_path)
    if not b64:
        return None
    content = [
        {"type": "text", "text": "Describe this figure concisely for retrieval. Mention key entities/values."},
        {"type": "image_url", "image_url": {"url": f"data:{_guess_mime(image_path)};base64,{b64}"}}
    ]
    resp = llm.invoke([HumanMessage(content=content)])
    return getattr(resp, "content", None)


def summarize_table_with_openai(table_md_or_html: str, llm: ChatOpenAI) -> str:
    prompt = (
        "Summarize the key information in this table in 1–3 sentences for search/retrieval. "
        "Do not invent values or columns.\n\n" + table_md_or_html[:8000]
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", "")


def convert_with_docling(src_path: str, enable_ocr_fallback: bool = True):
    """
    For PDFs, pass PdfPipelineOptions via PdfFormatOption.
    Try OCR fallback if the first pass yields no pages/elements.
    """
    def _make_converter(with_ocr: bool):
        if Path(src_path).suffix.lower() == ".pdf":
            opts = PdfPipelineOptions()
            if hasattr(opts, "do_ocr"):
                setattr(opts, "do_ocr", with_ocr)
            if hasattr(opts, "do_table_structure"):
                setattr(opts, "do_table_structure", True)
            return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        return DocumentConverter()

    conv = _make_converter(False)
    res = conv.convert(src_path)
    doc = res.document

    if enable_ocr_fallback:
        pages = getattr(doc, "pages", None)
        if not pages or (isinstance(pages, dict) and not pages) or (isinstance(pages, (list, tuple)) and len(pages) == 0):
            try:
                conv2 = _make_converter(True)
                res2 = conv2.convert(src_path)
                if getattr(res2.document, "pages", None):
                    return res2.document
            except Exception:
                pass
    return doc


def _pages_iter(doc) -> List[Any]:
    pages = getattr(doc, "pages", None)
    if not pages:
        return []
    if isinstance(pages, dict):
        return list(pages.values())
    if isinstance(pages, (list, tuple)):
        return list(pages)
    return []


def _export_doc_markdown(doc) -> Optional[str]:
    # Preferred API (per official docs): export_to_markdown()
    if hasattr(doc, "export_to_markdown"):
        try:
            md = doc.export_to_markdown()
            if isinstance(md, str) and md.strip():
                return md
        except Exception:
            pass
    # Fallback: save_as_markdown to temp file, then read it
    tmp = None
    try:
        if hasattr(doc, "save_as_markdown"):
            tmp = Path.cwd() / "__docling_tmp_export.md"
            doc.save_as_markdown(tmp)
            if tmp.exists():
                return tmp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    finally:
        try:
            if tmp and tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    return None


def docling_to_blocks(doc, src_path: str) -> List[EnrichedBlock]:
    """
    Prefer item-level traversal; if nothing extracted, fall back to doc-level Markdown.
    """
    blocks: List[EnrichedBlock] = []

    # 1) Try iterating items (tables, pictures, text)
    iter_items = getattr(doc, "iterate_items", None)
    if callable(iter_items):
        try:
            for el, _level in iter_items():
                cls = el.__class__.__name__
                # Tables
                if (TableItem is not object and isinstance(el, TableItem)) or "Table" in cls:
                    table_md = getattr(el, "markdown", None) or ""
                    table_html = getattr(el, "html", None)
                    text = table_md or "[TABLE]"
                    blocks.append(EnrichedBlock(text=text, block_type="table", page=-1, source_path=src_path,
                                                extra={"table_html": table_html or ""}))
                # Figures / Pictures
                elif (PictureItem is not object and isinstance(el, PictureItem)) or any(x in cls for x in ["Picture", "Figure", "Image"]):
                    caption = getattr(el, "caption", "") or getattr(el, "text", "") or ""
                    text = f"[FIGURE]{(' ' + caption) if caption else ''}" or "[FIGURE]"
                    blocks.append(EnrichedBlock(text=text, block_type="figure", page=-1, source_path=src_path,
                                                extra={"caption": caption}))
                else:
                    md = getattr(el, "markdown", None) or getattr(el, "text", None) or ""
                    if md and md.strip():
                        blocks.append(EnrichedBlock(text=md.strip(), block_type="text", page=-1,
                                                    source_path=src_path, extra={}))
        except Exception:
            pass

    # 2) If items didn’t yield anything, try page-level text
    if not blocks:
        for page in _pages_iter(doc):
            page_no = getattr(page, "page_no", getattr(page, "number", -1))
            for attr in ("markdown", "text", "content", "plain_text"):
                val = getattr(page, attr, None)
                if isinstance(val, str) and val.strip():
                    blocks.append(EnrichedBlock(text=val.strip(), block_type="text",
                                                page=page_no, source_path=src_path, extra={}))
                    break  # one best string per page

    # 3) Final fallback: whole-document Markdown
    if not blocks:
        md = _export_doc_markdown(doc)
        if md and md.strip():
            blocks.append(EnrichedBlock(text=md.strip(), block_type="text", page=-1,
                                        source_path=src_path, extra={"fallback": "export_to_markdown"}))

    # 4) If truly nothing, put a placeholder (prevents empty upserts)
    if not blocks:
        blocks.append(EnrichedBlock(text="[NO_EXTRACTED_TEXT]", block_type="summary",
                                    page=-1, source_path=src_path, extra={"note": "empty"}))
    return blocks


def smart_chunk(blocks: List[EnrichedBlock], max_chars: int = 1500) -> List[EnrichedBlock]:
    out: List[EnrichedBlock] = []
    buf: List[str] = []
    buf_pages: List[int] = []
    src = blocks[0].source_path if blocks else ""

    def flush():
        if buf:
            joined = "\n\n".join(x for x in buf if x and x.strip())
            if joined.strip():
                out.append(EnrichedBlock(text=joined, block_type="text",
                                         page=min(buf_pages) if buf_pages else -1,
                                         source_path=src, extra={}))
            buf.clear(); buf_pages.clear()

    for b in blocks:
        if b.block_type in {"table", "figure"}:
            if not (b.text and b.text.strip()):
                b.text = "[TABLE]" if b.block_type == "table" else "[FIGURE]"
            flush(); out.append(b)
        else:
            if sum(len(t) for t in buf) + len(b.text or "") > max_chars:
                flush()
            if b.text and b.text.strip():
                buf.append(b.text); buf_pages.append(b.page)
    flush()

    if not out:
        out.append(EnrichedBlock(text="[EMPTY_DOCUMENT_PLACEHOLDER]", block_type="summary",
                                 page=-1, source_path=src, extra={"note": "no chunks"}))
    return out


def enrich_blocks_with_llm(chunks: List[EnrichedBlock], model: str = "gpt-4o-mini") -> List[EnrichedBlock]:
    llm = _make_llm(model)
    out: List[EnrichedBlock] = []
    for ch in chunks:
        if ch.block_type == "figure" and not ch.extra.get("caption"):
            try:
                cap = caption_image_with_openai(ch.extra.get("image_path", "") or "", llm)
                if cap:
                    ch.text = f"[FIGURE] {cap}"; ch.extra["caption"] = cap
            except Exception:
                pass
        elif ch.block_type == "table":
            table_text = ch.extra.get("table_html") or ch.text
            if table_text and table_text.strip() and table_text != "[TABLE]":
                try:
                    summary = summarize_table_with_openai(table_text, llm)
                    if summary:
                        ch.text = f"{summary}\n\n{ch.text}"; ch.extra["table_summary"] = summary
                except Exception:
                    pass
        if not (ch.text and ch.text.strip()):
            ch.text = "[PLACEHOLDER_CHUNK]"
        out.append(ch)
    return out


def export_for_embedding(chunks: List[EnrichedBlock]):
    docs, metas = [], []
    for ch in chunks:
        txt = (ch.text or "").strip()
        if not txt:
            continue
        metas.append({
            "block_type": ch.block_type,
            "page": ch.page,
            "source_path": ch.source_path,
            **({"image_path": ch.extra.get("image_path")} if ch.extra.get("image_path") else {}),
            **({"caption": ch.extra.get("caption")} if ch.extra.get("caption") else {}),
            **({"table_html": ch.extra.get("table_html")} if ch.extra.get("table_html") else {}),
            **({"table_summary": ch.extra.get("table_summary")} if ch.extra.get("table_summary") else {}),
        })
        docs.append(txt)

    if not docs:
        docs = ["[NO_TEXT_EXTRACTED_PLACEHOLDER]"]
        metas = [{
            "block_type": "summary",
            "page": -1,
            "source_path": chunks[0].source_path if chunks else "",
            "note": "auto-placeholder-no-text"
        }]
    return docs, metas
