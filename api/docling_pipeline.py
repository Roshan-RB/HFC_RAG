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

# OpenAI (for captions on figures) via LangChain (optional)
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


def convert_with_docling(src_path: str, enable_ocr_fallback: bool = True):
    """
    For PDFs, pass PdfPipelineOptions via PdfFormatOption.
    Enable image/table image generation; try OCR fallback if the first pass yields no pages/elements.
    """
    def _make_converter(with_ocr: bool):
        if Path(src_path).suffix.lower() == ".pdf":
            opts = PdfPipelineOptions()

            # OCR toggle
            if hasattr(opts, "do_ocr"):
                setattr(opts, "do_ocr", with_ocr)

            # Better table extraction
            if hasattr(opts, "do_table_structure"):
                setattr(opts, "do_table_structure", True)

            # Generate images for figures/tables; pages optional
            if hasattr(opts, "generate_picture_images"):
                opts.generate_picture_images = True
            if hasattr(opts, "generate_table_images"):
                opts.generate_table_images = True
            if hasattr(opts, "generate_page_images"):
                opts.generate_page_images = False  # set True if you also want full-page images
            if hasattr(opts, "images_scale"):
                opts.images_scale = 2.0  # higher resolution

            return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        return DocumentConverter()

    conv = _make_converter(False)
    res = conv.convert(src_path)
    doc = res.document

    if enable_ocr_fallback:
        pages = getattr(doc, "pages", None)
        is_empty = (not pages) or (isinstance(pages, dict) and not pages) or (isinstance(pages, (list, tuple)) and len(pages) == 0)
        if is_empty:
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


def _safe_page_no(obj) -> int:
    for attr in ("page_no", "number", "page_number", "index"):
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                if isinstance(val, int):
                    return val
            except Exception:
                pass
    return -1


def _asset_dirs_for_file(src_path: str):
    """
    Return (asset_base, img_dir, tbl_dir) like:
    extracted_assets/<file_stem>_<mtime>/{images,tables}
    """
    p = Path(src_path)
    stem = p.stem
    try:
        mtime = int(os.path.getmtime(p))
    except Exception:
        mtime = 0
    base = Path("extracted_assets") / f"{stem}_{mtime}"
    img_dir = base / "images"
    tbl_dir = base / "tables"
    img_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    return base, img_dir, tbl_dir


def docling_to_blocks(doc, src_path: str) -> List[EnrichedBlock]:
    """
    Prefer item-level traversal; if nothing extracted, fall back to doc-level Markdown.
    Also saves original figures to PNG and tables to HTML (and CSV when available).
    Assets are stored under extracted_assets/<file_stem>_<mtime>/{images,tables}
    """
    blocks: List[EnrichedBlock] = []

    # Per-file asset folders
    asset_base, img_dir, tbl_dir = _asset_dirs_for_file(src_path)
    extra_common = {"asset_base": str(asset_base)}

    # 1) Try iterating items (tables, pictures, text)
    iter_items = getattr(doc, "iterate_items", None)
    if callable(iter_items):
        try:
            for el, _level in iter_items():
                cls = el.__class__.__name__

                # --------------------------
                # Tables (no LLM summary; keep real content)
                # --------------------------
                if (TableItem is not object and isinstance(el, TableItem)) or "Table" in cls:
                    table_md = ""
                    try:
                        table_md = getattr(el, "markdown", "") or ""
                    except Exception:
                        pass

                    # Preferred: export to HTML
                    table_html = None
                    try:
                        if hasattr(el, "export_to_html"):
                            table_html = el.export_to_html(doc=doc)
                        else:
                            table_html = getattr(el, "html", None)
                    except Exception:
                        table_html = getattr(el, "html", None)

                    # Optional: export to CSV (if DataFrame is available)
                    table_csv_path = None
                    try:
                        if hasattr(el, "export_to_dataframe"):
                            df = el.export_to_dataframe(doc=doc)  # pandas.DataFrame
                            if df is not None:
                                csv_filename = f"{Path(src_path).stem}_table_{len(blocks)+1}.csv"
                                table_csv_path = tbl_dir / csv_filename
                                df.to_csv(table_csv_path, index=False)
                    except Exception:
                        table_csv_path = None

                    # Text to embed: prefer Markdown; else compact HTML string; else placeholder
                    table_text = table_md or (table_html or "").strip() or "[TABLE]"

                    # Save HTML to disk for rendering
                    table_path = None
                    if table_html:
                        table_filename = f"{Path(src_path).stem}_table_{len(blocks)+1}.html"
                        table_path = tbl_dir / table_filename
                        try:
                            table_path.write_text(table_html, encoding="utf-8")
                        except Exception:
                            table_path = None

                    blocks.append(EnrichedBlock(
                        text=table_text,
                        block_type="table",
                        page=_safe_page_no(el),
                        source_path=src_path,
                        extra={
                            **extra_common,
                            "table_html": table_html or "",
                            "table_path": str(table_path) if table_path else None,
                            **({"table_csv_path": str(table_csv_path)} if table_csv_path else {})
                        }
                    ))

                # --------------------------
                # Figures / Pictures
                # --------------------------
                elif (PictureItem is not object and isinstance(el, PictureItem)) or any(x in cls for x in ["Picture", "Figure", "Image"]):
                    caption = getattr(el, "caption", "") or getattr(el, "text", "") or ""
                    text = f"[FIGURE]{(' ' + caption) if caption else ''}" or "[FIGURE]"

                    image_path = None
                    # Preferred: Docling provides PIL image via get_image(document)
                    try:
                        if hasattr(el, "get_image"):
                            pil_img = el.get_image(doc)  # PIL Image
                            image_filename = f"{Path(src_path).stem}_figure_{len(blocks)+1}.png"
                            image_path = img_dir / image_filename
                            pil_img.save(image_path, "PNG")
                    except Exception:
                        image_path = None  # ignore extraction errors

                    blocks.append(EnrichedBlock(
                        text=text,
                        block_type="figure",
                        page=_safe_page_no(el),
                        source_path=src_path,
                        extra={
                            **extra_common,
                            "caption": caption,
                            "image_path": str(image_path) if image_path else None
                        }
                    ))

                # --------------------------
                # Generic text
                # --------------------------
                else:
                    md = getattr(el, "markdown", None) or getattr(el, "text", None) or ""
                    if md and md.strip():
                        blocks.append(EnrichedBlock(
                            text=md.strip(),
                            block_type="text",
                            page=_safe_page_no(el),
                            source_path=src_path,
                            extra={**extra_common}
                        ))
        except Exception:
            pass

    # 2) If items didn’t yield anything, try page-level text
    if not blocks:
        for page in _pages_iter(doc):
            page_no = _safe_page_no(page)
            for attr in ("markdown", "text", "content", "plain_text"):
                val = getattr(page, attr, None)
                if isinstance(val, str) and val.strip():
                    blocks.append(EnrichedBlock(
                        text=val.strip(),
                        block_type="text",
                        page=page_no,
                        source_path=src_path,
                        extra={**extra_common}
                    ))
                    break  # one best string per page

    # 3) Final fallback: whole-document Markdown
    if not blocks:
        md = _export_doc_markdown(doc)
        if md and md.strip():
            blocks.append(EnrichedBlock(text=md.strip(), block_type="text", page=-1,
                                        source_path=src_path, extra={**extra_common, "fallback": "export_to_markdown"}))

    # 4) If truly nothing, put a placeholder (prevents empty upserts)
    if not blocks:
        blocks.append(EnrichedBlock(text="[NO_EXTRACTED_TEXT]", block_type="summary",
                                    page=-1, source_path=src_path, extra={**extra_common, "note": "empty"}))
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
                                         source_path=src, extra=blocks[0].extra if blocks else {}))
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
                                 page=-1, source_path=src, extra=blocks[0].extra if blocks else {}))
    return out


def enrich_blocks_with_llm(chunks: List[EnrichedBlock], model: str = "gpt-4o-mini") -> List[EnrichedBlock]:
    """
    - Figures: always attempt to generate a caption/summary via LLM (default model gpt-4o-mini).
    - Tables: keep original content (no summarization).
    """
    llm: Optional[ChatOpenAI] = _make_llm(model) if model else None
    out: List[EnrichedBlock] = []
    for ch in chunks:
        if ch.block_type == "figure" and llm is not None:
            try:
                cap = caption_image_with_openai(ch.extra.get("image_path", "") or "", llm)
                if cap:
                    ch.text = f"[FIGURE] {cap}"
                    ch.extra["caption"] = cap
            except Exception:
                pass

        # Tables: do not summarise; we keep the actual table content

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
            **({"asset_base": ch.extra.get("asset_base")} if ch.extra.get("asset_base") else {}),
            **({"image_path": ch.extra.get("image_path")} if ch.extra.get("image_path") else {}),
            **({"caption": ch.extra.get("caption")} if ch.extra.get("caption") else {}),
            **({"table_path": ch.extra.get("table_path")} if ch.extra.get("table_path") else {}),
            **({"table_html": ch.extra.get("table_html")} if ch.extra.get("table_html") else {}),
            **({"table_csv_path": ch.extra.get("table_csv_path")} if ch.extra.get("table_csv_path") else {}),
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
