# docling_utils.py
import os
from pathlib import Path
from typing import Tuple, Optional, List
import logging

# --- Docling + OpenAI-backed visual description (from your notebook) ---
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode  # optional (for saving MD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# Allow overriding model from env; defaults to your notebook’s model
DOCLING_VISION_MODEL = os.getenv("DOCLING_VISION_MODEL", "gpt-5-mini")

class DoclingProcessingError(RuntimeError):
    pass

def _make_pdf_converter_with_image_annotation() -> DocumentConverter:
    """
    Build a DocumentConverter that:
      - extracts pictures/tables
      - calls OpenAI to describe each image (like your notebook)
      - embeds those descriptions into the document structure
    """
    picture_desc_api_option = PictureDescriptionApiOptions(
        url="https://api.openai.com/v1/chat/completions",
        prompt="Describe this image in sentences in a single paragraph.",
        params=dict(model=DOCLING_VISION_MODEL),
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        timeout=60,
    )

    pipeline_options = PdfPipelineOptions(
        do_picture_description=True,
        picture_description_options=picture_desc_api_option,
        enable_remote_services=True,
        generate_picture_images=True,
        images_scale=2,
    )

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

def docling_to_markdown(file_path: str) -> Tuple[str, Optional[List[str]]]:
    logging.info(f"[Docling] Converting {file_path}")
    try:
        path = Path(file_path)
        if not path.exists():
            raise DoclingProcessingError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            logging.info("[Docling] Using PDF pipeline with image description")
            converter = _make_pdf_converter_with_image_annotation()
            conv_res = converter.convert(source=file_path)
            logging.info("[Docling] Conversion complete, exporting to Markdown")
            md_text = conv_res.document.export_to_markdown(
                mark_annotations=True, include_annotations=True
            )
            logging.info(f"[Docling] Exported Markdown length: {len(md_text)}")
            return md_text, None

        logging.info("[Docling] Using default converter for non-PDF")
        converter = DocumentConverter()
        conv_res = converter.convert(source=file_path)
        md_text = conv_res.document.export_to_markdown(
            mark_annotations=True, include_annotations=True
        )
        logging.info(f"[Docling] Exported Markdown length: {len(md_text)}")
        return md_text, None

    except Exception as e:
        logging.exception("[Docling] ERROR during conversion")
        raise DoclingProcessingError(f"Docling failed for {file_path}: {e}") from e


# Optional helper if you ever want to dump MD with external image refs (not used by ingestion)
def save_markdown_with_image_refs(conv_res, output_dir: str, replace_blank: str = "_") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem.replace(" ", replace_blank)
    md_filename = out_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(
        md_filename, image_mode=ImageRefMode.REFERENCED, include_annotations=True
    )
    return md_filename
