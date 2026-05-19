"""
ai_core.py — Production-Ready AI OCR Packing List Extraction Engine
====================================================================
Author  : Senior AI Engineer
Purpose : Extract structured packing list data from images / PDFs
          using an OCR pre-pass and OpenAI structured completion.
Usage   :
    extractor = PackingListAI()
    result    = extractor.process_file("sample.pdf")
    print(result)           # dict  — use directly in FastAPI / Django
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PackingListAI")


# ---------------------------------------------------------------------------
# Pydantic Schema Models
# ---------------------------------------------------------------------------


class PackingItem(BaseModel):
    """A single item inside a packing-list category."""

    title: str = Field(..., description="Normalised item name")
    quantity: Optional[int] = Field(None, description="Numeric quantity, null if unknown")
    is_required: bool = Field(True, description="True when the item is mandatory")
    note: Optional[str] = Field(None, description="Free-text note or qualifier")

    @field_validator("title")
    @classmethod
    def normalise_title(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("quantity", mode="before")
    @classmethod
    def coerce_quantity(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


class PackingCategory(BaseModel):
    """A logical grouping of packing items."""

    name: str = Field(..., description="Category label in UPPER CASE")
    sort_order: int = Field(0, description="Display order index")
    items: list[PackingItem] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def normalise_name(cls, v: str) -> str:
        return v.strip().upper()


class PackingList(BaseModel):
    """Root schema for a fully extracted packing list document."""

    title: str = Field(..., description="Document title")
    description: Optional[str] = Field(None)
    season: Optional[str] = Field(None)
    trip_type: Optional[str] = Field(None)
    is_system: bool = Field(True)
    categories: list[PackingCategory] = Field(default_factory=list)

    @model_validator(mode="after")
    def assign_sort_orders(self) -> "PackingList":
        for idx, cat in enumerate(self.categories):
            cat.sort_order = idx
        return self


# ---------------------------------------------------------------------------
# Image Preprocessor
# ---------------------------------------------------------------------------


class ImagePreprocessor:
    """
    Applies a series of OpenCV transforms to maximise Tesseract accuracy
    on real-world, noisy document scans.
    """

    def __init__(
        self,
        dpi_target: int = 300,
        denoise: bool = True,
        deskew: bool = True,
    ) -> None:
        self.dpi_target = dpi_target
        self.denoise = denoise
        self.deskew = deskew

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Return a cleaned PIL Image ready for OCR."""
        img = self._pil_to_cv(image)
        img = self._to_grayscale(img)
        img = self._scale_if_needed(img)
        if self.denoise:
            img = self._denoise(img)
        img = self._threshold(img)
        if self.deskew:
            img = self._deskew(img)
        return self._cv_to_pil(img)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_cv(image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _cv_to_pil(img: np.ndarray) -> Image.Image:
        if len(img.shape) == 2:  # grayscale
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _scale_if_needed(self, img: np.ndarray) -> np.ndarray:
        """Upscale small images to improve OCR accuracy."""
        h, w = img.shape[:2]
        min_dim = 1200  # heuristic for ~300 DPI letter-size page
        if max(h, w) < min_dim:
            scale = min_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            logger.debug("Scaled image by %.2f×", scale)
        return img

    @staticmethod
    def _denoise(img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(img, h=10)

    @staticmethod
    def _threshold(img: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )

    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        """Rotate the image to correct skew using the Hough line transform."""
        coords = np.column_stack(np.where(img > 0))
        if coords.size == 0:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return img
        h, w = img.shape[:2]
        centre = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Document Loader  (image + multi-page PDF)
# ---------------------------------------------------------------------------


class DocumentLoader:
    """
    Converts any supported input file (image or PDF) into a list of
    PIL Image objects — one per page / frame.
    """

    SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    SUPPORTED_PDF_EXT = {".pdf"}

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def load(self, file_path: Union[str, Path]) -> list[Image.Image]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()

        if ext in self.SUPPORTED_IMAGE_EXT:
            return self._load_image(path)
        if ext in self.SUPPORTED_PDF_EXT:
            return self._load_pdf(path)

        raise ValueError(f"Unsupported file type '{ext}'. Supported: {self.SUPPORTED_IMAGE_EXT | self.SUPPORTED_PDF_EXT}")

    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> list[Image.Image]:
        logger.info("Loading image: %s", path.name)
        img = Image.open(path).convert("RGB")
        return [img]

    def _load_pdf(self, path: Path) -> list[Image.Image]:
        logger.info("Loading PDF: %s (%s pages)", path.name, self._pdf_page_count(path))
        doc = fitz.open(str(path))
        pages: list[Image.Image] = []
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 DPI is PDF default
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes("png")
            pages.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
            logger.debug("  Rendered PDF page %d", page_num + 1)
        doc.close()
        return pages

    @staticmethod
    def _pdf_page_count(path: Path) -> int:
        doc = fitz.open(str(path))
        n = doc.page_count
        doc.close()
        return n


# ---------------------------------------------------------------------------
# OCR Engine
# ---------------------------------------------------------------------------


class OCREngine:
    """
    Wraps Tesseract via pytesseract to extract raw text from preprocessed
    page images.  Returns a single consolidated text string.
    """

    TESSERACT_CONFIG = "--oem 3 --psm 6"  # LSTM engine, assume uniform block text

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang
        self.preprocessor = ImagePreprocessor()

    def extract_text(self, pages: list[Image.Image]) -> str:
        """Return concatenated OCR text for all pages."""
        parts: list[str] = []
        for i, page in enumerate(pages):
            logger.info("OCR — processing page %d / %d", i + 1, len(pages))
            clean = self.preprocessor.preprocess(page)
            text = pytesseract.image_to_string(clean, lang=self.lang, config=self.TESSERACT_CONFIG)
            parts.append(text)
        combined = "\n\n---PAGE BREAK---\n\n".join(parts)
        logger.debug("OCR raw output length: %d chars", len(combined))
        return combined

    def extract_text_bytes(self, image_bytes: bytes, mime: str = "image/png") -> str:
        """Convenience: accept raw bytes instead of a PIL Image."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.extract_text([img])


# ---------------------------------------------------------------------------
# OpenAI Structured Extraction Service
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """
You are an expert document parser specialised in packing lists for camps,
travel, schools, and expeditions.

Your job:
1. Read the raw OCR text provided by the user.
2. Extract ALL items, categories, quantities, and notes.
3. Return ONLY a valid JSON object matching the schema below — no markdown fences,
   no commentary, just the raw JSON.

JSON Schema:
{
  "title":       string,
  "description": string | null,
  "season":      string | null,
  "trip_type":   string | null,
  "is_system":   true,
  "categories": [
    {
      "name":       string,       // UPPER CASE label
      "sort_order": integer,
      "items": [
        {
          "title":       string,
          "quantity":    integer | null,
          "is_required": boolean,
          "note":        string | null
        }
      ]
    }
  ]
}

Rules:
- Infer categories from context (CLOTHING, TOILETRIES, LINENS, UNIFORMS, EQUIPMENT,
  ACCESSORIES, MISCELLANEOUS, or any other logical group you detect).
- Normalise item names (title case, no stray symbols).
- Parse quantities written as words ("two", "a pair", "1 pr") → integers.
- If quantity is ambiguous or absent, use null.
- Mark items as is_required: false only if the text explicitly says optional /
  recommended / suggested.
- Ignore headers, footers, page numbers, and decorative text.
- Consolidate duplicate items across pages.
- Produce clean, backend-ready JSON.
""".strip()


class OpenAIExtractionService:
    """
    Sends OCR text + optionally a vision image to OpenAI and returns a
    validated PackingList instance.
    """

    MODEL = "gpt-4o"
    MAX_TOKENS = 4096

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file or pass api_key=."
            )
        self.client = OpenAI(api_key=key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        ocr_text: str,
        vision_pages: Optional[list[Image.Image]] = None,
    ) -> PackingList:
        """
        Run extraction.  If vision_pages is provided the first page is sent
        alongside the OCR text for multimodal context.
        """
        messages = self._build_messages(ocr_text, vision_pages)
        logger.info("Sending request to OpenAI (%s)…", self.MODEL)
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            max_tokens=self.MAX_TOKENS,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw_json = response.choices[0].message.content or "{}"
        logger.info("OpenAI response received (%d chars)", len(raw_json))
        return self._parse_and_validate(raw_json)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        ocr_text: str,
        vision_pages: Optional[list[Image.Image]],
    ) -> list[dict]:
        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"Here is the OCR-extracted text from the packing list document.\n\n"
                    f"<ocr_text>\n{ocr_text[:12_000]}\n</ocr_text>\n\n"
                    "Please extract the structured packing list JSON."
                ),
            }
        ]

        # Attach first page as vision context when available
        if vision_pages:
            b64 = self._pil_to_b64(vision_pages[0])
            user_content.insert(
                0,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                },
            )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _pil_to_b64(image: Image.Image, max_size: int = 1500) -> str:
        """Resize + encode PIL image to base64 PNG string."""
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _parse_and_validate(raw_json: str) -> PackingList:
        """Parse raw JSON string and validate against Pydantic schema."""
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.error("JSON decode failed: %s\nRaw: %s", exc, raw_json[:500])
            raise ValueError(f"OpenAI returned invalid JSON: {exc}") from exc

        try:
            return PackingList(**data)
        except Exception as exc:
            logger.error("Pydantic validation failed: %s", exc)
            raise ValueError(f"Schema validation error: {exc}") from exc


# ---------------------------------------------------------------------------
# Response Formatter
# ---------------------------------------------------------------------------


class ResponseFormatter:
    """
    Converts a validated PackingList model to a plain Python dict or
    a formatted JSON string suitable for API responses / logging.
    """

    @staticmethod
    def to_dict(packing_list: PackingList) -> dict:
        return packing_list.model_dump(mode="json")

    @staticmethod
    def to_json(packing_list: PackingList, indent: int = 2) -> str:
        return json.dumps(ResponseFormatter.to_dict(packing_list), indent=indent, ensure_ascii=False)

    @staticmethod
    def summary(packing_list: PackingList) -> str:
        total_items = sum(len(c.items) for c in packing_list.categories)
        lines = [
            f"Title       : {packing_list.title}",
            f"Season      : {packing_list.season or '—'}",
            f"Categories  : {len(packing_list.categories)}",
            f"Total items : {total_items}",
        ]
        for cat in packing_list.categories:
            lines.append(f"  [{cat.name}] — {len(cat.items)} items")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PackingListAI  — Main Facade (public SDK entry-point)
# ---------------------------------------------------------------------------


class PackingListAI:
    """
    Production-ready facade for the packing list extraction pipeline.

    Usage (backend integration)::

        extractor = PackingListAI()
        result = extractor.process_file("sample.pdf")
        # result is a plain dict — drop straight into FastAPI / Django response

    Optional kwargs:
        api_key         — OpenAI key (defaults to OPENAI_API_KEY env var)
        use_vision      — send first page as vision context (default True)
        ocr_lang        — Tesseract language code (default "eng")
        return_model    — return Pydantic model instead of dict (default False)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_vision: bool = True,
        ocr_lang: str = "eng",
    ) -> None:
        self.use_vision = use_vision
        self.loader = DocumentLoader()
        self.ocr = OCREngine(lang=ocr_lang)
        self.extractor = OpenAIExtractionService(api_key=api_key)
        self.formatter = ResponseFormatter()
        logger.info("PackingListAI initialised (vision=%s, lang=%s)", use_vision, ocr_lang)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(
        self,
        file_path: Union[str, Path],
        return_model: bool = False,
    ) -> Union[dict, PackingList]:
        """
        Full pipeline: load → OCR → AI extraction → validate → format.

        Args:
            file_path    : Path to a PDF or image file.
            return_model : If True, return the Pydantic PackingList model.
                           If False (default), return a plain dict.

        Returns:
            dict (or PackingList) with the extracted packing list data.

        Raises:
            FileNotFoundError : file_path does not exist.
            ValueError        : unsupported file type or extraction failed.
            EnvironmentError  : OPENAI_API_KEY missing.
        """
        path = Path(file_path)
        logger.info("=== Processing file: %s ===", path.name)

        # Stage 1 — Load document into page images
        pages = self.loader.load(path)
        logger.info("Loaded %d page(s)", len(pages))

        # Stage 2 — OCR
        ocr_text = self.ocr.extract_text(pages)

        # Stage 3 — AI structured extraction
        vision_pages = pages if self.use_vision else None
        packing_list = self.extractor.extract(ocr_text, vision_pages=vision_pages)

        logger.info("Extraction complete.\n%s", self.formatter.summary(packing_list))

        if return_model:
            return packing_list
        return self.formatter.to_dict(packing_list)

    def process_bytes(
        self,
        data: bytes,
        filename: str,
        return_model: bool = False,
    ) -> Union[dict, PackingList]:
        """
        Same pipeline but accepts raw bytes (e.g. from an HTTP upload).

        Args:
            data     : Raw file bytes.
            filename : Original filename — used to determine file type.

        Returns:
            dict (or PackingList) with the extracted packing list data.
        """
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            return self.process_file(tmp_path, return_model=return_model)
        finally:
            tmp_path.unlink(missing_ok=True)

    def process_image_object(
        self,
        image: Image.Image,
        return_model: bool = False,
    ) -> Union[dict, PackingList]:
        """
        Accept a pre-loaded PIL Image directly (useful for in-memory pipelines).
        """
        pages = [image]
        ocr_text = self.ocr.extract_text(pages)
        vision_pages = pages if self.use_vision else None
        packing_list = self.extractor.extract(ocr_text, vision_pages=vision_pages)
        if return_model:
            return packing_list
        return self.formatter.to_dict(packing_list)
