"""
ai_core.py — AI OCR Packing List Extraction Engine
====================================================
Pure AI extraction module. No backend code.

Usage:
    extractor = PackingListAI()
    result    = extractor.process_file("sample.pdf")
    print(result)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
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
    is_required: bool = Field(True)
    note: Optional[str] = Field(None)

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
    sort_order: int = Field(0)
    items: list[PackingItem] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def normalise_name(cls, v: str) -> str:
        return v.strip().upper()


class PackingList(BaseModel):
    """Root schema for the full extracted packing list."""

    title: str
    description: Optional[str] = None
    season: Optional[str] = None
    trip_type: Optional[str] = None
    is_system: bool = True
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
    OpenCV pipeline to clean document images before OCR:
    grayscale → upscale → denoise → adaptive threshold → deskew
    """

    def preprocess(self, image: Image.Image) -> Image.Image:
        img = self._to_cv(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._upscale(img)
        img = cv2.fastNlMeansDenoising(img, h=10)
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=10,
        )
        img = self._deskew(img)
        return Image.fromarray(img)

    @staticmethod
    def _to_cv(image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _upscale(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) < 1200:
            scale = 1200 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(img > 0))
        if coords.size == 0:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Document Loader
# ---------------------------------------------------------------------------


class DocumentLoader:
    """Loads a PDF or image file and returns a list of PIL Images (one per page)."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    PDF_EXTS   = {".pdf"}

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def load(self, file_path: Union[str, Path]) -> list[Image.Image]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        ext = path.suffix.lower()
        if ext in self.IMAGE_EXTS:
            return [Image.open(path).convert("RGB")]
        if ext in self.PDF_EXTS:
            return self._pdf_to_images(path)
        raise ValueError(f"Unsupported file type: '{ext}'")

    def _pdf_to_images(self, path: Path) -> list[Image.Image]:
        doc = fitz.open(str(path))
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        doc.close()
        logger.info("PDF loaded — %d page(s)", len(pages))
        return pages


# ---------------------------------------------------------------------------
# OCR Engine
# ---------------------------------------------------------------------------


class OCREngine:
    """Runs Tesseract OCR on preprocessed page images."""

    CONFIG = "--oem 3 --psm 6"

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang
        self.preprocessor = ImagePreprocessor()

    def extract_text(self, pages: list[Image.Image]) -> str:
        parts = []
        for i, page in enumerate(pages):
            logger.info("OCR — page %d / %d", i + 1, len(pages))
            clean = self.preprocessor.preprocess(page)
            text  = pytesseract.image_to_string(clean, lang=self.lang, config=self.CONFIG)
            parts.append(text)
        return "\n\n---PAGE BREAK---\n\n".join(parts)


# ---------------------------------------------------------------------------
# AI Extraction Service
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """
You are an expert document parser for packing lists (camps, travel, schools, expeditions).

Your job:
1. Read the OCR text provided.
2. Extract ALL items, categories, quantities, and notes.
3. Return ONLY a valid JSON object — no markdown, no commentary.

JSON Schema:
{
  "title":       string,
  "description": string | null,
  "season":      string | null,
  "trip_type":   string | null,
  "is_system":   true,
  "categories": [
    {
      "name":       string,
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
- Detect categories from context (CLOTHING, TOILETRIES, LINENS, UNIFORMS, EQUIPMENT, ACCESSORIES, MISCELLANEOUS, or any other logical group).
- Normalise item names to title case, remove stray symbols.
- Parse word quantities ("two", "a pair", "1 pr") → integers.
- Quantity absent or unclear → null.
- is_required: false ONLY when text says optional / recommended / suggested.
- Ignore headers, footers, page numbers, decorative text.
- Consolidate duplicates across pages.
""".strip()


class AIExtractionService:
    """Sends OCR text (+ optional vision page) to OpenAI and returns a validated PackingList."""

    MODEL      = "gpt-4o"
    MAX_TOKENS = 4096

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")
        self.client = OpenAI(api_key=key)

    def extract(self, ocr_text: str, vision_page: Optional[Image.Image] = None) -> PackingList:
        messages = self._build_messages(ocr_text, vision_page)
        logger.info("Calling OpenAI %s…", self.MODEL)
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            max_tokens=self.MAX_TOKENS,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        logger.info("Response received — %d chars", len(raw))
        return self._validate(raw)

    def _build_messages(self, ocr_text: str, vision_page: Optional[Image.Image]) -> list[dict]:
        user_content: list[dict] = [
            {
                "type": "text",
                "text": f"<ocr_text>\n{ocr_text[:12_000]}\n</ocr_text>\n\nExtract the packing list JSON.",
            }
        ]
        if vision_page:
            b64 = self._to_b64(vision_page)
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

    @staticmethod
    def _to_b64(image: Image.Image, max_px: int = 1500) -> str:
        image.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _validate(raw: str) -> PackingList:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"AI returned invalid JSON: {e}") from e
        try:
            return PackingList(**data)
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}") from e


# ---------------------------------------------------------------------------
# PackingListAI — Main Entry Point
# ---------------------------------------------------------------------------


class PackingListAI:
    """
    Main extraction engine.

    Usage:
        extractor = PackingListAI()
        result    = extractor.process_file("sample.pdf")
        print(result)   # plain dict
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_vision: bool = True,
        ocr_lang: str = "eng",
    ) -> None:
        self.use_vision = use_vision
        self.loader     = DocumentLoader()
        self.ocr        = OCREngine(lang=ocr_lang)
        self.ai         = AIExtractionService(api_key=api_key)
        logger.info("PackingListAI ready")

    def process_file(self, file_path: Union[str, Path]) -> dict:
        """
        Load file → OCR → AI extraction → return clean dict.

        Accepts: PDF, JPG, PNG, BMP, TIFF, WebP
        """
        path  = Path(file_path)
        pages = self.loader.load(path)
        logger.info("Loaded %d page(s) from %s", len(pages), path.name)

        ocr_text    = self.ocr.extract_text(pages)
        vision_page = pages[0] if self.use_vision else None
        result      = self.ai.extract(ocr_text, vision_page=vision_page)

        logger.info(
            "Done — %d categories, %d items",
            len(result.categories),
            sum(len(c.items) for c in result.categories),
        )
        return result.model_dump(mode="json")