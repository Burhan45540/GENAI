
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
)

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from pypdf import apply_configuration
from pypdf.errors import LimitReachedError

from functools import lru_cache

import logging
import pymupdf
import easyocr
import io
import base64

import numpy as np
from PIL import Image


# -----------------------------------
# 1. Lazy Loading for EasyOCR
# -----------------------------------


@lru_cache(maxsize=1)
def get_reader():

    logging.info("Initializing EasyOCR model...")

    return easyocr.Reader(["en"], gpu=False)


# -----------------------------------
# 2. Convert PDF Pages into Images
# -----------------------------------


def convert_pdf_to_images(file_path):

    pdf = pymupdf.open(file_path)

    images = []

    for page in pdf:

        pix = page.get_pixmap(dpi=200)

        image = pix.tobytes("png")

        images.append(image)

    pdf.close()

    return images


# -----------------------------------
# 3. Extract Text using EasyOCR
# -----------------------------------


def extract_text_from_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_array = np.array(image)

    reader = get_reader()

    results = reader.readtext(image_array)

    extracted_text = []

    for result in results:

        text = result[1]

        extracted_text.append(text)

    return "\n".join(extracted_text)


# -----------------------------------
# 4. Extract Handwriting using Vision LLM
# -----------------------------------


def ocr_vision_llm(image_bytes, vision_llm):

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_url = f"data:image/png;base64,{image_base64}"

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Extract all readable text from this document image. "
                    "Preserve the original meaning, headings, and structure. "
                    "Do not summarize or invent missing text."
                ),
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )

    response = vision_llm.invoke([message])

    return response.content


# -----------------------------------
# 5. Load PDF
# -----------------------------------


def load_pdf(
    file_path: str, handwritten: bool = False, vision_llm=None
) -> list[Document]:

    documents = []

    try:

        loader = PyPDFLoader(file_path=file_path)

        with apply_configuration(stream_parsing_work_maximum_count=10_000_000):

            documents = loader.load()

        logging.info(f"Normal PDF extraction completed: {file_path}")

    except LimitReachedError:

        logging.warning("PDF text extraction limit reached. Switching to OCR.")

    # Normal PDF: Return extracted text directly
    if not handwritten and documents and all(
        len(doc.page_content.strip()) >= 80
        for doc in documents
    ):

        return documents

    # Convert PDF pages into images only when OCR is required
    images = convert_pdf_to_images(file_path)

    extracted_documents = []

    for page_num, image in enumerate(images):

        page_text = ""

        # Handwritten document: use Vision LLM
        if handwritten:

            if vision_llm is None:

                raise ValueError("Vision LLM is required for handwritten documents.")

            page_text = ocr_vision_llm(image, vision_llm)

        else:

            # Get normal PDF extracted text for this page
            if page_num < len(documents):

                page_text = documents[page_num].page_content.strip()

            # If page has insufficient text, use EasyOCR
            if len(page_text) < 80:

                logging.info(f"Using EasyOCR for page {page_num + 1}")

                page_text = extract_text_from_image(image)

        if page_text.strip():

            extracted_documents.append(
                Document(
                    page_content=page_text,
                    metadata={"source": file_path, "page": page_num},
                )
            )

    if not extracted_documents:

        raise ValueError("PDF contains no extractable text, even after OCR.")

    logging.info(f"PDF Loaded Successfully: {file_path}")

    return extracted_documents


# -----------------------------------
# 6. Load DOCX
# -----------------------------------


def load_docx(file_path: str) -> list[Document]:

    try:

        loader = Docx2txtLoader(file_path=file_path)

        documents = loader.load()

        if not any(doc.page_content.strip() for doc in documents):

            raise ValueError("DOCX contains no extractable text.")

        logging.info(f"DOCX Loaded Successfully: {file_path}")

        return documents

    except Exception as e:

        logging.error(f"Error loading DOCX: {e}")

        raise


# -----------------------------------
# 7. Load TXT
# -----------------------------------


def load_txt(file_path: str) -> list[Document]:

    try:

        loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()

        if not any(doc.page_content.strip() for doc in documents):

            raise ValueError("TXT file is empty.")

        logging.info(f"Successfully loaded TXT: {file_path}")

        return documents

    except Exception as e:

        logging.error(f"Error loading TXT: {e}")

        raise


# -----------------------------------
# 8. Load URL
# -----------------------------------


def load_url(url: str) -> list[Document]:

    try:

        loader = WebBaseLoader(url)

        documents = loader.load()

        if not any(doc.page_content.strip() for doc in documents):

            raise ValueError("No extractable content found on webpage.")

        logging.info(f"Successfully loaded URL: {url}")

        return documents

    except Exception as e:

        logging.error(f"Error loading URL: {e}")

        raise
