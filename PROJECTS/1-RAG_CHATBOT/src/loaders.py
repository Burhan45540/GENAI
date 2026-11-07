from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader,WebBaseLoader
from langchain_core.documents import Document
import logging
from pypdf import apply_configuration
import easyocr
import pymupdf,io
import numpy as np
from PIL import Image
from pypdf.errors import LimitReachedError




def convert_pdf_to_images(file_path):

     pdf=pymupdf.open(file_path)

     images=[]

     for page in pdf:
          pix=page.get_pixmap(dpi=300)
          image=pix.tobytes("png")
          images.append(image)

     pdf.close()
     return images

reader=easyocr.Reader(['en'],gpu=False)
def extract_text_from_image(image_bytes):

     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

     image_array=np.array(image)

     results=reader.readtext(image_array)
     extracted_text=[]

     for result in results:
          text=result[1]
          extracted_text.append(text)

     return "\n".join(extracted_text)

def load_pdf(file_path:str)->list[Document]:
    """
    Load PDF and return its content as Document objects.
    """

    try:
        loader=PyPDFLoader(file_path=file_path)
        
        with apply_configuration(stream_parsing_work_maximum_count=10_000_000):

                documents=loader.load()


                total_text = sum(len(doc.page_content.strip())
                for doc in documents)

                if total_text >= 500:
                   return documents

                print("Very little text extracted. Switching to OCR...")

        logging.info(f"PDF Loaded Successfully : {file_path}")

        return documents

    except LimitReachedError:
         print("Normal PDF extraction failed. Trying OCR...")


    images=convert_pdf_to_images(file_path)

    for page_num , image in enumerate(images):

         text=extract_text_from_image(image)

         if text.strip():
              documents.append(
                   Document(page_content=text,metadeta={"source": file_path,
                        "page": page_num})
              )

    if not documents:
         raise ValueError("PDF contains no extractable text, even after OCR.")

    return documents

def load_docx(file_path:str)->list[Document]:

     """Load DOCX and return its content as Document objects."""  

     try:
          loader=Docx2txtLoader(file_path=file_path)

          documents=loader.load()

          if not any (docs.page_content.strip() for docs in documents):

               raise ValueError("DOCX contains no extractable text.")

          logging.info("DOCX Loaded Successfully : {file_path}")

          return documents


     except Exception as e:
          logging.error(f"Error in loading docx : {e}")
          raise



def load_txt(file_path: str) -> list[Document]:
    """
    Load TXT file and return its content as Document objects.
    """

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


def load_url(url: str) -> list[Document]:
    """
    Load webpage content and return it as Document objects.
    """

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