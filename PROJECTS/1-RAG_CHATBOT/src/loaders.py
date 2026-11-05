from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader,WebBaseLoader
from langchain_core.documents import Document
import logging



def load_pdf(file_path:str)->list[Document]:
    """
    Load PDF and return its content as Document objects.
    """

    try:
        loader=PyPDFLoader(file_path=file_path)

        documents=loader.load()


        if not any(doc.page_content.strip() for doc in documents):
                raise ValueError("PDF contains no extractable text.")

        logging.info(f"PDF Loaded Successfully : {file_path}")

        return documents

    except Exception as e:
         logging.error(f"Error in loading pdf : {e}")
         raise


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