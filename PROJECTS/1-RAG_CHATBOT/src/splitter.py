from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents:list[Document],
                    chunk_size:int=1000,
                    chunk_overlap:int=200)->list[Document]:

    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

    chunks=splitter.split_documents(documents=documents)

    return chunks