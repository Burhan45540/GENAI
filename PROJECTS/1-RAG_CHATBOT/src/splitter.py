from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents:list[Document])->list[Document]:

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    chunks=splitter.split_documents(documents=documents)

    return chunks