from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import hashlib
import chromadb
import logging
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embeddings():

    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return embeddings

def get_vectorstore(kb_id:str):


    vector_store=Chroma(embedding_function=get_embeddings(),
                        persist_directory="./chroma_db",
                        collection_name=kb_id)

    return vector_store


def create_vectorstore(chunks:list[Document],kb_id:str):


    vector_store=get_vectorstore(kb_id=kb_id)
    vector_store.add_documents(chunks)

    return vector_store


def get_retreiver(vectorstore:Chroma,k:int=4):

    retreiver=vectorstore.as_retriever(
        search_kwargs={"k":k}
    )

    return retreiver


def delete(kb_id: str) -> None:
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(kb_id)
    except Exception:
        logging.warning("Collection %s not found", kb_id)




def generate_kb_id(
        documents:list[Document],
        chunk_size:int=1000,
        chunk_overlap:int=200):

    content=sorted([
        hashlib.sha256(doc.page_content.encode("utf-8") ).hexdigest()
        for doc in documents
        ])


    fingerprint=(
        "".join(content)
        +str(chunk_size)
        +str(chunk_overlap)
        +"sentence-transformers/all-MiniLM-L6-v2"
    )

    kb_hash=hashlib.sha256(
        fingerprint.encode("utf-8")
    ).hexdigest()

    return f"kb_{kb_hash[:16]}"



def kb_id_exists(kb_id: str) -> bool:
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection(kb_id)
        return collection.count() > 0
    except Exception:
        return False



def get_create_vectorstore(chunks:list[Document],kb_id:str):

    if kb_id_exists(kb_id=kb_id):

        print("Existing Knowledge Base Found!")

        return get_vectorstore(kb_id=kb_id)

    else:
         print("Creating New Knowledge Base!")

         return create_vectorstore(chunks=chunks,kb_id=kb_id)

    


