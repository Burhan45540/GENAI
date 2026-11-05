from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import hashlib
import chromadb

def get_embeddings():

    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return embeddings

def create_vectorstore(chunks:list[Document],kb_id:str):

    embeddings=get_embeddings()

    vector_store=Chroma(embedding_function=embeddings,
                        persist_directory="./chroma_db",
                        collection_name=kb_id)

    vector_store.add_documents(chunks)

    return vector_store

def get_vectorstore(kb_id:str):

    embeddings=get_embeddings()

    vector_store=Chroma(embedding_function=embeddings,
                        persist_directory="./chroma_db",
                        collection_name=kb_id)

    return vector_store

def get_retreiver(vectorstore:Chroma):

    retreiver=vectorstore.as_retriever(
        search_kwargs={"k":4}
    )

    return retreiver


def delete(kb_id:str):

    embeddings=get_embeddings()

    vectorstore=Chroma(collection_name=kb_id,
        embedding_function=embeddings,
        persist_directory="./chroma_db")

    vectorstore.delete_collection()




def generate_kb_id(
        documents:list[Document],
        chunk_size:int=1000,
        chunk_overlap:int=200):

    content=[
        hashlib.sha256(doc.page_content.encode("utf-8") for doc in documents).hexdigest()
        ]

    content.sort()

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



def kb_id_exists(kb_id:str):

    client=chromadb.PersistentClient(path="./chromadb")

    collections=client.list_collections()

    for collection in collections:

        if collection.name==kb_id:
            return True

    return False


def get_create_vectorstore(chunks:list[Document],kb_id:str):

    if kb_id_exists(kb_id=kb_id):

        print("Existing Knowledge Base Found!")

        vectorstore=get_vectorstore(kb_id=kb_id)

    else:
         print("Creating New Knowledge Base!")

         vectorstore=create_vectorstore(chunks=chunks,kb_id=kb_id)

    return vectorstore


