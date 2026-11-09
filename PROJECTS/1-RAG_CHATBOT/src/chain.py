from langchain_groq import ChatGroq
from config import groq_api_key
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    llm=ChatGroq(model="openai/gpt-oss-20b",api_key=groq_api_key)
    return llm


system_prompt = (
    "You are a helpful document assistant. "
    "Answer the user's question using only the provided context. "
    "Do not use outside knowledge. "
    "If the answer is not available in the context, "
    "respond exactly with: I couldn't find this in your documents.\n\n"
    "Context:\n{context}"
)

qa_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)



contextualize_prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "Given the conversation history and the latest user question, "
        "rewrite the question as a standalone question. "
        "Do not answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)
def get_rag_chain(retreiver):

    llm=get_llm()

    history_aware_retreiver=create_history_aware_retriever(llm,retreiver,contextualize_prompt)

    qa_chain=create_stuff_documents_chain(llm,qa_prompt)

    rag_chain=create_retrieval_chain(
        history_aware_retreiver,qa_chain
    )

    return rag_chain


def get_vision_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=groq_api_key
    )

    return llm