
import streamlit as st
import tempfile
import os
import logging

from src.loaders import (
    load_url,
    load_pdf,
    load_txt,
    load_docx
)

from src.splitter import split_documents

from src.vectorstore import (
    generate_kb_id,
    get_create_vectorstore,
    get_retreiver,
    delete
)

from src.chain import (
    get_rag_chain,
    get_vision_llm
)

from langchain_core.messages import HumanMessage, AIMessage


st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind")
st.caption("Chat with your documents using AI.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "kb_id" not in st.session_state:
    st.session_state.kb_id = None


with st.sidebar:

    st.header("Document Settings")

    source_types = st.selectbox(
        "Select Source",
        ["PDF", "DOCX", "TXT", "Web URL"]
    )

    chunk_size = st.number_input("Chunk Size", value=1000)
    chunk_overlap = st.number_input("Chunk Overlap", value=200)

    if source_types == "Web URL":
        url = st.text_input("Enter Website URL : ")

    else:
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

    process_button = st.button("Process Documents")

    if st.button("Clear Chat"):

        st.session_state.chat_history = []
        st.rerun()

    if st.button("Reset Knowledge Base"):

        if st.session_state.kb_id:

            delete(st.session_state.kb_id)

            st.session_state.kb_id = None
            st.session_state.rag_chain = None
            st.session_state.chat_history = []

            st.success("Knowledge Base reset successfully!")

            st.rerun()

        else:
            st.warning("No Knowledge Base found.")


if process_button:

    documents = []

    # Validate chunk settings
    if chunk_overlap >= chunk_size:

        st.error("Chunk overlap must be smaller than chunk size.")

    else:

        try:

            with st.spinner("Loading Documents..."):

                # Web URL loading
                if source_types == "Web URL":

                    if not url.strip():
                        raise ValueError("Please enter a website URL.")

                    documents = load_url(url=url)

                # Multiple file loading
                else:

                    if not uploaded_files:
                        raise ValueError("Please upload at least one document.")

                    for file in uploaded_files:

                        # Temporary file creation
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=os.path.splitext(file.name)[1]
                        ) as temp:

                            temp.write(file.getvalue())

                            file_path = temp.name

                        try:

                            # Select loader automatically
                            extension = os.path.splitext(file.name)[1].lower()

                            if extension == ".pdf":

                                docs = load_pdf(file_path)

                            elif extension == ".docx":

                                docs = load_docx(file_path)

                            elif extension == ".txt":

                                docs = load_txt(file_path)

                            else:

                                raise ValueError(
                                    f"Unsupported file format: {extension}"
                                )

                            # Preserve original filename
                            for doc in docs:
                                doc.metadata["source"] = file.name

                            documents.extend(docs)

                        finally:

                            if os.path.exists(file_path):
                                os.remove(file_path)

            # Check extracted documents
            if not documents:

                raise ValueError("No readable content found in documents.")

            st.success(
                f"Successfully loaded {len(documents)} document pages."
            )
            
            with st.spinner("Creating Knowledge Base..."):

                chunks = split_documents(
                    documents,
                    chunk_size,
                    chunk_overlap
                )

                kb_id = generate_kb_id(
                    documents,
                    chunk_size,
                    chunk_overlap
                )

                vectorstore = get_create_vectorstore(
                    chunks,
                    kb_id
                )

                retreiver = get_retreiver(
                    vectorstore=vectorstore
                )

                st.session_state.rag_chain = get_rag_chain(
                    retreiver=retreiver
                )

                if st.session_state.kb_id != kb_id:
                    st.session_state.chat_history = []

                st.session_state.kb_id = kb_id

            st.success(
                f"Knowledge Base created successfully! Total chunks: {len(chunks)}"
            )

        except Exception as e:

            logging.exception("Document loading failed")

            st.error(f"Document loading failed: {e}")


for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    else:
        with st.chat_message("assistant"):
            st.write(message.content)


question = st.chat_input("Ask something about your documents...")

if question:

    if st.session_state.rag_chain is None:
        st.warning("Please process your documents first.")

    else:
        response = st.session_state.rag_chain.invoke(
            {
                "input": question,
                "chat_history": st.session_state.chat_history
            }
        )

        answer = response["answer"]

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(answer)

            with st.expander("view sources:"):

                for doc in response["context"]:
                    st.write(doc.metadata)
                    st.write(doc.page_content[:300])
                    st.divider()

        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
