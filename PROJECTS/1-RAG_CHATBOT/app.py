import streamlit as st
from src.loaders import load_url,load_pdf,load_txt,load_docx
import tempfile,os
from src.splitter import split_documents
from src.vectorstore import generate_kb_id,get_create_vectorstore,get_retreiver
from src.chain import get_rag_chain
from langchain_core.messages import HumanMessage,AIMessage

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"

)

st.title("📚 DocuMind")
st.caption("Chat with your documents using AI.")



if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain=None

if "kb_id" not in st.session_state:
    st.session_state.kb_id=None



with st.sidebar:

    st.header("Document Settings")

    source_types=st.selectbox(
        "Select Source",
        ["PDF", "DOCX", "TXT", "Web URL"]
    )

    chunk_size=st.number_input("Chunk Size",value=1000)
    chunk_overlap = st.number_input("Chunk Overlap", value=200)

    if source_types=="Web URL":
        url=st.text_input("Enter Website URL : ")

    else:
        uploaded_files=st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

    process_button=st.button("Process Documents")

    if st.button("Clear Chat"):

        st.session_state.chat_history=[]
        st.rerun()


    if process_button:

        documents=[]

        with st.spinner("Loading Documents..."):

            if source_types=="Web URL":
                if url:
                    documents=load_url(url=url)

            elif uploaded_files:

                for file in uploaded_files:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(file.name)[1]
                    ) as temp:

                        temp.write(file.getvalue())

                        file_path=temp.name

                    try:
                        if source_types == "PDF":
                           docs = load_pdf(file_path)

                        elif source_types == "DOCX":
                            docs = load_docx(file_path)

                        else:
                            docs = load_txt(file_path)

                        for doc in docs:
                            doc.metadata["source"]=file.name

                        documents.extend(docs)

                    finally:
                        os.remove(file_path)

        print("Total Documents:", len(documents))



        if documents:

            with st.spinner("Creating Knowledge Base..."):

                chunks=split_documents(documents,chunk_size,chunk_overlap)

                kb_id=generate_kb_id(documents,chunk_size,chunk_overlap)

                vectorstore=get_create_vectorstore(chunks,kb_id)

                retreiver=get_retreiver(vectorstore=vectorstore)

                st.session_state.rag_chain=get_rag_chain(retreiver=retreiver)

                if st.session_state.kb_id != kb_id:
                    st.session_state.chat_history=[]

                st.session_state.kb_id=kb_id

            st.success(f"Documents processed successfully! Total chunks: {len(chunks)}")

        else:
            st.warning("Please upload documents or enter a URL.")





for message in st.session_state.chat_history:

    if isinstance(message,HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    else:
        with st.chat_message("assistant"):
            st.write(message.content)


question=st.chat_input("Ask something about your documents...")

if question:

    if st.session_state.rag_chain is None:
        st.warning("Please process your documents first.")

    else:
        response=st.session_state.rag_chain.invoke(
            {
            "input": question,
            "chat_history": st.session_state.chat_history
            }
        )

        answer=response["answer"]

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