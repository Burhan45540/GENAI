from dotenv import load_dotenv
import os

load_dotenv() 

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "GENAIAPP")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions asked."),
        ("user", "Question: {question}"),
    ]
)

st.title("Simple GENAI APPLICATION With OLLAMA")
input_text = st.text_input("What question you have in your mind?")

llm = OllamaLLM(model="llama3.2:latest")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))