from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from fastapi import FastAPI
from langserve import add_routes


groq_api_key=os.getenv("GROQ_API_KEY")


#LLM MODEL
model=ChatGroq(model="openai/gpt-oss-20b",api_key=groq_api_key)

#prompt

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert language translator. Translate the input into {language}."),
    ("human", "Input: {input}")
])

#parser

parser=StrOutputParser()

#LangChain Expression Language

chain=prompt|model|parser

#app definition

app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A Simple API server using Runnable interface")

add_routes(app,chain,path="/chain")

if __name__=="__main__":
    import uvicorn

    uvicorn.run(app=app,host="localhost",port=8000)
