import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
hugging_face_key=os.getenv("HUGGINGFACE_API_KEY")

langsmith_api_key=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
langchain_project=os.getenv("LANGCHAIN_PROJECT")