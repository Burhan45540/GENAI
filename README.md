GenAI Learning Repository

A collection of my hands-on practice, notes, and projects while learning
Generative AI, LangChain, RAG, chatbots, vector databases, and AI
agents.

This repository keeps my learning code organized by topic so I can
revise concepts, experiment with examples, and build projects step by
step.

Repository Structure

GENAI/
├── 1-DataIngestion/       # Loading data from different sources
├── 2-DataTransformation/  # Cleaning and transforming documents/data
├── 3-Embeddings/          # Text embeddings
├── 4-Vectorstore/         # Vector stores and similarity search
├── 5-Application/         # GenAI application practice
├── 6-Langserve/           # Serving LangChain chains through an API
├── 7-Chatbot/             # Chatbot and RAG chatbot notebooks
├── 8-Agents/              # AI agents and the AI Data Analyst Agent
├── .gitignore
├── README.md
└── requirements.txt

Projects and Topics

1. Data Ingestion

Practice loading data from different sources for use in GenAI
applications.

2. Data Transformation

Practice preparing and transforming loaded data before passing it to
later stages of a pipeline.

3. Embeddings

Explore how text can be represented as numerical vectors for semantic
search and retrieval.

4. Vector Stores

Practice storing embeddings and retrieving similar content through
similarity search.

5. GenAI Applications

Experiments with building applications powered by language models.

6. LangServe

Practice exposing a LangChain chain through an API using FastAPI and
LangServe.

7. Chatbots and RAG

Includes chatbot and retrieval-augmented generation (RAG) notebooks.

A typical RAG workflow is:

Load documents.

Split documents into chunks.

Create embeddings.

Store embeddings in a vector store.

Retrieve relevant chunks for a question.

Send the question and retrieved context to the language model.

Return an answer based on the retrieved context.

8. AI Agents --- AI Data Analyst Agent

The 8-Agents folder includes an AI Data Analyst Agent built with
Streamlit, Pandas, LangChain, and Groq.

Current features: - Upload a CSV dataset. - Preview the dataset. -
View the number of rows and columns. - Inspect column names and data
types. - View a statistical summary and missing-value count. - Ask
questions about the dataset in natural language. - Use tools for dataset
information, sums, averages, minimum/maximum values, and group-wise
aggregations.

Main application file: 8-Agents/app.py

Setup

1. Clone the repository

git clone https://github.com/Burhan455540/GENAI.git
cd GENAI

2. Create and activate a virtual environment

Using Python:

python -m venv .venv

On Windows Command Prompt:

.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Alternatively, use your existing Conda environment.

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Configure environment variables

Create a .env file in the repository root for local development:

GROQ_API_KEY=your_groq_api_key

Keep API keys private. Do not commit .env or any file containing a
real API key to GitHub.

The application uses a Groq API key to access its language model.

5. Run the AI Data Analyst Agent

From the repository root:

streamlit run 8-Agents/app.py

If your environment or dependency versions differ, install the packages
listed in requirements.txt and check the relevant project folder for
any project-specific setup.

Deploying the Streamlit App

The AI Data Analyst Agent can be deployed using Streamlit Community
Cloud.

Sign in to Streamlit Community Cloud with your GitHub account.

Select the Burhan455540/GENAI repository.

Select the main branch.

Set the main file path to 8-Agents/app.py.

Choose a Python version supported by Streamlit Community Cloud and
compatible with the project's dependencies.

Add GROQ_API_KEY under the app's Secrets settings.

Deploy the app.

Example secret entry:

GROQ_API_KEY = "your_actual_groq_api_key"

Do not put the real key in this README or commit it to the repository.

Technologies

Python

Pandas

Streamlit

LangChain

LangServe

FastAPI

Groq

Embeddings and vector stores

Retrieval-Augmented Generation (RAG)

The exact packages used by each project may vary; see requirements.txt
and the code in each folder.

Learning Goals

Understand the building blocks of GenAI applications.

Practice document loading, chunking, embeddings, and retrieval.

Build chatbots and RAG pipelines.

Learn how to serve chains through APIs.

Build tool-using AI agents.

Keep practice code and projects organized for revision and interview
preparation.

Security Notes

Never upload API keys, passwords, or other secrets.

Keep .env, virtual environments, cache folders, and local secret
files out of version control.

If an API key is accidentally published, revoke it and create a new
one.

Author

Burhan

This repository is a learning workspace and will continue to evolve as I
explore more GenAI concepts and projects.