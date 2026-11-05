# GenAI Learning Repository

A collection of my hands-on practice, notes, and projects while learning Generative AI, LangChain, RAG, chatbots, vector databases, and AI agents.

This repository keeps my learning code organized by topic so I can revise concepts, experiment with examples, and build projects step by step.

## Repository Structure

```
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
```

## Projects and Topics

### 1. Data Ingestion
Practice loading data from different sources for use in GenAI applications.

### 2. Data Transformation
Practice preparing and transforming loaded data before passing it to later stages of a pipeline.

### 3. Embeddings
Explore how text can be represented as numerical vectors for semantic search and retrieval.

### 4. Vector Stores
Practice storing embeddings and retrieving similar content through similarity search.

### 5. GenAI Applications
Experiments with building applications powered by language models.

### 6. LangServe
Practice exposing a LangChain chain through an API using FastAPI and LangServe.

### 7. Chatbots and RAG
Includes chatbot and retrieval-augmented generation (RAG) notebooks.

A typical RAG workflow:
1. Load documents.
2. Split documents into chunks.
3. Create embeddings.
4. Store embeddings in a vector store.
5. Retrieve relevant chunks for a question.
6. Send the question and retrieved context to the language model.
7. Return an answer based on the retrieved context.

---

## ⭐ 8. AI Agents: AI Data Analyst Agent

Upload a CSV file and ask questions about your data in plain English. The agent picks the right tool, runs the calculation with Pandas, and explains the result in simple language.

🔗 **Live Demo:** _Add your Streamlit Cloud link here after deployment_

![Demo](assets/demo.png)

### Features

- Upload a CSV dataset with validation (empty file, no columns, read errors)
- Preview the dataset, with rows, columns, data types, statistical summary, and missing-value count
- Ask questions in natural language using **12 analysis tools**:
  - Dataset info, missing values, value counts
  - Sum, average, median, standard deviation, min/max
  - Correlation between two numerical columns
  - Group-wise aggregation (sum, mean, min, max, count)
  - Row filtering (`>`, `<`, `>=`, `<=`, `==`, `!=`)
  - Top N rows by a numerical column
- Interactive Plotly charts: bar chart, histogram, line chart
- Prompt-injection protection: CSV content is treated as data, never as instructions
- Chat resets automatically when a new file is uploaded (SHA-256 file hash)
- Every tool validates column names and data types and returns helpful errors

### Example Questions

- "What is the average of `Salary`?"
- "Show the top 5 rows by `Sales`"
- "Which `Department` has the highest total `Salary`?"
- "Filter rows where `Age` > 30"
- "What is the correlation between `Price` and `Quantity`?"
- "Are there any missing values?"

> Use the exact column names from your dataset for best results.

### How It Works

```
CSV Upload → Pandas DataFrame → Tools created for that DataFrame
                                        ↓
User Question → LangChain Agent (Groq LLM) → Picks a tool → Result → Simple explanation
```

### Limitations

- Supports only the fixed set of tools listed above (no free-form pandas queries yet)
- Works best with clean CSV files of small to medium size
- Column names must match exactly

### Future Improvements

- Safe pandas query tool for any custom question
- Excel and multi-file support
- Charts directly from chat

Main application file: `8-Agents/app.py`

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Burhan45540/GENAI.git
cd GENAI
```

### 2. Create and activate a virtual environment

Using Python:

```bash
python -m venv .venv
```

On Windows Command Prompt:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Alternatively, use your existing Conda environment.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the repository root for local development:

```env
GROQ_API_KEY=your_groq_api_key
```

Keep API keys private. Do not commit `.env` or any file containing a real API key to GitHub.

### 5. Run the AI Data Analyst Agent

From the repository root:

```bash
streamlit run 8-Agents/app.py
```

If your environment or dependency versions differ, install the packages listed in `requirements.txt` and check the relevant project folder for any project-specific setup.

## Deploying the Streamlit App

The AI Data Analyst Agent can be deployed using Streamlit Community Cloud.

1. Sign in to Streamlit Community Cloud with your GitHub account.
2. Select the `Burhan45540/GENAI` repository.
3. Select the `main` branch.
4. Set the main file path to `8-Agents/app.py`.
5. Choose a Python version supported by Streamlit Community Cloud and compatible with the project's dependencies.
6. Add `GROQ_API_KEY` under the app's **Secrets** settings.
7. Deploy the app.

Example secret entry:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

Do not put the real key in this README or commit it to the repository.

## Technologies

Python, Pandas, Plotly, Streamlit, LangChain, LangServe, FastAPI, Groq, Embeddings and vector stores, Retrieval-Augmented Generation (RAG)

The exact packages used by each project may vary; see `requirements.txt` and the code in each folder.

## Learning Goals

- Understand the building blocks of GenAI applications.
- Practice document loading, chunking, embeddings, and retrieval.
- Build chatbots and RAG pipelines.
- Learn how to serve chains through APIs.
- Build tool-using AI agents.
- Keep practice code and projects organized for revision and interview preparation.

## Security Notes

- Never upload API keys, passwords, or other secrets.
- Keep `.env`, virtual environments, cache folders, and local secret files out of version control.
- If an API key is accidentally published, revoke it and create a new one.

## Author

**Burhan**
GitHub: [@Burhan45540](https://github.com/Burhan45540)

This repository is a learning workspace and will continue to evolve as I explore more GenAI concepts and projects.