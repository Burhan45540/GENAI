# ask-my-pdf

Chat with your PDFs **100% locally**. A small RAG (Retrieval-Augmented Generation) pipeline built with **Ollama**, **ChromaDB** and Python. No API keys, no cloud, and your documents never leave your machine.

> Add a demo GIF here, e.g. `![demo](demo.gif)`

## How it works

```
PDF ──► extract text ──► split into chunks ──► embed (nomic-embed-text)
                                                      │
                                                      ▼
                                              ChromaDB (vectors)
                                                      │
Question ──► embed ──► top-k similar chunks ◄─────────┘
                              │
                              ▼
              chunks + question ──► Ollama LLM (llama3.2) ──► answer + page sources
```

1. **Extract** text from each page with `pypdf`
2. **Chunk** it into ~800-character pieces with 150 characters of overlap
3. **Embed** each chunk with an Ollama embedding model
4. **Store** the vectors in a local ChromaDB database (cached, so a PDF is only indexed once)
5. **Retrieve** the most relevant chunks for your question
6. **Generate** an answer with a local LLM, told to use only the retrieved context


Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--chat-model` | `llama3.2` | Any Ollama chat model (`mistral`, `qwen2.5`, ...) |
| `--embed-model` | `nomic-embed-text` | Any Ollama embedding model |
| `--top-k` | `4` | Number of chunks retrieved per question |
| `--db-dir` | `chroma_db` | Where the vector database is stored |

## Limitations

- **Scanned PDFs** (images only) have no extractable text. Run OCR first, e.g. `ocrmypdf in.pdf out.pdf`.
- One PDF per session.
- Answer quality depends on the local model you choose and on chunk size. Tweak `CHUNK_SIZE`, `CHUNK_OVERLAP` and `TOP_K` at the top of `ask_my_pdf.py`.

## Ideas for next steps

- [ ] Multiple PDFs in one knowledge base
- [ ] Streamlit web UI
- [ ] Conversation memory for follow-up questions



