# 📚 LLM Book RAG Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-orange.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.38+-red.svg)](https://streamlit.io/)
[![DeepSeek R1](https://img.shields.io/badge/LLM-DeepSeek%20R1-blueviolet.svg)](https://github.com/deepseek-ai/DeepSeek-R1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, production-ready Retrieval-Augmented Generation (RAG) system tailored for technical books and documents. Featuring **semantic chunking**, **FAISS vector search**, **high-resolution diagram extraction**, and reasoning powered by **DeepSeek R1** via local (Ollama) or cloud (Groq, OpenRouter, DeepSeek API) providers.

---

## 🏗️ System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │                       User                             │
               └──────────────────┬─────────────────▲───────────────────┘
                                  │                 │
                           Asks Question     Displays Answer & Diagrams
                                  │                 │
                                  ▼                 │
               ┌────────────────────────────────────┴───────────────────┐
               │              Streamlit Web App / CLI                   │
               └──────────┬─────────────────────────▲───────────────────┘
                          │                         │
                     Uploads PDF             Synthesizes Answer
                          │                         │
                          ▼                         │
               ┌──────────────────────┐   ┌─────────────────────────────┐
               │   PDFPlumberLoader   │   │       DeepSeek R1 LLM       │
               │  & Image Extractor   │   │ (Ollama / Groq / OpenRouter)│
               └──────────┬───────────┘   └─────────▲───────────────────┘
                          │                         │
                      Full Text               Top-k Chunks
                          │                   & Page Figures
                          ▼                         │
               ┌──────────────────────┐   ┌─────────┴───────────────────┐
               │   Semantic Chunker   │   │       FAISS Retriever       │
               │  (LangChain Exp.)    │   │      (Similarity Top-K)     │
               └──────────┬───────────┘   └─────────▲───────────────────┘
                          │                         │
                    Semantic Chunks            Dense Search
                          │                         │
                          ▼                         │
               ┌────────────────────────────────────┴───────────────────┐
               │                   FAISS Vector Store                   │
               │           (Dense HuggingFace Embeddings)               │
               └────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Semantic Chunking**: Splits text based on embedding distance shifts rather than fixed character counts, keeping context and paragraphs conceptually coherent.
- **Automated Figure & Diagram Extraction**: Automatically identifies and renders high-resolution diagrams/figures referenced on retrieved book pages.
- **DeepSeek R1 Reasoning**: Seamlessly streams or synthesizes chain-of-thought reasoning responses grounded in the book content.
- **Multiple LLM Providers**:
  - 🏠 **Ollama** (Local, offline, privacy-first with `deepseek-r1:8b`)
  - ⚡ **Groq** (Ultra-fast cloud inference with `deepseek-r1-distill-llama-70b`)
  - 🌐 **OpenRouter** (Unified cloud access to full `deepseek/deepseek-r1`)
  - 🏢 **DeepSeek API** (Official `deepseek-reasoner` endpoint)
- **Multi-Document & Multi-Chat Support**: Manage multiple ingested books and maintain dedicated, persistent conversation threads for each book.
- **Persistent Local Storage**: FAISS vector indices and chat histories are stored locally on disk for zero-latency restarts without re-indexing.
- **Dual Interface**: Full-featured interactive **Streamlit Web UI** and a clean **Command-Line Interface (CLI)**.

---

## 📁 Repository Structure

```text
lllm_book_rag/
├── data/
│   ├── chats/                      # Persistent chat history JSON files
│   │   └── .gitkeep
│   ├── images/                     # Extracted diagram figures from book pages
│   │   └── .gitkeep
│   ├── raw/                        # Uploaded / source PDF documents
│   │   └── .gitkeep
│   └── vectorstore/                # FAISS vector indices (per document)
│       └── .gitkeep
├── src/
│   ├── chunking/                   # Semantic chunking logic
│   │   ├── __init__.py
│   │   └── semantic.py
│   ├── config/                     # Settings and session management
│   │   ├── __init__.py
│   │   ├── chat_store.py           # Persistent multi-chat store
│   │   └── settings.py             # Environment variables configuration
│   ├── document_loader/            # PDF ingestion & diagram extraction
│   │   ├── __init__.py
│   │   ├── image_extractor.py      # High-res figure cropper & renderer
│   │   └── pdf_loader.py           # PDFPlumber text extractor
│   ├── embeddings/                 # HuggingFace dense embedding loader
│   │   ├── __init__.py
│   │   └── hf_embeddings.py
│   ├── llm/                        # DeepSeek R1 provider integrations
│   │   ├── __init__.py
│   │   └── deepseek.py
│   ├── pipeline/                   # End-to-end RAG retrieval & prompt chain
│   │   ├── __init__.py
│   │   └── rag_chain.py
│   ├── retriever/                  # FAISS top-k similarity retriever
│   │   ├── __init__.py
│   │   └── faiss_retriever.py
│   ├── ui/                         # Streamlit multi-chat interface
│   │   ├── __init__.py
│   │   └── app.py
│   └── vectorstore/                # FAISS indexing & disk serialization
│       ├── __init__.py
│       └── faiss_store.py
├── .env.example                    # Template for environment configuration
├── .gitignore                      # Git exclusion rules (protects data & secrets)
├── .streamlit/
│   └── config.toml                 # Streamlit server & UI configuration
├── main.py                         # CLI entrypoint (index, ask, ui)
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation & setup guide
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites

- **Python**: Version `3.10`, `3.11`, or `3.12` (recommended for PyTorch and FAISS binary wheel compatibility).
- **Git**: Installed and configured on your machine.
- **Hardware**: Any modern CPU (M-series Apple Silicon or Intel/AMD).

---

### 2. Clone Repository & Setup Virtual Environment

```bash
# Clone your repository (replace with your repo URL)
git clone https://github.com/<your-username>/lllm_book_rag.git
cd lllm_book_rag

# Create a virtual environment
# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (Command Prompt):
python -m venv .venv
.venv\Scripts\activate.bat

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

The project uses `.env` for secrets and configuration. A documented template is provided in `.env.example`.

Create your local `.env` file:
```bash
cp .env.example .env
```

Open `.env` in your editor and choose your LLM provider:

#### Option A: Local Ollama (Completely Free & Private)
1. Install [Ollama](https://ollama.com/) on your system.
2. Pull the DeepSeek R1 distilled model:
   ```bash
   ollama pull deepseek-r1:8b
   ```
3. Configure `.env`:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=deepseek-r1:8b
   ```

#### Option B: Groq (Ultra-Fast Cloud Inference)
1. Get a free API key from [Groq Console](https://console.groq.com/).
2. Configure `.env`:
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=deepseek-r1-distill-llama-70b
   ```

#### Option C: OpenRouter
1. Get an API key from [OpenRouter](https://openrouter.ai/).
2. Configure `.env`:
   ```env
   LLM_PROVIDER=openrouter
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OPENROUTER_MODEL=deepseek/deepseek-r1
   ```

#### Option D: Official DeepSeek API
1. Get an API key from [DeepSeek Platform](https://platform.deepseek.com/).
2. Configure `.env`:
   ```env
   LLM_PROVIDER=deepseek_api
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
   DEEPSEEK_MODEL=deepseek-reasoner
   ```

> [!IMPORTANT]
> Never commit your `.env` file to GitHub. The `.gitignore` file is pre-configured to ensure that `.env` and any secret files are ignored.

---

## 💻 Usage

### 1. Launch the Streamlit Web Application

Launch the interactive workspace with document ingestion, chat tabs, and diagram inspectors:

```bash
python main.py ui
```
*Alternatively:*
```bash
streamlit run src/ui/app.py
```

Once running, navigate to `http://localhost:8501`:
1. Use the sidebar to **upload a PDF** or choose an existing indexed book.
2. Click **Process & Index Book** to run semantic chunking and FAISS indexing.
3. Start chatting! You can inspect retrieved source passages and referenced diagram figures inline.

---

### 2. Command Line Interface (CLI)

You can also index books and query them directly from the terminal.

#### Step 1: Index a PDF Book
```bash
python main.py index --pdf data/raw/your_book.pdf
```

#### Step 2: Query the Indexed Book
```bash
python main.py ask "What are the core components of a Transformer encoder?"
```

#### Step 3: Override the Provider on the Fly
```bash
python main.py ask "Explain attention mechanisms in detail" --provider groq
```

---

## ⚙️ Advanced Configuration

All default parameters can be tuned in `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `ollama` | Active provider (`ollama`, `groq`, `openrouter`, `deepseek_api`) |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face model for vector embeddings |
| `EMBEDDING_DEVICE` | `cpu` | Device for embeddings (`cpu`, `cuda`, `mps`) |
| `BREAKPOINT_THRESHOLD_TYPE` | `percentile` | Threshold method for `SemanticChunker` |
| `BREAKPOINT_THRESHOLD_AMOUNT` | `95.0` | Semantic distance threshold percentile |
| `TOP_K_CHUNKS` | `3` | Number of context chunks retrieved per query |
| `DATA_RAW_DIR` | `data/raw` | Destination directory for uploaded PDFs |
| `VECTOR_STORE_DIR` | `data/vectorstore` | Destination directory for FAISS indexes |

---

## 🔒 Security & Git Hygiene

- **API Keys & Secrets**: `.env` and any `.env.*` files are excluded by `.gitignore`. Only `.env.example` is tracked.
- **Large Binaries & PDFs**: Local PDFs (`data/raw/*`), FAISS indexes (`data/vectorstore/*`), diagram renders (`data/images/*`), and chat logs (`data/chats/*`) are excluded from Git to prevent repository bloat.
- **Folder Preservation**: `.gitkeep` files ensure the necessary empty directories exist on fresh clones.

---

## 📤 Pushing to GitHub (Quick Reference)

When you are ready to push to your GitHub repository:

```bash
# 1. Check git status to ensure only tracked files are included
git status

# 2. Stage all project files
git add .

# 3. Create initial commit
git commit -m "feat: initial commit of LLM Book RAG Assistant"

# 4. Set main branch
git branch -M main

# 5. Link to your remote GitHub repository
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 6. Push code to GitHub
git push -u origin main
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
