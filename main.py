"""LLM Book RAG - Command Line Interface & Runner."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings


def run_indexing(pdf_path: str):
    """Run extraction, semantic chunking, and FAISS vector indexing."""
    from src.document_loader.pdf_loader import load_pdf_document
    from src.chunking.semantic import split_documents_semantically
    from src.vectorstore.faiss_store import create_vector_store

    settings = get_settings()
    target_file = Path(pdf_path)

    print(f"[*] Step 1/3: Loading PDF via PDFPlumberLoader from: {target_file}")
    docs = load_pdf_document(target_file)
    print(f"[+] Loaded {len(docs)} pages.")

    print(f"[*] Step 2/3: Semantic chunking with HuggingFace embeddings ({settings.embedding_model_name})...")
    chunks = split_documents_semantically(docs)
    print(f"[+] Generated {len(chunks)} semantic chunks.")

    print(f"[*] Step 3/3: Embedding chunks & creating FAISS Vector Store at {settings.vector_store_dir}...")
    vector_store = create_vector_store(chunks)
    print("[✓] Ingestion & FAISS indexing completed successfully!")


def run_ask(question: str, provider: str = None):
    """Query the vector store (top 3 chunks) and generate answer via DeepSeek R1."""
    from src.vectorstore.faiss_store import load_vector_store
    from src.pipeline.rag_chain import create_rag_chain

    settings = get_settings()
    print(f"[*] Loading FAISS vector store from {settings.vector_store_dir}...")
    vector_store = load_vector_store()

    print(f"[*] Initializing RAG pipeline (fetching Top {settings.top_k_chunks} chunks & calling DeepSeek R1)...")
    pipeline = create_rag_chain(
        vector_store=vector_store,
        top_k=settings.top_k_chunks,
        llm_provider=provider,
    )

    print(f"\n[?] Question: {question}\n")
    response = pipeline.query(question)

    print("[Top 3 Retrieved Chunks]:")
    for i, doc in enumerate(response.source_documents, 1):
        page = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f"  ({i}) [Page {page}]: {preview}...")

    print(f"\n[DeepSeek R1 Answer]:\n{response.answer}\n")

    if response.images:
        print("[🖼️ Associated Diagram Images]:")
        for img_path in response.images:
            print(f"  - {img_path}")
        print()


def run_ui():
    """Launch Streamlit user interface."""
    import subprocess

    app_path = PROJECT_ROOT / "src" / "ui" / "app.py"
    print(f"[*] Launching Streamlit UI from {app_path}...")
    subprocess.run(["streamlit", "run", str(app_path)])


def main():
    parser = argparse.ArgumentParser(description="LLM Book RAG Application")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a PDF book into FAISS")
    index_parser.add_argument(
        "--pdf",
        type=str,
        default="data/raw/Hands-On_Large_Language_Models.pdf",
        help="Path to the PDF file",
    )

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question against the FAISS index")
    ask_parser.add_argument("question", type=str, help="Question to ask")
    ask_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["ollama", "deepseek_api", "groq", "openrouter"],
        help="LLM Provider override",
    )

    # UI command
    subparsers.add_parser("ui", help="Launch Streamlit web UI")

    args = parser.parse_args()

    if args.command == "index":
        run_indexing(args.pdf)
    elif args.command == "ask":
        run_ask(args.question, args.provider)
    elif args.command == "ui":
        run_ui()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

