"""Streamlit User Interface for LLM Book RAG with Multi-Document & Multi-Chat support."""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.config.settings import get_settings
from src.config.chat_store import (
    ChatSession,
    load_chats_for_doc,
    save_or_update_chat,
    create_new_chat,
    delete_chat,
)
from src.document_loader.pdf_loader import load_pdf_document
from src.chunking.semantic import split_documents_semantically
from src.vectorstore.faiss_store import (
    create_vector_store,
    load_vector_store,
    get_document_vector_dir,
    list_available_vector_stores,
)
from src.pipeline.rag_chain import create_rag_chain


def init_page():
    """Configure Streamlit page layout and title."""
    st.set_page_config(
        page_title="Book RAG Workspace",
        page_icon="📚",
        layout="wide",
    )


def get_clean_doc_id(file_name: str) -> str:
    """Derive clean document ID from file name."""
    base = Path(file_name).stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)


def render_sidebar(settings):
    """Render sidebar with document switcher, chat sessions, and ingestion."""
    with st.sidebar:
        st.title("📚 Book RAG Hub")

        # -------------------------------------------------------------
        # 1. Document / Book Selection & Ingestion
        # -------------------------------------------------------------
        st.subheader("📖 Books & Documents")

        available_books = list_available_vector_stores()
        # Ensure default book is present if files exist
        if not available_books and (settings.data_raw_dir / "Hands-On_Large_Language_Models.pdf").exists():
            available_books = ["Hands-On_Large_Language_Models"]

        # Track active document in session_state
        if "active_doc_id" not in st.session_state:
            st.session_state["active_doc_id"] = (
                available_books[0] if available_books else "Hands-On_Large_Language_Models"
            )

        # Build display labels with chat counts
        doc_options = {}
        for b in available_books:
            chats = load_chats_for_doc(b)
            chat_count = len(chats)
            doc_options[b] = f"📖 {b} ({chat_count} {'chat' if chat_count == 1 else 'chats'})"

        selected_doc = None
        if available_books:
            current_index = 0
            if st.session_state["active_doc_id"] in available_books:
                current_index = available_books.index(st.session_state["active_doc_id"])

            selected_doc = st.selectbox(
                "Select Active Book:",
                options=available_books,
                index=current_index,
                format_func=lambda x: doc_options.get(x, x),
                label_visibility="collapsed",
            )
            if selected_doc != st.session_state["active_doc_id"]:
                st.session_state["active_doc_id"] = selected_doc
                st.session_state["active_chat_id"] = None
                st.session_state["vector_store"] = None
                st.rerun()

        # Upload New Book Accordion
        with st.expander("📥 Ingest New PDF Book", expanded=not bool(available_books)):
            uploaded_file = st.file_uploader("Drop a new PDF book here", type=["pdf"])
            if uploaded_file is not None:
                new_doc_id = get_clean_doc_id(uploaded_file.name)
                save_path = settings.data_raw_dir / uploaded_file.name
                settings.data_raw_dir.mkdir(parents=True, exist_ok=True)

                if st.button("🚀 Process & Index Book", use_container_width=True):
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner("1/3 Extracting text with PDFPlumberLoader..."):
                        docs = load_pdf_document(save_path)

                    with st.spinner("2/3 Splitting with SemanticChunker & HuggingFace..."):
                        chunks = split_documents_semantically(docs)

                    with st.spinner("3/3 Indexing into FAISS vector store..."):
                        doc_vector_dir = settings.vector_store_dir / new_doc_id
                        create_vector_store(documents=chunks, save_directory=doc_vector_dir)

                    st.session_state["active_doc_id"] = new_doc_id
                    st.session_state["active_chat_id"] = None
                    st.session_state["vector_store"] = None
                    st.success(f"Indexed '{uploaded_file.name}'!")
                    st.rerun()

        st.divider()

        # -------------------------------------------------------------
        # 2. Chat Sessions for Active Book
        # -------------------------------------------------------------
        active_doc = st.session_state.get("active_doc_id", "Hands-On_Large_Language_Models")
        chats_for_doc = load_chats_for_doc(active_doc)

        st.subheader("💬 Chats for this Book")

        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_session = create_new_chat(active_doc, title=f"Chat {len(chats_for_doc) + 1}")
            st.session_state["active_chat_id"] = new_session.id
            st.rerun()

        # Ensure active_chat_id is valid
        if "active_chat_id" not in st.session_state or not st.session_state["active_chat_id"]:
            st.session_state["active_chat_id"] = chats_for_doc[0].id if chats_for_doc else None

        # Render list of chats
        for c in chats_for_doc:
            is_active = (c.id == st.session_state.get("active_chat_id"))
            msg_count = len(c.messages)
            label = f"{'🟢' if is_active else '💬'} {c.title} ({msg_count})"

            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if st.button(label, key=f"select_{c.id}", use_container_width=True):
                    st.session_state["active_chat_id"] = c.id
                    st.rerun()
            with col2:
                if len(chats_for_doc) > 1 and st.button("🗑️", key=f"del_{c.id}", help="Delete chat"):
                    delete_chat(active_doc, c.id)
                    st.session_state["active_chat_id"] = None
                    st.rerun()

        st.divider()

        # -------------------------------------------------------------
        # 3. Model Configuration
        # -------------------------------------------------------------
        provider = st.selectbox(
            "LLM Provider",
            options=["openrouter", "groq", "deepseek_api", "ollama"],
            index=["openrouter", "groq", "deepseek_api", "ollama"].index(
                settings.llm_provider
            )
            if settings.llm_provider in ["openrouter", "groq", "deepseek_api", "ollama"]
            else 0,
        )

        return active_doc, provider


def render_diagram_gallery(image_paths: list, key_suffix: str = ""):
    """Render extracted diagrams in a single horizontal row with an interactive inspector."""
    valid_images = [p for p in image_paths if Path(p).exists()]
    if not valid_images:
        return

    st.markdown("##### 🖼️ Referenced Book Diagrams:")

    # 1. Render all images in ONE single horizontal row
    num_cols = min(len(valid_images), 3)
    cols = st.columns(num_cols)
    for idx, img_path in enumerate(valid_images):
        col = cols[idx % num_cols]
        with col:
            stem = Path(img_path).stem
            parts = stem.split("_")
            page_label = f"Page {parts[1]}" if len(parts) >= 2 else stem
            st.image(
                img_path,
                caption=f"Diagram #{idx+1} ({page_label})",
                use_container_width=True,
            )

    # 2. Interactive full-resolution click-to-open viewer
    with st.expander("🔎 Click to Open & Inspect Diagrams in High Resolution", expanded=False):
        tab_labels = []
        for i, img_path in enumerate(valid_images, start=1):
            stem = Path(img_path).stem
            parts = stem.split("_")
            page_str = f"Page {parts[1]}" if len(parts) >= 2 else f"Fig {i}"
            tab_labels.append(f"🖼️ Diagram {i} ({page_str})")

        tabs = st.tabs(tab_labels)
        for tab, img_path, label in zip(tabs, valid_images, tab_labels):
            with tab:
                st.image(img_path, caption=f"{label} - Full High Resolution", use_container_width=True)


def main():
    settings = get_settings()
    init_page()
    active_doc, provider = render_sidebar(settings)

    # -------------------------------------------------------------
    # 4. Load Active Chat Session
    # -------------------------------------------------------------
    chats = load_chats_for_doc(active_doc)
    active_chat_id = st.session_state.get("active_chat_id") or (chats[0].id if chats else None)
    active_chat: Optional[ChatSession] = next((c for c in chats if c.id == active_chat_id), None)

    if not active_chat:
        active_chat = create_new_chat(active_doc, title="Chat 1")
        st.session_state["active_chat_id"] = active_chat.id

    # Title header
    st.title("📚 Book RAG Assistant")
    doc_label = active_doc.replace("_", " ")
    st.caption(f"Active Book: **{doc_label}**  |  Session: **{active_chat.title}**  |  Model: **DeepSeek R1**")

    # -------------------------------------------------------------
    # 5. Load or Cache Vector Store for Active Document
    # -------------------------------------------------------------
    if "vector_store" not in st.session_state or st.session_state["vector_store"] is None:
        target_dir = get_document_vector_dir(active_doc)
        if (target_dir / "index.faiss").exists():
            with st.spinner("Loading book vector index..."):
                st.session_state["vector_store"] = load_vector_store(save_directory=target_dir)
        else:
            st.warning(f"⚠️ Vector index for '{active_doc}' not found. Please index it from the sidebar.")
            return

    # -------------------------------------------------------------
    # 6. Render Messages for Active Chat
    # -------------------------------------------------------------
    for idx, msg in enumerate(active_chat.messages):
        with st.chat_message(msg["role"]):
            if msg.get("standalone_query"):
                st.caption(f"🔍 Contextual Search: *{msg['standalone_query']}*")

            # 1. ALWAYS render text first!
            st.markdown(msg["content"])

            # 2. Render diagrams in a clean horizontal gallery below text
            if msg.get("images"):
                render_diagram_gallery(msg["images"], key_suffix=f"prev_{idx}")

            # 3. Render retrieved chunks in expander
            if msg.get("sources"):
                with st.expander("🔍 Retrieved Top Chunks"):
                    for i, chunk in enumerate(msg["sources"], 1):
                        meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else chunk.metadata
                        text = chunk.get("page_content", "") if isinstance(chunk, dict) else chunk.page_content
                        page = meta.get("page", meta.get("page_number", "N/A"))
                        st.markdown(f"**Chunk {i} (Page {page})**")
                        st.text(text[:500] + ("..." if len(text) > 500 else ""))

    # -------------------------------------------------------------
    # 7. Chat Input and Conversational Processing
    # -------------------------------------------------------------
    if user_query := st.chat_input("Ask a question about the book..."):
        # Add User message
        active_chat.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant response with history
        with st.chat_message("assistant"):
            with st.spinner("Thinking & reasoning with DeepSeek R1..."):
                import importlib
                import src.pipeline.rag_chain
                importlib.reload(src.pipeline.rag_chain)

                pipeline = src.pipeline.rag_chain.create_rag_chain(
                    vector_store=st.session_state["vector_store"],
                    top_k=settings.top_k_chunks,
                    llm_provider=provider,
                )

                # Pass prior conversation history for contextual query reformulation
                prior_history = active_chat.messages[:-1]
                response = pipeline.query(user_query, chat_history=prior_history)

                if response.standalone_query:
                    st.caption(f"🔍 Contextual Search: *{response.standalone_query}*")

                # 1. ALWAYS display text answer first!
                st.markdown(response.answer)

                # 2. Render extracted figures in one horizontal row with full-res click-to-open
                extracted_images = getattr(response, "images", [])
                if extracted_images:
                    render_diagram_gallery(extracted_images, key_suffix="live")

                # 3. Render retrieved sources
                if response.source_documents:
                    with st.expander("🔍 Retrieved Top 3 Chunks"):
                        for i, chunk in enumerate(response.source_documents, 1):
                            page = chunk.metadata.get("page", chunk.metadata.get("page_number", "N/A"))
                            st.markdown(f"**Chunk {i} (Page {page})**")
                            st.text(chunk.page_content[:500] + ("..." if len(chunk.page_content) > 500 else ""))

        # Auto-update chat title on first question
        if active_chat.title.startswith("Chat ") or active_chat.title == "New Chat":
            clean_title = user_query.strip()[:30] + ("..." if len(user_query) > 30 else "")
            active_chat.title = clean_title

        # Serialize source docs for JSON persistence
        serializable_sources = []
        for doc in response.source_documents:
            serializable_sources.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })

        active_chat.messages.append({
            "role": "assistant",
            "content": response.answer,
            "images": extracted_images,
            "sources": serializable_sources,
            "standalone_query": response.standalone_query,
        })

        # Persist chat session to disk
        save_or_update_chat(active_doc, active_chat)
        st.rerun()


if __name__ == "__main__":
    main()
