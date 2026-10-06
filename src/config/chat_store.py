"""Persistent multi-document chat session store."""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any


@dataclass
class ChatSession:
    """Represents a single chat thread belonging to a document."""

    id: str
    title: str
    created_at: str
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            title=data.get("title", "New Chat"),
            created_at=data.get("created_at", time.strftime("%Y-%m-%d %H:%M")),
            messages=data.get("messages", []),
        )


def _get_storage_path(doc_id: str, storage_dir: str | Path = "data/chats") -> Path:
    """Get json file path for a specific document's chats."""
    dir_path = Path(storage_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    clean_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in doc_id)
    return dir_path / f"{clean_id}.json"


def load_chats_for_doc(doc_id: str, storage_dir: str | Path = "data/chats") -> List[ChatSession]:
    """Load all chat sessions associated with a document ID."""
    file_path = _get_storage_path(doc_id, storage_dir)
    if not file_path.exists():
        # Initialize with one default chat session
        default_chat = create_new_chat(doc_id, title="Chat 1", storage_dir=storage_dir)
        return [default_chat]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            sessions = [ChatSession.from_dict(item) for item in data.get("chats", [])]
            if not sessions:
                default_chat = create_new_chat(doc_id, title="Chat 1", storage_dir=storage_dir)
                return [default_chat]
            return sessions
    except Exception:
        default_chat = create_new_chat(doc_id, title="Chat 1", storage_dir=storage_dir)
        return [default_chat]


def save_chats_for_doc(
    doc_id: str,
    chats: List[ChatSession],
    storage_dir: str | Path = "data/chats",
) -> None:
    """Save all chat sessions for a document to disk."""
    file_path = _get_storage_path(doc_id, storage_dir)
    payload = {
        "doc_id": doc_id,
        "chats": [c.to_dict() for c in chats],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_or_update_chat(
    doc_id: str,
    chat_session: ChatSession,
    storage_dir: str | Path = "data/chats",
) -> List[ChatSession]:
    """Insert or update a specific chat session for a document."""
    all_chats = load_chats_for_doc(doc_id, storage_dir)
    found = False
    for i, c in enumerate(all_chats):
        if c.id == chat_session.id:
            all_chats[i] = chat_session
            found = True
            break
    if not found:
        all_chats.insert(0, chat_session)

    save_chats_for_doc(doc_id, all_chats, storage_dir)
    return all_chats


def create_new_chat(
    doc_id: str,
    title: str = "New Chat",
    storage_dir: str | Path = "data/chats",
) -> ChatSession:
    """Create and persist a new chat session for a document."""
    new_id = f"chat_{str(uuid.uuid4())[:8]}"
    created_at = time.strftime("%Y-%m-%d %H:%M")
    new_session = ChatSession(
        id=new_id,
        title=title,
        created_at=created_at,
        messages=[],
    )

    file_path = _get_storage_path(doc_id, storage_dir)
    existing_chats = []
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_chats = [ChatSession.from_dict(item) for item in data.get("chats", [])]
        except Exception:
            existing_chats = []

    existing_chats.insert(0, new_session)
    save_chats_for_doc(doc_id, existing_chats, storage_dir)
    return new_session


def delete_chat(
    doc_id: str,
    chat_id: str,
    storage_dir: str | Path = "data/chats",
) -> List[ChatSession]:
    """Delete a specific chat session."""
    all_chats = load_chats_for_doc(doc_id, storage_dir)
    all_chats = [c for c in all_chats if c.id != chat_id]
    if not all_chats:
        all_chats = [create_new_chat(doc_id, title="Chat 1", storage_dir=storage_dir)]
    else:
        save_chats_for_doc(doc_id, all_chats, storage_dir)
    return all_chats
