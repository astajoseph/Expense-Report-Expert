"""
unit3_memory/__init__.py
────────────────────────
Course Concept: Unit 3 — Agent Memory Architecture (Long-term semantic memory
                using vector store embeddings, and episodic user corrections).
"""

from .memory_store import ExpenseMemoryStore

MemoryStore = ExpenseMemoryStore
__all__ = ["ExpenseMemoryStore", "MemoryStore"]
