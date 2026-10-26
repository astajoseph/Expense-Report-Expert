"""
unit3_memory/memory_store.py
────────────────────────────
Course Concept: Unit 3 — Agent Memory Architecture
                - Long-term Semantic Memory (Vector Store Embeddings)
                - Episodic User Corrections (Human Feedback Loop)
                - In-Context Few-Shot Retrieval

Why this concept?
Expense report categorization often encounters merchants that are ambiguous
(e.g., 'The Capital Grille' or 'Olive Garden' or 'Enterprise'). Calling an LLM
for every repeated expense is slow, expensive, and non-deterministic.
By persisting categorized expenses in a vector store (ChromaDB) with embeddings
(BAAI/bge-small-en-v1.5):
1. Exact or near-identical merchants can be resolved directly from long-term memory
   when similarity exceeds a threshold (skipping the LLM entirely).
2. For ambiguous merchants, top-k similar memories are injected as few-shot
   in-context examples into the LLM prompt, improving accuracy.
3. User corrections take highest priority and are permanently recorded in episodic
   memory with source='user_correction'.
"""

import os
# ── Sentence-Transformers Offline Caching ────────────────────────────────────
# Configure local model caching for fully offline operation after first run
hf_home = os.getenv("HF_HOME", "./huggingface_cache")
st_home = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./huggingface_cache/sentence_transformers")
os.environ.setdefault("HF_HOME", hf_home)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", st_home)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Suppress noisy external logs
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.80"))
DEFAULT_DB_PATH = os.getenv("CHROMA_DB_PATH", str(Path(__file__).parent.parent / "data" / "chroma_db"))


class ExpenseMemoryStore:
    """
    Persistent Long-Term Memory for expense categorization using ChromaDB
    and sentence-transformers embeddings (BAAI/bge-small-en-v1.5).
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        collection_name: str = "expense_memory",
    ):
        self.db_path = str(Path(db_path).resolve())
        self.similarity_threshold = similarity_threshold
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_store()

    def _init_store(self) -> None:
        """Initialize ChromaDB client and collection with BGE-small embeddings."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)

            # Use local BAAI/bge-small-en-v1.5 as per hard constraints
            try:
                emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="BAAI/bge-small-en-v1.5"
                )
            except Exception as e:
                logger.warning(f"Could not load BAAI/bge-small-en-v1.5: {e}. Using Chroma default.")
                emb_fn = embedding_functions.DefaultEmbeddingFunction()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"},
            )

            # If brand new store, seed with some canonical vendor memories
            if self._collection.count() == 0:
                self._seed_initial_memories()

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB memory store: {e}")
            self._client = None
            self._collection = None

    def _seed_initial_memories(self) -> None:
        """Seed baseline memories so initial queries demonstrate semantic retrieval."""
        seeds = [
            ("Delta Airlines", "Flight SFO to NYC round trip", "Flights", "canonical"),
            ("United Airlines", "Flight EWR to SFO", "Flights", "canonical"),
            ("Hilton Garden Inn", "Hotel stay NYC 3 nights", "Hotels", "canonical"),
            ("Marriott Marquis", "Hotel lodging downtown", "Hotels", "canonical"),
            ("Starbucks", "Coffee and morning snack", "Meals", "canonical"),
            ("The Capital Grille", "Client dinner steakhouse", "Meals", "canonical"),
            ("Uber", "Rideshare airport pickup", "Transport", "canonical"),
            ("BART", "Subway transit ticket", "Transport", "canonical"),
        ]
        for vendor, desc, cat, source in seeds:
            self.store_memory(vendor=vendor, description=desc, category=cat, source=source)

    @staticmethod
    def _make_doc_text(vendor: str, description: str) -> str:
        """Format the text representation for embedding."""
        v = (vendor or "").strip()
        d = (description or "").strip()
        return f"Vendor: {v} | Description: {d}"

    @staticmethod
    def _make_id(vendor: str, description: str, source: str) -> str:
        """Deterministic ID for memory deduplication."""
        import hashlib
        key = f"{vendor.lower().strip()}_{description.lower().strip()}_{source}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def store(
        self,
        merchant: str = "",
        vendor: str = "",
        description: str = "",
        category: str = "Misc",
        source: str = "model",
    ) -> bool:
        """Alias for store_memory supporting either merchant or vendor parameter."""
        v = vendor or merchant
        return self.store_memory(vendor=v, description=description, category=category, source=source)

    def store_memory(
        self,
        vendor: str,
        description: str,
        category: str,
        source: str = "model",
    ) -> bool:
        """
        Store a categorized expense record in persistent ChromaDB.
        source can be 'model', 'user_correction', or 'canonical'.
        """
        if not self._collection:
            return False

        doc_text = self._make_doc_text(vendor, description)
        doc_id = self._make_id(vendor, description, source)

        metadata = {
            "vendor": vendor,
            "description": description or "",
            "category": category,
            "source": source,
        }

        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            logger.error(f"Error storing memory for '{vendor}': {e}")
            return False

    def query_similar(self, vendor: str, description: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Query top-k most similar records from long-term memory.
        Returns list of dicts with: vendor, description, category, source, similarity.
        """
        if not self._collection or self._collection.count() == 0:
            return []

        doc_text = self._make_doc_text(vendor, description)
        try:
            results = self._collection.query(
                query_texts=[doc_text],
                n_results=min(top_k, self._collection.count()),
                include=["metadatas", "distances", "documents"],
            )
        except Exception as e:
            logger.error(f"Error querying memory for '{vendor}': {e}")
            return []

        matches: list[dict[str, Any]] = []
        if not results or not results.get("metadatas"):
            return matches

        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[1.0]])[0]

        for meta, dist in zip(metadatas, distances):
            # Chroma cosine distance: distance = 1 - cosine_sim, so similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            matches.append({
                "vendor": meta.get("vendor", ""),
                "description": meta.get("description", ""),
                "category": meta.get("category", "Misc"),
                "source": meta.get("source", "model"),
                "similarity": round(similarity, 4),
            })

        # Sort matches so user_corrections get prioritized if similarities are close
        def rank_key(m: dict[str, Any]) -> tuple[int, float]:
            priority = 2 if m["source"] == "user_correction" else 1
            return (priority, m["similarity"])

        matches.sort(key=rank_key, reverse=True)
        return matches

    def resolve_category(
        self, vendor: str, description: str, top_k: int = 3
    ) -> tuple[Optional[str], list[dict[str, Any]], str]:
        """
        Attempt to resolve category purely from long-term memory:
        1. If user_correction exists with high similarity -> REUSE.
        2. If top match >= similarity_threshold and top categories agree -> REUSE.
        3. Otherwise -> return None so LLM is called (with matches passed as few-shot).
        """
        matches = self.query_similar(vendor, description, top_k=top_k)
        if not matches:
            print(f"   [TRACE] TOOL: chroma.query vendor='{vendor}' -> 0 matches (skip LLM)")
            return None, [], "no_memories"

        top_match = matches[0]

        # Check for user correction priority
        for m in matches:
            if m["source"] == "user_correction" and m["similarity"] >= 0.70:
                reason = f"user_correction priority ({m['vendor']} -> {m['category']}, sim={m['similarity']:.2f})"
                print(f"   [TRACE] TOOL: chroma.query vendor='{vendor}' -> {m['category']} (user correction, sim={m['similarity']:.2f}, skip LLM)")
                return m["category"], matches, reason

        # Check threshold and category agreement among top matches
        if top_match["similarity"] >= self.similarity_threshold:
            # Check if top results agree on category
            agreeing = [m for m in matches if m["similarity"] >= self.similarity_threshold]
            cats = {m["category"] for m in agreeing}
            if len(cats) == 1:
                cat = list(cats)[0]
                reason = f"high_similarity memory match (sim={top_match['similarity']:.2f} >= {self.similarity_threshold})"
                print(f"   [TRACE] TOOL: chroma.query vendor='{vendor}' -> {cat} (memory hit, sim={top_match['similarity']:.2f}, skip LLM)")
                return cat, matches, reason

        print(f"   [TRACE] TOOL: chroma.query vendor='{vendor}' -> below threshold (sim={top_match['similarity']:.2f}, use LLM with {len(matches)} few-shots)")
        return None, matches, f"below_threshold (top sim={top_match['similarity']:.2f} < {self.similarity_threshold})"

    def add_user_correction(self, vendor: str, category: str, description: str = "") -> bool:
        """
        Record an explicit human correction with highest priority (source='user_correction').
        """
        success = self.store_memory(
            vendor=vendor,
            description=description or f"User correction for {vendor}",
            category=category,
            source="user_correction",
        )
        if success:
            logger.info(f"✔ User correction saved: '{vendor}' -> '{category}' (source=user_correction)")
        return success


# Class alias for backwards compatibility
MemoryStore = ExpenseMemoryStore

