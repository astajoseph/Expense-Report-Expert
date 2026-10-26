"""
llm_config.py
─────────────
Course concept: LOCAL / OPEN-SOURCE LLMs with graceful fallback.

This module is the single source-of-truth for LLM initialisation.
It uses ONLY local Ollama — no paid APIs, no rate limits, fully offline.

Priority chain:
  1. Try local Ollama (model from OLLAMA_MODEL env var, default qwen2.5:7b)
  2. If Ollama is unavailable → return None and signal "fallback mode"

The rest of the pipeline handles None gracefully:
  - categorize step → rule-based keyword categorisation
  - draft_email step → professional template email
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ────────────────────────────────────────────────────────────────────────────
# Internal helper
# ────────────────────────────────────────────────────────────────────────────

def _try_ollama() -> tuple:
    """
    Attempt to create a local Ollama ChatLLM.
    Tries langchain_ollama (newer) then langchain_community fallback.
    Returns (llm, label) or raises an exception.
    """
    model    = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Try the dedicated langchain_ollama package first
    try:
        from langchain_ollama import ChatOllama  # type: ignore
        logger.info("   Using langchain_ollama.ChatOllama")
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
            logger.info("   Using langchain_community ChatOllama (legacy)")
        except ImportError:
            raise ImportError(
                "Cannot import ChatOllama. Install with:\n"
                "  pip install langchain-ollama\n"
                "OR\n"
                "  pip install langchain-community"
            )

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,   # deterministic — essential for strict JSON output
        format="json",     # request JSON output from the model
    )

    # Quick connectivity + model-availability check
    llm.invoke("Reply with: {\"ok\": true}")
    return llm, f"Local Ollama ({model})"


# ────────────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────────────

def get_llm() -> tuple:
    """
    Returns a ready-to-use LangChain chat model and a human-readable label.

    Always tries local Ollama only (no cloud providers).
    If Ollama is unavailable, returns (None, label) — callers must handle this.

    Never raises — logs the error and returns (None, ...) for offline safety.
    """
    logger.info("   Attempting Local Ollama …")
    try:
        llm, label = _try_ollama()
        logger.info("   ✔ LLM provider: %s", label)
        return llm, label
    except Exception as exc:
        logger.warning(
            "   ⚠ Ollama unavailable: %s\n"
            "   → FALLBACK MODE: rule-based categorisation + template email.\n"
            "   → To fix: run `ollama serve` and `ollama pull %s`",
            exc,
            os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        )
        return None, "None — Ollama unavailable (fallback mode)"
