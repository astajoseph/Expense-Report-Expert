"""
unit4_agents/acl_message.py
───────────────────────────
Course Concept: Unit 4 — FIPA-ACL Agent Communication Language.

Why this concept?
In multi-agent systems, agents need a structured protocol to communicate intentions,
delegations, questions, and status updates. FIPA-ACL standardizes message structure:
- performative: The speech act (request, inform, refuse, propose, accept-proposal)
- sender: The originating agent role
- receiver: The target agent role
- content: Structured payload or descriptive message
- ontology: Domain knowledge domain (e.g. 'expense-reporting-v1')
- reply_with: Unique message/correlation identifier for tracking dialogues
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ACLPerformative(str, Enum):
    REQUEST = "request"
    INFORM = "inform"
    REFUSE = "refuse"
    PROPOSE = "propose"
    ACCEPT_PROPOSAL = "accept-proposal"


@dataclass
class ACLMessage:
    """Standard FIPA-ACL Message Representation."""
    performative: str
    sender: str
    receiver: str
    content: Any
    ontology: str = "expense-reporting-v1"
    reply_with: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        return {
            "performative": self.performative,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "ontology": self.ontology,
            "reply_with": self.reply_with,
        }

    def format_line(self) -> str:
        """Formatted one-liner for live terminal display."""
        perf_color = self.performative.upper()
        summary = str(self.content)
        if len(summary) > 75:
            summary = summary[:72] + "..."
        return f"[{perf_color}] {self.sender} -> {self.receiver} | {summary}"
