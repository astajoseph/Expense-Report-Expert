"""
unit4_agents/__init__.py
────────────────────────
Course Concept: Unit 4 — Multi-Agent Systems
                - Role-Based Agent Specialization
                - FIPA-ACL Agent Communication Language
                - Shared Blackboard Architecture
                - Human Approval Gate (Adjustable Autonomy)
"""

from .acl_message import ACLMessage, ACLPerformative

__all__ = ["ACLMessage", "ACLPerformative"]
