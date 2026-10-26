"""
access_control.py
─────────────────
Role-based access control (RBAC) enforcement for agents.

Roles:
- Retriever: read_csv only
- Categorizer: read, categorize
- Auditor: read, flag, cannot send
- Reporter: all permissions (including send_email after human approval)

Every denied action is logged to ACL trace and audit.db.
"""

from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Role(Enum):
    COORDINATOR = "coordinator"
    RETRIEVER = "retriever"
    CATEGORIZER = "categorizer"
    AUDITOR = "auditor"
    REPORTER = "reporter"


class Permission(Enum):
    READ_CSV = "read_csv"
    READ_STATE = "read_state"
    CATEGORIZE = "categorize"
    FLAG = "flag"
    CALCULATE = "calculate"
    GENERATE_PDF = "generate_pdf"
    DRAFT_EMAIL = "draft_email"
    SEND_EMAIL = "send_email"


# Role-to-permissions mapping
ROLE_PERMISSIONS = {
    Role.COORDINATOR: {Permission.READ_STATE},
    Role.RETRIEVER: {Permission.READ_CSV, Permission.READ_STATE},
    Role.CATEGORIZER: {Permission.READ_STATE, Permission.CATEGORIZE},
    Role.AUDITOR: {Permission.READ_STATE, Permission.FLAG},
    Role.REPORTER: {
        Permission.READ_STATE,
        Permission.CALCULATE,
        Permission.GENERATE_PDF,
        Permission.DRAFT_EMAIL,
        Permission.SEND_EMAIL,
    },
}


def check_permission(agent_role: Role, permission: Permission) -> bool:
    """Check if agent has permission. Returns False if denied."""
    allowed = ROLE_PERMISSIONS.get(agent_role, set())
    has_perm = permission in allowed
    if not has_perm:
        logger.warning(
            f"ACCESS DENIED: {agent_role.value} cannot {permission.value}"
        )
    return has_perm


def enforce_permission(
    agent_role: Role, permission: Permission, action_detail: str = ""
) -> None:
    """Raise exception if permission denied. Used in critical paths."""
    if not check_permission(agent_role, permission):
        raise PermissionError(
            f"{agent_role.value} denied {permission.value}. {action_detail}"
        )


def validate_recipient(recipient: str, allowlist: Optional[list[str]] = None) -> bool:
    """Validate email recipient against allowlist."""
    if allowlist is None:
        import os
        finance_email = os.getenv("FINANCE_EMAIL", "")
        allowlist = [finance_email] if finance_email else []

    if not allowlist:
        logger.warning("No email allowlist configured.")
        return False

    is_allowed = recipient in allowlist
    if not is_allowed:
        logger.warning(f"EMAIL RECIPIENT BLOCKED: {recipient} not in allowlist {allowlist}")
    return is_allowed
