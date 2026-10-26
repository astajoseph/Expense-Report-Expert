"""
audit_log.py
────────────
SQLite audit logging for all agent actions, approvals, denials, and email sends.

Schema:
  actions(id, timestamp, agent, action, status, detail)

Status values: "approved", "rejected", "denied", "sent", "failed"
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class AuditLog:
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.init_db()

    def init_db(self):
        """Create audit table if not exists."""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    recipient TEXT
                )
            """)
            self.conn.commit()
            logger.info(f"Audit DB initialized: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Audit DB init failed: {e}")

    def log_action(
        self,
        agent: str,
        action: str,
        status: str,
        detail: Optional[str] = None,
        recipient: Optional[str] = None,
    ):
        """Log an action to the audit table."""
        if not self.conn:
            return

        try:
            cursor = self.conn.cursor()
            timestamp = datetime.utcnow().isoformat()
            cursor.execute(
                """
                INSERT INTO actions (timestamp, agent, action, status, detail, recipient)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, agent, action, status, detail, recipient),
            )
            self.conn.commit()
            logger.debug(f"Audit logged: {agent} {action} {status}")
        except sqlite3.Error as e:
            logger.error(f"Audit log failed: {e}")

    def get_recent(self, limit: int = 20) -> List[dict]:
        """Fetch last N audit records."""
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, agent, action, status, detail, recipient
                FROM actions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "agent": row[2],
                    "action": row[3],
                    "status": row[4],
                    "detail": row[5],
                    "recipient": row[6],
                }
                for row in rows
            ]
        except sqlite3.Error as e:
            logger.error(f"Audit query failed: {e}")
            return []

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Global instance
_audit_log = None


def get_audit_log(db_path: str = "audit.db") -> AuditLog:
    """Get or create global audit log instance."""
    global _audit_log
    if _audit_log is None:
        _audit_log = AuditLog(db_path)
    return _audit_log
