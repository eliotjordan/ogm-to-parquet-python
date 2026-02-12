"""Database utilities for ogm-to-parquet.

Provides a DatabaseManager class for handling SQLite connections with
WAL mode, busy timeouts, and retry logic for concurrent access.
"""

import logging
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database connections with WAL mode and retry logic.

    This class provides:
    - WAL mode for better concurrent read/write access
    - Configurable busy timeout for handling locks
    - Retry logic with exponential backoff for locked databases
    - Context manager for safe connection handling
    """

    def __init__(
        self,
        db_path: Path | str,
        busy_timeout_ms: int = 30000,
        connection_timeout: float = 30.0,
    ):
        """Initialize the database manager.

        Args:
            db_path: Path to SQLite database file
            busy_timeout_ms: Milliseconds to wait for locks (default: 30s)
            connection_timeout: Connection timeout in seconds
        """
        self.db_path = Path(db_path).resolve()
        self.busy_timeout_ms = busy_timeout_ms
        self.connection_timeout = connection_timeout

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections.

        Sets up WAL mode and busy timeout for concurrent access.

        Yields:
            Configured SQLite connection

        Example:
            with db_manager.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        conn = sqlite3.connect(str(self.db_path), timeout=self.connection_timeout)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            yield conn
        finally:
            conn.close()

    def execute_with_retry(
        self,
        query: str,
        params: tuple = (),
        max_retries: int = 5,
        commit: bool = True,
    ) -> list[tuple[Any, ...]]:
        """Execute a query with retry logic for locked databases.

        Uses exponential backoff when the database is locked.

        Args:
            query: SQL query to execute
            params: Query parameters
            max_retries: Maximum retry attempts
            commit: Whether to commit after execution

        Returns:
            List of result rows (empty for non-SELECT queries)

        Raises:
            sqlite3.OperationalError: If database remains locked after all retries
        """
        for attempt in range(max_retries):
            try:
                with self.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    if commit:
                        conn.commit()
                    return results
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    sleep_time = (2**attempt) * 0.1  # Exponential backoff
                    logger.warning(f"Database locked, retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    raise
        return []

    def update_status(
        self,
        table: str,
        doc_id: str,
        status: str,
        extra_fields: dict[str, Any] | None = None,
        max_retries: int = 5,
    ) -> bool:
        """Update document status in database with retry logic.

        Args:
            table: Table name (e.g., 'cloud_derivatives', 'text_extraction')
            doc_id: Document ID to update
            status: New status value
            extra_fields: Additional fields to update (e.g., {'derivative_url': path})
            max_retries: Maximum retry attempts

        Returns:
            True if a row was updated, False if no matching row found
        """
        fields = {"status": status}
        if extra_fields:
            fields.update(extra_fields)

        set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values()) + [doc_id]
        query = f"UPDATE {table} SET {set_clause} WHERE id = ?"  # noqa: S608

        for attempt in range(max_retries):
            try:
                with self.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, tuple(values))
                    conn.commit()
                    rows_affected = cursor.rowcount
                    if rows_affected == 0:
                        logger.warning(
                            f"[{doc_id}] No row found in {table} to update status to '{status}'"
                        )
                        return False
                    logger.debug(f"[{doc_id}] Status updated to '{status}'")
                    return True
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    sleep_time = (2**attempt) * 0.1
                    logger.warning(f"Database locked, retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"[{doc_id}] Database error updating status: {e}")
                    raise
        return False

    def ensure_column(self, table: str, column: str, column_type: str, default: str = "") -> bool:
        """Add a column to a table if it doesn't exist.

        Args:
            table: Table name
            column: Column name to add
            column_type: SQL type for the column
            default: Default value clause (e.g., "DEFAULT 0")

        Returns:
            True if column was added, False if it already existed
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]

            if column not in columns:
                default_clause = f" {default}" if default else ""
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {column_type}{default_clause}"
                )
                conn.commit()
                logger.info(f"Added {column} column to {table} table")
                return True
            return False

    def table_exists(self, table: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table: Table name to check

        Returns:
            True if table exists, False otherwise
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            return cursor.fetchone() is not None

    def get_count(self, table: str, where_clause: str = "", params: tuple = ()) -> int:
        """Get count of rows in a table with optional filter.

        Args:
            table: Table name
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            params: Parameters for the WHERE clause

        Returns:
            Row count
        """
        query = f"SELECT COUNT(*) FROM {table}"  # noqa: S608
        if where_clause:
            query += f" WHERE {where_clause}"

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else 0


def get_db_connection(db_path: str | Path, timeout: float = 30.0) -> sqlite3.Connection:
    """Get a database connection with WAL mode and busy timeout.

    This is a convenience function for simple use cases where
    DatabaseManager is not needed.

    Args:
        db_path: Path to SQLite database
        timeout: Connection timeout in seconds

    Returns:
        Database connection configured for concurrent access
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn
