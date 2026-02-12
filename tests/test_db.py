"""Tests for ogm_to_parquet.db module."""

from pathlib import Path

import pytest

from ogm_to_parquet.db import DatabaseManager, get_db_connection


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    @pytest.fixture
    def db_manager(self, tmp_path: Path) -> DatabaseManager:
        """Create a DatabaseManager with a test database."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager(db_path)

        # Create a test table
        with manager.connection() as conn:
            conn.execute("""
                CREATE TABLE test_docs (
                    id TEXT PRIMARY KEY,
                    status TEXT,
                    output_url TEXT,
                    error_message TEXT
                )
            """)
            conn.execute("INSERT INTO test_docs (id, status) VALUES ('doc1', 'pending')")
            conn.execute("INSERT INTO test_docs (id, status) VALUES ('doc2', 'in_progress')")
            conn.commit()

        return manager

    def test_connection_context_manager(self, db_manager: DatabaseManager):
        """Should provide working connection via context manager."""
        with db_manager.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test_docs")
            count = cursor.fetchone()[0]

        assert count == 2

    def test_wal_mode_enabled(self, db_manager: DatabaseManager):
        """Should enable WAL mode on connection."""
        with db_manager.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]

        assert mode.lower() == "wal"

    def test_execute_with_retry_select(self, db_manager: DatabaseManager):
        """Should execute SELECT queries and return results."""
        results = db_manager.execute_with_retry(
            "SELECT id, status FROM test_docs WHERE status = ?",
            ("pending",),
        )

        assert len(results) == 1
        assert results[0] == ("doc1", "pending")

    def test_execute_with_retry_update(self, db_manager: DatabaseManager):
        """Should execute UPDATE queries."""
        db_manager.execute_with_retry(
            "UPDATE test_docs SET status = ? WHERE id = ?",
            ("complete", "doc1"),
        )

        # Verify update
        results = db_manager.execute_with_retry(
            "SELECT status FROM test_docs WHERE id = ?",
            ("doc1",),
        )
        assert results[0][0] == "complete"

    def test_update_status_success(self, db_manager: DatabaseManager):
        """Should update status for existing document."""
        result = db_manager.update_status("test_docs", "doc1", "complete")

        assert result is True

        # Verify
        with db_manager.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM test_docs WHERE id = ?", ("doc1",))
            assert cursor.fetchone()[0] == "complete"

    def test_update_status_with_extra_fields(self, db_manager: DatabaseManager):
        """Should update status with additional fields."""
        result = db_manager.update_status(
            "test_docs",
            "doc1",
            "complete",
            extra_fields={"output_url": "/path/to/output.pmtiles"},
        )

        assert result is True

        with db_manager.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, output_url FROM test_docs WHERE id = ?", ("doc1",))
            row = cursor.fetchone()
            assert row[0] == "complete"
            assert row[1] == "/path/to/output.pmtiles"

    def test_update_status_nonexistent_doc(self, db_manager: DatabaseManager):
        """Should return False for non-existent document."""
        result = db_manager.update_status("test_docs", "nonexistent", "complete")

        assert result is False

    def test_ensure_column_adds_new_column(self, db_manager: DatabaseManager):
        """Should add new column to table."""
        result = db_manager.ensure_column("test_docs", "retry_count", "INTEGER", "DEFAULT 0")

        assert result is True

        # Verify column exists
        with db_manager.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(test_docs)")
            columns = [row[1] for row in cursor.fetchall()]
            assert "retry_count" in columns

    def test_ensure_column_existing(self, db_manager: DatabaseManager):
        """Should return False for existing column."""
        result = db_manager.ensure_column("test_docs", "status", "TEXT")

        assert result is False

    def test_table_exists_true(self, db_manager: DatabaseManager):
        """Should return True for existing table."""
        result = db_manager.table_exists("test_docs")

        assert result is True

    def test_table_exists_false(self, db_manager: DatabaseManager):
        """Should return False for non-existent table."""
        result = db_manager.table_exists("nonexistent_table")

        assert result is False

    def test_get_count_all(self, db_manager: DatabaseManager):
        """Should count all rows in table."""
        count = db_manager.get_count("test_docs")

        assert count == 2

    def test_get_count_with_filter(self, db_manager: DatabaseManager):
        """Should count rows matching filter."""
        count = db_manager.get_count("test_docs", "status = ?", ("pending",))

        assert count == 1


class TestGetDbConnection:
    """Tests for get_db_connection function."""

    def test_returns_connection(self, tmp_path: Path):
        """Should return working connection."""
        db_path = tmp_path / "test.db"
        conn = get_db_connection(db_path)

        try:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")
            cursor.execute("INSERT INTO test VALUES (1)")
            conn.commit()

            cursor.execute("SELECT * FROM test")
            assert cursor.fetchone() == (1,)
        finally:
            conn.close()

    def test_wal_mode_enabled(self, tmp_path: Path):
        """Should enable WAL mode."""
        db_path = tmp_path / "test.db"
        conn = get_db_connection(db_path)

        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            conn.close()
