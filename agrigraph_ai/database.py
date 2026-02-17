import sqlite3
import json
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Database:
    """Thread-safe SQLite database for AgriGraph AI persistence."""

    def __init__(self, db_path: str = "outputs/agrigraph.db"):
        """
        Initialize database connection pool with thread-local storage.

        Args:
            db_path: Path to SQLite database file (default: outputs/agrigraph.db)
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        self._local = threading.local()

        # Ensure outputs directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema on first run
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection with row factory."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    def _execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute query with thread safety."""
        with self._lock:
            conn = self._get_connection()
            return conn.execute(query, params)

    def _commit(self) -> None:
        """Commit transaction with thread safety."""
        with self._lock:
            conn = self._get_connection()
            conn.commit()

    def init_db(self) -> None:
        """Create database schema if it doesn't exist."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create training_runs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'running',
                    seed INTEGER,
                    num_epochs INTEGER,
                    num_nodes INTEGER,
                    test_r2 REAL,
                    test_mse REAL,
                    test_mae REAL,
                    model_path TEXT,
                    data_source TEXT,
                    config_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create alerts table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    location_id TEXT,
                    x REAL,
                    y REAL,
                    risk_level TEXT,
                    risk_score REAL,
                    primary_gas TEXT,
                    gas_concentration REAL,
                    recommendation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
                )
            """
            )

            # Create epoch_history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS epoch_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL,
                    val_loss REAL,
                    train_r2 REAL,
                    val_r2 REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
                )
            """
            )

            # Create indices for better query performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_runs_job_id ON training_runs(job_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_runs_session_id ON training_runs(session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_run_id ON alerts(run_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_risk_level ON alerts(risk_level)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_epoch_history_run_id ON epoch_history(run_id)"
            )

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def save_training_run(
        self,
        job_id: str,
        session_id: str,
        seed: Optional[int] = None,
        num_epochs: Optional[int] = None,
        num_nodes: Optional[int] = None,
        data_source: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a new training run record.

        Args:
            job_id: Unique job identifier
            session_id: Session identifier
            seed: Random seed
            num_epochs: Number of training epochs
            num_nodes: Number of graph nodes
            data_source: Data source path/description
            config: Configuration dictionary

        Returns:
            run_id: Database ID of the new training run
        """
        config_json = json.dumps(config) if config else None
        started_at = datetime.now().isoformat()

        cursor = self._execute(
            """
            INSERT INTO training_runs
            (job_id, session_id, started_at, seed, num_epochs, num_nodes, data_source, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                session_id,
                started_at,
                seed,
                num_epochs,
                num_nodes,
                data_source,
                config_json,
            ),
        )
        self._commit()
        run_id = cursor.lastrowid
        logger.info(f"Created training run {run_id} with job_id {job_id}")
        return run_id

    def update_training_run(
        self,
        job_id: str,
        status: Optional[str] = None,
        completed_at: Optional[str] = None,
        test_r2: Optional[float] = None,
        test_mse: Optional[float] = None,
        test_mae: Optional[float] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        """
        Update training run with results.

        Args:
            job_id: Job identifier to update
            status: Training status (running, completed, failed)
            completed_at: Completion timestamp
            test_r2: Test R² score
            test_mse: Test mean squared error
            test_mae: Test mean absolute error
            model_path: Path to saved model

        Returns:
            True if update was successful, False otherwise
        """
        # Build dynamic update query
        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)
        elif status == "completed":
            updates.append("completed_at = ?")
            params.append(datetime.now().isoformat())

        if test_r2 is not None:
            updates.append("test_r2 = ?")
            params.append(test_r2)

        if test_mse is not None:
            updates.append("test_mse = ?")
            params.append(test_mse)

        if test_mae is not None:
            updates.append("test_mae = ?")
            params.append(test_mae)

        if model_path is not None:
            updates.append("model_path = ?")
            params.append(model_path)

        if not updates:
            return False

        params.append(job_id)
        query = f"UPDATE training_runs SET {', '.join(updates)} WHERE job_id = ?"

        cursor = self._execute(query, tuple(params))
        self._commit()

        if cursor.rowcount > 0:
            logger.info(f"Updated training run {job_id}")
            return True
        else:
            logger.warning(f"No training run found with job_id {job_id}")
            return False

    def save_alerts(self, run_id: int, alerts: List[Dict[str, Any]]) -> int:
        """
        Save alerts for a training run.

        Args:
            run_id: Training run ID
            alerts: List of alert dictionaries

        Returns:
            Number of alerts saved
        """
        cursor = self._get_connection().cursor()

        for alert in alerts:
            cursor.execute(
                """
                INSERT INTO alerts
                (run_id, location_id, x, y, risk_level, risk_score, primary_gas, gas_concentration, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    alert.get("location_id"),
                    alert.get("x"),
                    alert.get("y"),
                    alert.get("risk_level"),
                    alert.get("risk_score"),
                    alert.get("primary_gas"),
                    alert.get("gas_concentration"),
                    alert.get("recommendation"),
                ),
            )

        self._commit()
        logger.info(f"Saved {len(alerts)} alerts for run {run_id}")
        return len(alerts)

    def save_epoch_history(
        self,
        run_id: int,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_r2: Optional[float] = None,
        val_r2: Optional[float] = None,
    ) -> bool:
        """
        Save epoch training metrics.

        Args:
            run_id: Training run ID
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_r2: Training R² score
            val_r2: Validation R² score

        Returns:
            True if save was successful
        """
        cursor = self._execute(
            """
            INSERT INTO epoch_history (run_id, epoch, train_loss, val_loss, train_r2, val_r2)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, epoch, train_loss, val_loss, train_r2, val_r2),
        )
        self._commit()

        if cursor.rowcount > 0:
            logger.debug(f"Saved epoch {epoch} metrics for run {run_id}")
            return True
        return False

    def get_training_runs(
        self,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Fetch training run history.

        Args:
            session_id: Filter by session ID (optional)
            status: Filter by status (optional)
            limit: Maximum number of runs to return
            offset: Number of runs to skip

        Returns:
            List of training run dictionaries
        """
        query = "SELECT * FROM training_runs WHERE 1=1"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._execute(query, tuple(params))
        runs = []

        for row in cursor.fetchall():
            run = dict(row)
            # Parse JSON config if present
            if run.get("config_json"):
                try:
                    run["config"] = json.loads(run["config_json"])
                except json.JSONDecodeError:
                    run["config"] = None
            runs.append(run)

        return runs

    def get_run_by_job_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get training run details by job ID.

        Args:
            job_id: Job identifier

        Returns:
            Training run dictionary or None if not found
        """
        cursor = self._execute(
            "SELECT * FROM training_runs WHERE job_id = ?", (job_id,)
        )
        row = cursor.fetchone()

        if row:
            run = dict(row)
            if run.get("config_json"):
                try:
                    run["config"] = json.loads(run["config_json"])
                except json.JSONDecodeError:
                    run["config"] = None
            return run

        return None

    def get_run_alerts(
        self,
        run_id: int,
        risk_level: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get alerts for a training run.

        Args:
            run_id: Training run ID
            risk_level: Filter by risk level (optional)
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        query = "SELECT * FROM alerts WHERE run_id = ?"
        params = [run_id]

        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level)

        query += " ORDER BY risk_score DESC LIMIT ?"
        params.append(limit)

        cursor = self._execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_epoch_history(self, run_id: int) -> List[Dict[str, Any]]:
        """
        Get epoch training history for a run.

        Args:
            run_id: Training run ID

        Returns:
            List of epoch history dictionaries
        """
        cursor = self._execute(
            """
            SELECT * FROM epoch_history
            WHERE run_id = ?
            ORDER BY epoch ASC
            """,
            (run_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def cleanup_old_runs(self, days: int = 30) -> int:
        """
        Remove training runs older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of runs deleted
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor = self._execute(
            """
            DELETE FROM training_runs
            WHERE created_at < ?
            """,
            (cutoff_date,),
        )
        self._commit()

        rows_deleted = cursor.rowcount
        logger.info(f"Cleaned up {rows_deleted} training runs older than {days} days")
        return rows_deleted

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {}

        # Count total runs
        cursor = self._execute("SELECT COUNT(*) as count FROM training_runs")
        stats["total_runs"] = cursor.fetchone()["count"]

        # Count runs by status
        cursor = self._execute(
            "SELECT status, COUNT(*) as count FROM training_runs GROUP BY status"
        )
        stats["runs_by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

        # Count total alerts
        cursor = self._execute("SELECT COUNT(*) as count FROM alerts")
        stats["total_alerts"] = cursor.fetchone()["count"]

        # Count alerts by risk level
        cursor = self._execute(
            "SELECT risk_level, COUNT(*) as count FROM alerts GROUP BY risk_level"
        )
        stats["alerts_by_risk"] = {row["risk_level"]: row["count"] for row in cursor.fetchall()}

        # Average test metrics
        cursor = self._execute(
            """
            SELECT AVG(test_r2) as avg_r2, AVG(test_mse) as avg_mse, AVG(test_mae) as avg_mae
            FROM training_runs WHERE test_r2 IS NOT NULL
            """
        )
        row = cursor.fetchone()
        stats["avg_metrics"] = {
            "r2": row["avg_r2"],
            "mse": row["avg_mse"],
            "mae": row["avg_mae"],
        }

        return stats

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if hasattr(self._local, "connection") and self._local.connection:
                self._local.connection.close()
                self._local.connection = None


# Global database singleton
db = Database()
