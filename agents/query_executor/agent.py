"""
Query Executor Agent Module

This module contains the QueryExecutorAgent that executes SQL queries
with error handling and retry logic.
"""

import sqlite3
import time
from typing import Callable, Optional, List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

class ExecutionError(BaseModel):
    """Error information for query execution"""
    error_type: str  # "schema", "connection", "data", "permission"
    details: str
    query: str
    required_schema: Optional[Dict[str, List[str]]] = None
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SQLResult(BaseModel):
    """Result of SQL query execution"""
    data: List[Dict]
    execution_time: float
    schema_used: Dict

class QueryExecutorAgent:
    """
    Executes SQL queries with error handling and retry logic.
    
    Features:
    - Automatic retry for connection issues
    - Schema error detection and reporting
    - Performance monitoring
    - Error classification and handling
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.error_handler: Optional[Callable[[ExecutionError], None]] = None
        
    def set_error_handler(self, handler: Callable[[ExecutionError], None]):
        """Set callback for error handling"""
        self.error_handler = handler

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=lambda e: isinstance(e, sqlite3.OperationalError) and "locked" in str(e)
    )
    def execute(self, query: str) -> SQLResult:
        """Execute SQL query with error handling"""
        start_time = time.time()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query)
                results = [dict(row) for row in cursor.fetchall()]
                return SQLResult(
                    data=results,
                    execution_time=time.time() - start_time,
                    schema_used=self._get_table_schema(query, conn)
                )
        except sqlite3.Error as e:
            error = self._create_execution_error(e, query)
            if error.error_type == "schema" and self.error_handler:
                self.error_handler(error)
            raise

    def _create_execution_error(self, error: sqlite3.Error, query: str) -> ExecutionError:
        """Create structured error information"""
        error_type = self._classify_error(error)
        return ExecutionError(
            error_type=error_type,
            details=str(error),
            query=query,
            required_schema=self._extract_required_schema(error),
            suggested_fix=self._suggest_fix(error_type)
        )

    def _classify_error(self, error: sqlite3.Error) -> str:
        """Classify error type for appropriate handling"""
        error_msg = str(error).lower()
        if any(msg in error_msg for msg in ["no such table", "no column"]):
            return "schema"
        elif "locked" in error_msg or "busy" in error_msg:
            return "connection"
        elif "permission" in error_msg:
            return "permission"
        return "data"

    def _extract_required_schema(self, error: sqlite3.Error) -> Dict[str, List[str]]:
        """Extract missing schema elements from error"""
        error_msg = str(error)
        if "no such table" in error_msg:
            table = error_msg.split(":")[-1].strip()
            return {"missing_tables": [table]}
        elif "no column" in error_msg:
            col = error_msg.split(":")[-1].strip()
            return {"missing_columns": [col]}
        return {}

    def _suggest_fix(self, error_type: str) -> Optional[str]:
        """Suggest fixes based on error type"""
        suggestions = {
            "schema": "Verify table/column names in schema",
            "connection": "Retrying with exponential backoff",
            "permission": "Check database user permissions"
        }
        return suggestions.get(error_type)

    def _get_table_schema(self, query: str, conn: sqlite3.Connection) -> Dict:
        """Get schema information for executed query"""
        return {"tables": ["bookings"]}  # Simplified for demo

