from __future__ import annotations

"""Central validation and repair service used across agents.

This module provides a `ValidationEngine` that wraps the existing
`SQLValidatorAgent` and the SQL repair logic previously embedded in
`ReasoningAgent._repair_sql`.  Agents should import and use this service for
plan validation and SQL error repair instead of implementing their own logic.
"""

from typing import Any, Dict, Optional, Tuple
import re

from agents.sql_generator.validator import SQLValidatorAgent
from services.llm_service import LLMService

try:  # optional dependency used for accounting
    from utils.token_tracker import record_openai_usage_from_response
except Exception:  # pragma: no cover - best effort
    record_openai_usage_from_response = lambda _resp: None


class ValidationEngine:
    """Wrapper around SQL plan validation and SQL repair helpers."""

    def __init__(self):
        self._validator = SQLValidatorAgent()

    # ------------------------------------------------------------------
    # Plan validation
    # ------------------------------------------------------------------
    def validate(self, plan: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        """Validate a DSL plan against ``schema``.

        Returns the normalized plan and a ValidationReport.
        """
        return self._validator.validate_plan(plan, schema)

    # ------------------------------------------------------------------
    # SQL repair
    # ------------------------------------------------------------------
    def repair(self, sql: str, error: str, schema_context: str) -> Optional[str]:
        """Attempt to repair ``sql`` given ``error`` and ``schema_context``.

        Returns a fixed SQL string or ``None`` if repair fails.
        """
        err_type = self._classify_error(error)

        system = {
            "role": "system",
            "content": (
                "You are a SQL fixer. Return ONE corrected SQLite query only. "
                "No markdown, no explanations."
            ),
        }
        user = {
            "role": "user",
            "content": self._prompt_for(err_type, schema_context, sql, error),
        }

        try:
            resp = LLMService.invoke(
                model="gpt-3.5-turbo",
                messages=[system, user],
                temperature=0.0,
                max_tokens=400,
            )
            record_openai_usage_from_response(resp)
            fixed = resp.choices[0].message.content.strip()
            if re.match(r"^\s*(select|with|update|delete|insert)\b", fixed, flags=re.I):
                return fixed
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _classify_error(self, msg: str) -> str:
        m = msg.lower()
        if "no such table" in m or "no such column" in m:
            return "schema"
        if "syntax error" in m:
            return "syntax"
        if "misuse of aggregate" in m or "group by" in m:
            return "aggregate"
        if "ambiguous column" in m:
            return "ambiguous"
        return "semantic"

    def _prompt_for(self, err_type: str, schema_context: str, sql: str, error_msg: str) -> str:
        common = f"""Schema:
{schema_context}

Previous SQL:
{sql}

Error:
{error_msg}
"""
        if err_type == "schema":
            checklist = (
                "- Use only tables/columns present in the Schema.\n"
                "- If a column is missing, pick the closest semantically relevant existing column.\n"
                "- Preserve the original intent: keep aggregates, filters, and grouping.\n"
            )
        elif err_type == "aggregate":
            checklist = (
                "- If SELECT mixes aggregates and non-aggregates, add GROUP BY for all non-aggregated columns.\n"
                "- Apply SUM/AVG/COUNT only to numeric columns.\n"
            )
        elif err_type == "ambiguous":
            checklist = (
                "- Qualify columns with table aliases.\n"
                "- Join only on identically named key columns (e.g., *_id) that exist in both tables.\n"
            )
        elif err_type == "syntax":
            checklist = (
                "- Fix SQL syntax without changing intent.\n"
                "- Ensure correct order/placement of WHERE, GROUP BY, ORDER BY, LIMIT.\n"
            )
        else:
            checklist = (
                "- For top-N, ensure ORDER BY a metric DESC and include LIMIT N.\n"
                "- If empty results, relax strict equality to LIKE or >= where reasonable.\n"
                "- If too many rows, add LIMIT 100 with ORDER BY a sensible metric.\n"
            )

        return (
            f"{common}Fix the SQL using the checklist:\n\n"  # noqa: W503 - readability
            f"Checklist:\n{checklist}"
            "Return only ONE corrected SQLite query (no markdown, no prose)."
        )
