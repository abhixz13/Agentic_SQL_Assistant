"""
Updated ReasoningAgent with Visualization Support

Reasoning Agent - Retry, Classify and Repair
"""
import os, re, json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from agents.visualization.agent import VisualizationAgent, VisualizationOptions
from agents.base import Agent
from services.validation_engine import ValidationEngine

class ExecutionPlan:
    """Tracks current SQL, retries, errors, and optional visualization options."""
    def __init__(self, sql: str, viz_options: Optional[Dict[str, Any]] = None, max_retries: int = 3):
        self.original_sql = sql
        self.current_sql = sql
        self.viz_options = viz_options or {}
        self.attempts = 0
        self.max_retries = max_retries
        self.errors: List[Any] = []

    def can_retry(self) -> bool:
        return self.attempts < self.max_retries and not self._has_critical_error()

    def _has_critical_error(self) -> bool:
        # Bail on non-retryable conditions if executor marks them
        return any(getattr(e, "error_type", "") in ("permission", "auth", "timeout_hard") for e in self.errors)

class ReasoningAgent(Agent):
    """
    Execute → observe → repair → re-execute (MAGIC-style).
    - Bounded retries with a cheap probe (LIMIT 1) for SELECTs
    - Error classification -> targeted checklist prompts (schema-aware)
    - Light semantic tweaks (e.g., add ORDER BY for top-N)
    - Preserves your visualization flow
    """
    def __init__(self, executor, max_retries: int = 3):
        self.executor = executor
        self.max_retries = max_retries
        if hasattr(self.executor, "set_error_handler"):
            self.executor.set_error_handler(self.handle_execution_error)
        self.viz_agent = VisualizationAgent()
        self.validation_engine = ValidationEngine()
        self.current_plan: Optional[ExecutionPlan] = None

    def run(self, payload: str, context: str):
        return self.execute_query(payload, context)

    # ===== Public entry point =====
    def execute_query(
        self,
        sql: str,
        schema_context: str = "",
        viz_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Run SQL with guided retries using schema-aware fixes.
        Returns the executor result; if viz_options provided, returns dict with {'data', 'visualization'}.
        """
        self.current_plan = ExecutionPlan(sql, viz_options, max_retries=self.max_retries)
        last_sql = self.current_plan.current_sql

        while self.current_plan.can_retry():
            try:
                self.current_plan.attempts += 1

                # 1) Probe run to catch syntax/schema cheaply
                probe_sql, probed = self._with_probe_limit(last_sql)
                probe_result = self.executor.execute(probe_sql)

                # 2) If probed, run the full query now
                result = probe_result
                if probed:
                    result = self.executor.execute(self._ensure_terminated(last_sql))

                # 3) Optional semantic adjustment (top-N, empty results, etc.)
                if self._needs_semantic_adjustment(last_sql, result):
                    fixed = self._fix_semantic(last_sql, schema_context, result_size=len(getattr(result, "data", [])))
                    if fixed and fixed.strip() != last_sql.strip():
                        last_sql = fixed.strip()
                        self.current_plan.current_sql = last_sql
                        continue  # retry with adjusted SQL

                # 4) Visualization (kept exactly like your flow)
                if self.current_plan.viz_options:
                    fig = self.viz_agent.visualize(
                        data=result.data,
                        options=VisualizationOptions(**self.current_plan.viz_options)
                    )
                    return {"data": result, "visualization": fig}

                return result

            except Exception as e:
                # Track error
                self.current_plan.errors.append(e)

                if not self.current_plan.can_retry():
                    break

                # 5) Classify & repair
                err_msg = str(e)
                err_type = self.validation_engine._classify_error(err_msg)
                repaired_sql = self.validation_engine.repair(last_sql, err_msg, schema_context)
                if repaired_sql and repaired_sql.strip() != last_sql.strip():
                    # Optional: log for guideline mining
                    self._log_guideline_case(
                        failure_type=err_type,
                        question=None,  # pass the user question here if you have it upstream
                        bad_sql=last_sql,
                        fixed_sql=repaired_sql,
                    )
                    last_sql = repaired_sql.strip()
                    self.current_plan.current_sql = last_sql
                    continue
                else:
                    break  # nothing to repair or repair failed

        # Final attempt (no probe) so user sees true outcome/error
        return self.executor.execute(self._ensure_terminated(self.current_plan.current_sql or sql))

    # ===== Optional: executor error callback you already supported =====
    def handle_execution_error(self, error):
        if not self.current_plan:
            return
        self.current_plan.errors.append(error)
        if getattr(error, "error_type", "") == "schema":
            print(f"\n[Schema Error Detected]\nError: {getattr(error, 'details', '')}")
            print(f"Required: {getattr(error, 'required_schema', '')}")
            print("Regenerating query with corrected schema...")

    # ===== Internals =====
    def _ensure_terminated(self, sql: str) -> str:
        s = sql.strip()
        return s if s.endswith(";") else s + ";"

    def _with_probe_limit(self, sql: str):
        """
        Return (probe_sql, probed_flag). Probe only SELECTs with no existing LIMIT.
        """
        s = sql.strip().rstrip(";")
        if not s.lower().startswith("select"):
            return self._ensure_terminated(s), False
        if re.search(r"\blimit\s+\d+\b", s, flags=re.I):
            return self._ensure_terminated(s), False
        return self._ensure_terminated(s + " LIMIT 1"), True


    def _needs_semantic_adjustment(self, sql: str, result) -> bool:
        """
        Example heuristic: LIMIT present but no ORDER BY; or LIMIT with empty results (common top-N pitfall).
        """
        s = sql.lower()
        data_len = len(getattr(result, "data", []))
        if " limit " in s and " order by " not in s:
            return True
        if " limit " in s and data_len == 0:
            return True
        return False

    def _fix_semantic(self, sql: str, schema_context: str, result_size: int) -> Optional[str]:
        """
        Deterministic tweak without an LLM:
        - If LIMIT present but no ORDER BY, add DESC on a likely metric.
        """
        if " limit " in sql.lower() and " order by " not in sql.lower():
            metric_candidates = ["total_price", "revenue", "sales", "amount", "total", "quantity"]
            for m in metric_candidates:
                if re.search(rf"\b{re.escape(m)}\b", sql, flags=re.I):
                    s = sql.strip().rstrip(";")
                    match = re.split(r"(?i)\s+limit\s+", s, maxsplit=1)
                    if len(match) == 2:
                        prefix, limit_val = match
                        return f"{prefix} ORDER BY {m} DESC LIMIT {limit_val};"
                    return f"{s} ORDER BY {m} DESC;"
        return None

    def _log_guideline_case(self, failure_type: str, question: Optional[str], bad_sql: str, fixed_sql: str):
        """
        Optional: append a case to a JSONL file for future guideline mining (MAGIC-style).
        """
        try:
            entry = {
                "failure_type": failure_type,
                "question": question,
                "bad_sql": bad_sql,
                "fixed_sql": fixed_sql,
                "ask_myself": [
                    "Do all referenced tables/columns exist in the schema?",
                    "If SELECT mixes aggregates + non-aggregates, did I add GROUP BY?",
                    "Are aggregates applied to numeric fields?",
                    "Does top-N include ORDER BY metric DESC and LIMIT N?"
                ]
            }
            with open("reasoning_guideline.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
