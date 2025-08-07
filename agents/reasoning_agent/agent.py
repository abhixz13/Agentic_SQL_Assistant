"""
Updated ReasoningAgent with Visualization Support
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from agents.visualization.agent import VisualizationAgent, VisualizationOptions

class ExecutionPlan:
    """Now includes visualization options"""
    def __init__(self, sql: str, viz_options: Optional[Dict] = None, max_retries: int = 3):
        self.original_sql = sql
        self.current_sql = sql
        self.viz_options = viz_options or {}
        self.attempts = 0
        self.max_retries = max_retries
        self.errors: List = []

    def can_retry(self) -> bool:
        return self.attempts < self.max_retries and not self._has_critical_error()

    def _has_critical_error(self) -> bool:
        return any(e.error_type in ["permission"] for e in self.errors)

class ReasoningAgent:
    """Now handles visualization workflow"""
    
    def __init__(self, executor):
        self.executor = executor
        self.executor.set_error_handler(self.handle_execution_error)
        self.viz_agent = VisualizationAgent()
        self.current_plan: Optional[ExecutionPlan] = None

    def handle_execution_error(self, error):
        """Callback for QueryExecutor errors"""
        if not self.current_plan:
            return
        self.current_plan.errors.append(error)
        if error.error_type == "schema":
            self._handle_schema_error(error)

    def _handle_schema_error(self, error):
        """Handle schema-related errors"""
        print(f"\n[Schema Error Detected]")
        print(f"Error: {error.details}")
        print(f"Required: {error.required_schema}")
        print("Regenerating query with corrected schema...")
        # In full implementation, would call SchemaLoader and SQLGenerator here

    def execute_query(self, sql: str, viz_options: Optional[Dict] = None):
        """Execute query with optional visualization"""
        self.current_plan = ExecutionPlan(sql, viz_options)
        
        while self.current_plan.can_retry():
            try:
                self.current_plan.attempts += 1
                result = self.executor.execute(self.current_plan.current_sql)
                
                # Add visualization if requested
                if self.current_plan.viz_options:
                    fig = self.viz_agent.visualize(
                        data=result.data,
                        options=VisualizationOptions(**self.current_plan.viz_options)
                    )
                    return {"data": result, "visualization": fig}
                return result
                
            except Exception as e:
                if not self.current_plan.can_retry():
                    raise
        
        raise RuntimeError("Max retries exceeded") 