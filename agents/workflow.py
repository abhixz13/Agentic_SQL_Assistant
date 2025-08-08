# agents/workflow.py
from agents.query_executor import QueryExecutorAgent  # This should now work
from agents.reasoning_agent import ReasoningAgent

class SQLWorkflow:
    def __init__(self, db_path: str):
        self.executor = QueryExecutorAgent(db_path)
        self.reasoning = ReasoningAgent(self.executor)

    def run(self, sql_query: str, schema_context:str = None):
        return self.reasoning.execute_query(sql_query, schema_context or "")

