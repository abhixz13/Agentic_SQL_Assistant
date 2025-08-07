from pydantic import BaseModel
from typing import Optional, Dict, Any

class SQLQuery(BaseModel):
    """
    Structured representation of a generated SQL query.
    
    Attributes:
        sql (str): The actual SQL query string
        query_type (str): Type of query (SELECT, INSERT, UPDATE, DELETE)
        table_name (str): Primary table being queried
        columns (list): List of columns involved
        conditions (dict): WHERE conditions applied
        confidence (float): Confidence score for the generated SQL
        metadata (dict): Additional query information
    """
    sql: str
    query_type: str
    table_name: str
    columns: list[str] = []
    conditions: dict = {}
    confidence: float = 0.9
    metadata: dict = {}

class SQLGenerationRequest(BaseModel):
    """
    Input request for SQL generation.
    
    Attributes:
        action (str): The intended SQL operation (filter, aggregate, select, etc.)
        entity (str): Target table name
        params (dict): Parameters for the operation
        schema_info (dict): Database schema information
        additional_context (dict): Any additional context for generation
    """
    action: str
    entity: str
    params: dict
    schema_info: dict
    additional_context: dict = {} 