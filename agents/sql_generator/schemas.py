from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

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
        plan_context (str): Optional SQL execution plan context
    """
    action: str
    entity: str
    params: dict
    schema_info: dict
    additional_context: dict = {}
    plan_context: str = ""

class SQLPlan(BaseModel):
    """Structured representation of a SQL execution plan."""
    expressions: Dict[str, str]  # metric_name -> DSL expression
    group_by: List[str] = []     # table.column references
    time: Optional[Dict[str, Any]] = Field(default_factory=dict)  # time configuration - allows None and converts to {}
    filters: List[Dict[str, str]] = []  # filter conditions
    notes: List[str] = []        # planning notes
    confidence: float = 0.9      # planning confidence
    metadata: Dict[str, Any] = {}
    
    def __init__(self, **data):
        # Handle the case where time is None by converting to empty dict
        if data.get('time') is None:
            data['time'] = {}
        super().__init__(**data)

class PlanningRequest(BaseModel):
    """Request for SQL planning with optional semantic context."""
    intent: dict
    schema_info: dict  # Renamed to avoid shadowing BaseModel.schema
    semantic_context: str = ""  # Enhanced semantic context for business understanding