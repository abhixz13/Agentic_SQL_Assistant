from pydantic import BaseModel

class UserIntent(BaseModel):
    """
    Structured representation of a parsed user query.
    
    Attributes:
        action (str): The operation to perform (e.g., 'filter', 'aggregate').
        entity (str): The target database entity (e.g., 'bookings').
        params (dict): Key-value pairs for the operation (e.g., {'city': 'New York'}).
        confidence (float): LLM's confidence score (0-1) for the parsed intent.
    """
    action: str               
    entity: str               
    params: dict              
    confidence: float = 0.9   

class ExecutionPlan(BaseModel):
    """
    Blueprint for executing a user intent across specialized agents.
    
    Attributes:
        steps (list[dict]): Ordered tasks for agents (e.g., [{'agent': 'SchemaLoader', 'task': '...'}]).
        priority (int): Execution priority (higher = more urgent).
    """
    steps: list[dict]         
    priority: int = 1         