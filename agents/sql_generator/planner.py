"""
SQL Planning Agent Module

This module contains the SQLPlannerAgent that converts structured intents and schema
into SQL planning DSL before feeding to the SQL generator. Adapted from external
planner code to integrate with the existing SQL_Assistant_2 architecture.
"""

import json
import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.base import Agent
from services.llm_service import LLMService
from pydantic import BaseModel
from .schemas import SQLPlan, PlanningRequest

from utils.token_tracker import record_openai_usage_from_response
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ------------------------------
# SQL Planning Schemas
# ------------------------------

# Use SQLPlan and PlanningRequest from schemas.py to avoid duplication

# ------------------------------
# Core Function Catalog (SQLite-optimized)
# ------------------------------

SQLITE_FUNCTION_CATALOG: List[Dict[str, str]] = [
    # Basic aggregations
    {"name": "SUM", "desc": "Sum a numeric column/expression."},
    {"name": "AVG", "desc": "Average of a numeric column/expression."},
    {"name": "COUNT", "desc": "Row count or count of non-null values."},
    {"name": "COUNT_DISTINCT", "desc": "Count of distinct values."},
    {"name": "MIN", "desc": "Minimum value of a column."},
    {"name": "MAX", "desc": "Maximum value of a column."},
    
    # Mathematical operations
    {"name": "SAFE_DIV", "desc": "Divide x by y, guarding division by zero."},
    {"name": "RATIO_OF_SUMS", "desc": "SUM(numerator) / SUM(denominator); default for rates."},
    {"name": "PCT_OF_TOTAL", "desc": "Share of a grand total across the whole result."},
    {"name": "CLAMP_0_1", "desc": "Clamp a value to [0,1]. Use for rates/percentages."},
    
    # Time operations (SQLite compatible)
    {"name": "TIME_BUCKET", "desc": "Truncate time to a grain: day|week|month|quarter|year using strftime."},
    {"name": "DATE_FILTER", "desc": "Time filter using SQLite date functions."},
    
    # Window functions (basic)
    {"name": "RUNNING_SUM", "desc": "Cumulative sum over time (with optional partitions)."},
    {"name": "LAG", "desc": "Previous row's value by time (with optional partitions)."},
    {"name": "RANK", "desc": "Rank values within partitions."},
    
    # String operations
    {"name": "CONCAT", "desc": "Concatenate string values."},
    {"name": "UPPER", "desc": "Convert to uppercase."},
    {"name": "LOWER", "desc": "Convert to lowercase."},
]

# ------------------------------
# SQL Planner Agent
# ------------------------------

class SQLPlannerAgent(Agent):
    """
    Converts structured intents and schema into SQL planning DSL.
    
    The planner creates an intermediate representation that is more structured
    than direct SQL generation, making it easier to validate and optimize
    before actual SQL generation.
    
    Features:
    - Intent-to-DSL conversion
    - Schema-aware planning
    - Function catalog validation
    - SQLite-optimized operations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SQL Planning Agent.
        
        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
        """
        if not (api_key or os.getenv("OPENAI_API_KEY")):
            raise ValueError("OpenAI API key not found in .env or arguments")

        self.model = "gpt-4o-mini"  # Cost-effective for planning
        self.function_catalog = SQLITE_FUNCTION_CATALOG
        self.rules = [
            "Use only columns and tables present in the provided schema.",
            "Prefer RATIO_OF_SUMS for rates and percentages.",
            "Wrap any division using SAFE_DIV to avoid division by zero.",
            "Use TIME_BUCKET for time-based grouping with SQLite date functions.",
            "Output must be valid JSON matching the required schema.",
            "Do NOT output SQL - only DSL planning expressions.",
            "Choose the simplest correct expression using available functions.",
        ]
        
        logger.info("SQLPlannerAgent initialized with OpenAI GPT-4o-mini")
    
    def create_plan(self, intent_data: Dict[str, Any], schema_info: Dict[str, Any]) -> SQLPlan:
        """
        Create SQL execution plan from intent and schema.
        
        Args:
            intent_data: Structured intent from IntentParserAgent
            schema_info: Database schema information (may include semantic_schema)
            
        Returns:
            SQLPlan: Structured SQL execution plan
        """
        try:
            # Check if semantic schema is available
            semantic_context = ""
            if schema_info.get("has_semantic") and schema_info.get("semantic_context"):
                semantic_context = schema_info["semantic_context"]
                logger.info("🧠 Using semantic schema context for enhanced planning")
            else:
                logger.info("📊 Using basic schema for planning")
            
            # Create planning request with enhanced context
            request = PlanningRequest(
                intent=intent_data,
                schema_info=schema_info,
                semantic_context=semantic_context
            )
            
            logger.info(f"Creating SQL plan for action: {intent_data.get('action')}, entity: {intent_data.get('entity')}")
            
            # Generate plan using LLM with semantic awareness
            plan_result = self._generate_plan_with_llm(request)
            
            # Validate and enhance the plan
            validated_plan = self._validate_and_enhance_plan(plan_result, schema_info)
            
            return validated_plan
            
        except Exception as e:
            logger.error(f"Error creating SQL plan: {e}")
            return self._fallback_plan_generation(intent_data, schema_info)
    
    def _generate_plan_with_llm(self, request: PlanningRequest) -> SQLPlan:
        """Generate SQL plan using OpenAI LLM with semantic awareness."""
        # Use semantic context if available, otherwise basic schema
        if request.semantic_context:
            schema_context = request.semantic_context
            logger.info("🧠 Using enhanced semantic context for SQL planning")
        else:
            schema_context = self._create_schema_context(request.schema_info)
            logger.info("📊 Using basic schema context for SQL planning")
        
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(request.intent, schema_context)
        
        try:
            response = LLMService.invoke(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            
            # Record token usage
            record_openai_usage_from_response(response)
            
            # Parse response
            content = response.choices[0].message.content.strip()
            plan_data = json.loads(content)
            
            # Convert to SQLPlan object
            return SQLPlan(**plan_data)
            
        except Exception as e:
            logger.error(f"Error in LLM plan generation: {e}")
            raise
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for SQL planning."""
        catalog_lines = "\n".join([f"- {f['name']}: {f['desc']}" for f in self.function_catalog])
        rules_lines = "\n".join([f"- {r}" for r in self.rules])
        
        return f"""You are a SQL Planning Expert. Convert business intent into a structured DSL plan for SQLite databases.

Function Catalog (use only these):
{catalog_lines}

Planning Rules:
{rules_lines}

Output JSON Schema:
{{
  "expressions": {{ "<metric_name>": "<DSL_EXPR>", ... }},
  "group_by": [ "<table>.<column>", ... ],
  "time": {{ "col": "<table>.<time_col>", "grain": "day|week|month|quarter|year", "range": "<description>" }},
  "filters": [ {{ "sql": "<condition>" }}, ... ],
  "notes": [ "<planning_note>", ... ],
  "confidence": <0.0-1.0>,
  "metadata": {{ "complexity": "simple|medium|complex" }}
}}

IMPORTANT: 
- The "time" field should be an empty object {{}} if no temporal operations are needed
- NEVER use null or omit the "time" field - always include "time": {{}} for non-temporal queries
- Only populate "time" with col/grain/range when temporal grouping or filtering is actually needed

DSL Expression Examples:
- SUM(table.column) - basic aggregation
- RATIO_OF_SUMS(table.numerator, table.denominator) - for rates
- COUNT_DISTINCT(table.column) - unique counts
- TIME_BUCKET(table.date_col, "month") - time grouping

Always output valid JSON. Never include raw SQL."""
    
    def _create_user_prompt(self, intent: Dict[str, Any], schema_context: str) -> str:
        """Create user prompt with intent and schema context."""
        return f"""
Intent: {json.dumps(intent, indent=2)}

Schema:
{schema_context}

Create a SQL execution plan that:
1. Converts the intent into appropriate DSL expressions
2. Uses only columns from the provided schema
3. Applies appropriate grouping and filtering
4. Handles time-based operations if needed

Return only the JSON plan following the specified schema.
"""
    
    def _create_schema_context(self, schema_info: Dict[str, Any]) -> str:
        """Create schema context for the planner."""
        tables = schema_info.get("tables", {})
        lines = []
        
        for table_name, table_info in tables.items():
            lines.append(f"Table: {table_name}")
            columns = table_info.get("columns", [])
            
            for col in columns:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                flags = []
                
                if col.get("primary_key"):
                    flags.append("PK")
                if col.get("not_null"):
                    flags.append("NN")
                
                flag_str = f" ({', '.join(flags)})" if flags else ""
                lines.append(f"  - {col_name}: {col_type}{flag_str}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _validate_and_enhance_plan(self, plan: SQLPlan, schema_info: Dict[str, Any]) -> SQLPlan:
        """Validate and enhance the generated plan."""
        try:
            # Basic validation
            if not plan.expressions:
                raise ValueError("Plan must contain at least one expression")
            
            # Schema validation
            self._validate_columns_exist(plan, schema_info)
            
            # Enhance metadata
            plan.metadata["planning_method"] = "llm_generated"
            plan.metadata["function_count"] = len(plan.expressions)
            plan.metadata["has_grouping"] = len(plan.group_by) > 0
            plan.metadata["has_time"] = bool(plan.time)
            plan.metadata["has_filters"] = len(plan.filters) > 0
            
            return plan
            
        except Exception as e:
            logger.warning(f"Plan validation failed: {e}")
            return plan  # Return original plan if validation fails
    
    def _validate_columns_exist(self, plan: SQLPlan, schema_info: Dict[str, Any]) -> None:
        """Validate that all referenced columns exist in schema."""
        # Get available columns
        available_columns = set()
        tables = schema_info.get("tables", {})
        
        for table_name, table_info in tables.items():
            for col in table_info.get("columns", []):
                available_columns.add(f"{table_name}.{col['name']}")
        
        # Check group_by columns
        for col in plan.group_by:
            if col not in available_columns:
                logger.warning(f"Group by column not found in schema: {col}")
        
        # Check time column
        if plan.time and "col" in plan.time:
            time_col = plan.time["col"]
            if time_col not in available_columns:
                logger.warning(f"Time column not found in schema: {time_col}")
    
    def _fallback_plan_generation(self, intent_data: Dict[str, Any], schema_info: Dict[str, Any]) -> SQLPlan:
        """Generate fallback plan when LLM planning fails."""
        logger.warning("Using fallback plan generation")
        
        action = intent_data.get("action", "select")
        entity = intent_data.get("entity", "")
        params = intent_data.get("params", {})
        
        # Simple fallback plan
        expressions = {}
        group_by = []
        
        if action == "aggregate":
            column = params.get("column", "*")
            function = params.get("function", "count")
            
            if column != "*" and function:
                expressions["result"] = f"{function.upper()}({entity}.{column})"
            else:
                expressions["result"] = f"COUNT(*)"
            
            if params.get("group_by"):
                group_by.append(f"{entity}.{params['group_by']}")
        
        else:
            expressions["result"] = f"COUNT(*)"
        
        return SQLPlan(
            expressions=expressions,
            group_by=group_by,
            confidence=0.5,
            metadata={"fallback": True, "method": "rule_based"}
        )
    
    def plan_to_context(self, plan: SQLPlan) -> str:
        """Convert SQL plan to context string for SQL generator."""
        context_parts = []
        
        context_parts.append("=== SQL EXECUTION PLAN ===")
        
        # Expressions
        if plan.expressions:
            context_parts.append("Expressions:")
            for name, expr in plan.expressions.items():
                context_parts.append(f"  {name}: {expr}")
        
        # Grouping
        if plan.group_by:
            context_parts.append(f"Group By: {', '.join(plan.group_by)}")
        
        # Time operations
        if plan.time:
            time_info = []
            if "col" in plan.time:
                time_info.append(f"column={plan.time['col']}")
            if "grain" in plan.time:
                time_info.append(f"grain={plan.time['grain']}")
            if "range" in plan.time:
                time_info.append(f"range={plan.time['range']}")
            
            if time_info:
                context_parts.append(f"Time: {', '.join(time_info)}")
        
        # Filters
        if plan.filters:
            context_parts.append("Filters:")
            for f in plan.filters:
                context_parts.append(f"  {f.get('sql', '')}")
        
        # Planning notes
        if plan.notes:
            context_parts.append("Planning Notes:")
            for note in plan.notes:
                context_parts.append(f"  - {note}")
        
        context_parts.append(f"Confidence: {plan.confidence}")
        context_parts.append("=== END PLAN ===")

        return "\n".join(context_parts)

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> SQLPlan:
        return self.create_plan(payload, context)
