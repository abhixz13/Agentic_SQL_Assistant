"""
SQL Generator Agent Module

This module contains the SQLGeneratorAgent that converts structured intents
into executable SQL queries using OpenAI GPT-3.5-turbo with schema-aware
context for accurate and optimized SQL generation.
"""

from openai import OpenAI
from pydantic import BaseModel
import json
from typing import Optional, Dict, Any
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .schemas import SQLQuery, SQLGenerationRequest

from dotenv import load_dotenv
import os

load_dotenv()

class SQLGeneratorAgent:
    """
    Converts structured intents into executable SQL queries.
    
    Uses OpenAI GPT-3.5-turbo with detailed schema context to generate
    accurate, optimized SQL queries for various database operations.
    
    Features:
    - LLM-based SQL generation with schema awareness
    - SQL validation and optimization
    - Support for complex queries (joins, subqueries, aggregations)
    - Error handling and fallback mechanisms
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SQL Generator Agent.
        
        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")  # Checks both sources
        )
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found in .env or arguments")

        self.model = "gpt-3.5-turbo"
        logger.info("SQLGeneratorAgent initialized with OpenAI GPT-3.5-turbo")
    
    def generate_sql(self, intent_data: dict, schema_info: dict) -> SQLQuery:
        """
        Generate SQL query from structured intent and schema information.
        
        Args:
            intent_data (dict): Structured intent from IntentParserAgent
            schema_info (dict): Database schema information
            
        Returns:
            SQLQuery: Structured SQL query with metadata
        """
        try:
            # Create generation request
            request = SQLGenerationRequest(
                action=intent_data["action"],
                entity=intent_data["entity"],
                params=intent_data["params"],
                schema_info=schema_info
            )
            
            logger.info(f"Generating SQL for action: {request.action}, entity: {request.entity}")
            
            # Generate SQL using LLM
            sql_result = self._generate_with_llm(request)
            
            # Validate and optimize the generated SQL
            validated_sql = self._validate_and_optimize_sql(sql_result, schema_info)
            
            return validated_sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return self._fallback_sql_generation(intent_data, schema_info)
    
    def _generate_with_llm(self, request: SQLGenerationRequest) -> SQLQuery:
        """
        Generate SQL using OpenAI LLM with schema context.
        
        Args:
            request: SQLGenerationRequest with all necessary information
            
        Returns:
            SQLQuery: Generated SQL query with metadata
        """
        # Create schema context for the LLM
        schema_context = self._create_schema_context(request.schema_info)
        
        # Build the SQL generation prompt
        generation_prompt = f"""You are an expert SQL query generator. Your job is to convert structured intents into accurate, optimized SQL queries.

{schema_context}

USER INTENT:
- Action: {request.action}
- Entity: {request.entity}
- Parameters: {request.params}

INSTRUCTIONS:
1. Analyze the intent and generate appropriate SQL
2. Use only tables and columns that exist in the schema
3. Apply proper SQL syntax and best practices
4. Optimize the query for performance where possible
5. Return a JSON object with this structure:
   {{
     "sql": "the actual SQL query",
     "query_type": "SELECT|INSERT|UPDATE|DELETE",
     "table_name": "primary table name",
     "columns": ["list", "of", "columns"],
     "conditions": {{"column": "value"}},
     "confidence": 0.0-1.0,
     "metadata": {{
       "estimated_rows": "low|medium|high",
       "complexity": "simple|medium|complex",
       "notes": "any additional notes"
     }}
   }}

EXAMPLES:

Simple Filter:
Input: {{"action": "filter", "entity": "bookings", "params": {{"city": "New York"}}}}
Output: {{
  "sql": "SELECT * FROM bookings WHERE city = 'New York'",
  "query_type": "SELECT",
  "table_name": "bookings",
  "columns": ["*"],
  "conditions": {{"city": "New York"}},
  "confidence": 0.95,
  "metadata": {{"estimated_rows": "medium", "complexity": "simple", "notes": "Simple filter query"}}
}}

Complex Filter:
Input: {{"action": "filter", "entity": "bookings", "params": {{"city": "NYC", "amount": "> 1000", "date": "2024-01-01"}}}}
Output: {{
  "sql": "SELECT * FROM bookings WHERE city = 'NYC' AND amount > 1000 AND date >= '2024-01-01'",
  "query_type": "SELECT",
  "table_name": "bookings",
  "columns": ["*"],
  "conditions": {{"city": "NYC", "amount": "> 1000", "date": "2024-01-01"}},
  "confidence": 0.9,
  "metadata": {{"estimated_rows": "low", "complexity": "medium", "notes": "Multiple condition filter"}}
}}

Aggregation:
Input: {{"action": "aggregate", "entity": "bookings", "params": {{"group_by": "city", "function": "sum", "column": "amount"}}}}
Output: {{
  "sql": "SELECT city, SUM(amount) as total_amount FROM bookings GROUP BY city",
  "query_type": "SELECT",
  "table_name": "bookings",
  "columns": ["city", "amount"],
  "conditions": {{}},
  "confidence": 0.95,
  "metadata": {{"estimated_rows": "low", "complexity": "medium", "notes": "Aggregation with grouping"}}
}}

Generate SQL for the user intent above:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL query generator."},
                    {"role": "user", "content": generation_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_tokens=300
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            sql_data = json.loads(content)
            
            logger.info(f"Successfully generated SQL: {sql_data['sql']}")
            return SQLQuery(**sql_data)
            
        except Exception as e:
            logger.error(f"Error in LLM SQL generation: {e}")
            raise e
    
    def _create_schema_context(self, schema_info: dict) -> str:
        """
        Create detailed schema context for the LLM prompt.
        
        Args:
            schema_info: Database schema information
            
        Returns:
            str: Formatted schema context
        """
        context = "DATABASE SCHEMA:\n"
        context += "=" * 50 + "\n"
        
        for table_name, table_info in schema_info["tables"].items():
            context += f"\nTABLE: {table_name}\n"
            context += "-" * 30 + "\n"
            
            for col in table_info["columns"]:
                pk_marker = " (PRIMARY KEY)" if col["primary_key"] else ""
                null_marker = " NOT NULL" if col["not_null"] else ""
                context += f"  - {col['name']}: {col['type']}{pk_marker}{null_marker}\n"
            
            if table_info["foreign_keys"]:
                context += "\n  FOREIGN KEYS:\n"
                for fk in table_info["foreign_keys"]:
                    context += f"    - {fk['column']} -> {fk['references_table']}.{fk['references_column']}\n"
        
        return context
    
    def _validate_and_optimize_sql(self, sql_query: SQLQuery, schema_info: dict) -> SQLQuery:
        """
        Validate and optimize the generated SQL query.
        
        Args:
            sql_query: Generated SQL query
            schema_info: Database schema for validation
            
        Returns:
            SQLQuery: Validated and optimized query
        """
        try:
            # Basic SQL validation
            if not self._validate_sql_syntax(sql_query.sql):
                logger.warning("SQL syntax validation failed, attempting to fix")
                sql_query.sql = self._fix_sql_syntax(sql_query.sql)
                sql_query.confidence *= 0.8  # Reduce confidence for fixed queries
            
            # Schema validation
            if not self._validate_against_schema(sql_query, schema_info):
                logger.warning("Schema validation failed, attempting to fix")
                sql_query.sql = self._fix_schema_issues(sql_query, schema_info)
                sql_query.confidence *= 0.7  # Reduce confidence for schema fixes
            
            # Query optimization
            optimized_sql = self._optimize_query(sql_query.sql)
            if optimized_sql != sql_query.sql:
                sql_query.sql = optimized_sql
                sql_query.metadata["optimized"] = True
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error in SQL validation/optimization: {e}")
            return sql_query  # Return original if validation fails
    
    def _validate_sql_syntax(self, sql: str) -> bool:
        """
        Basic SQL syntax validation.
        
        Args:
            sql: SQL query string
            
        Returns:
            bool: True if syntax appears valid
        """
        # Basic checks
        required_keywords = ["SELECT", "FROM"]
        sql_upper = sql.upper()
        
        for keyword in required_keywords:
            if keyword not in sql_upper:
                return False
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False
        
        return True
    
    def _fix_sql_syntax(self, sql: str) -> str:
        """
        Attempt to fix basic SQL syntax issues.
        
        Args:
            sql: SQL query string
            
        Returns:
            str: Fixed SQL query
        """
        # Basic fixes
        sql = sql.strip()
        
        # Ensure proper spacing
        sql = re.sub(r'\s+', ' ', sql)
        
        # Fix common issues
        sql = sql.replace('SELECT*', 'SELECT *')
        sql = sql.replace('FROM*', 'FROM *')
        
        return sql
    
    def _validate_against_schema(self, sql_query: SQLQuery, schema_info: dict) -> bool:
        """
        Validate that the SQL query references valid tables and columns.
        
        Args:
            sql_query: Generated SQL query
            schema_info: Database schema
            
        Returns:
            bool: True if schema validation passes
        """
        try:
            # Check if table exists
            if sql_query.table_name not in schema_info["tables"]:
                logger.warning(f"Table '{sql_query.table_name}' not found in schema")
                return False
            
            # Check if columns exist (basic check)
            table_info = schema_info["tables"][sql_query.table_name]
            available_columns = [col["name"] for col in table_info["columns"]]
            
            for column in sql_query.columns:
                if column != "*" and column not in available_columns:
                    logger.warning(f"Column '{column}' not found in table '{sql_query.table_name}'")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in schema validation: {e}")
            return False
    
    def _fix_schema_issues(self, sql_query: SQLQuery, schema_info: dict) -> str:
        """
        Attempt to fix schema-related issues in the SQL query.
        
        Args:
            sql_query: Generated SQL query
            schema_info: Database schema
            
        Returns:
            str: Fixed SQL query
        """
        # For now, return the original SQL
        # In a full implementation, this would fix column names, table names, etc.
        return sql_query.sql
    
    def _optimize_query(self, sql: str) -> str:
        """
        Apply basic query optimizations.
        
        Args:
            sql: SQL query string
            
        Returns:
            str: Optimized SQL query
        """
        # Basic optimizations
        sql = sql.strip()
        
        # Remove unnecessary whitespace
        sql = re.sub(r'\s+', ' ', sql)
        
        # Ensure proper formatting
        sql = sql.replace('SELECT *', 'SELECT *')
        sql = sql.replace('FROM *', 'FROM *')
        
        return sql
    
    def _fallback_sql_generation(self, intent_data: dict, schema_info: dict) -> SQLQuery:
        """
        Fallback SQL generation when LLM fails.
        
        Args:
            intent_data: Structured intent
            schema_info: Database schema
            
        Returns:
            SQLQuery: Basic fallback SQL query
        """
        logger.warning("Using fallback SQL generation")
        
        action = intent_data.get("action", "select")
        entity = intent_data.get("entity", "unknown")
        params = intent_data.get("params", {})
        
        # Generate basic SQL based on action
        if action == "filter":
            conditions = []
            for key, value in params.items():
                if isinstance(value, str):
                    conditions.append(f"{key} = '{value}'")
                else:
                    conditions.append(f"{key} = {value}")
            
            where_clause = " AND ".join(conditions) if conditions else ""
            sql = f"SELECT * FROM {entity}"
            if where_clause:
                sql += f" WHERE {where_clause}"
        
        elif action == "aggregate":
            sql = f"SELECT * FROM {entity}"
            if "group_by" in params:
                sql += f" GROUP BY {params['group_by']}"
        
        else:
            sql = f"SELECT * FROM {entity}"
        
        return SQLQuery(
            sql=sql,
            query_type="SELECT",
            table_name=entity,
            columns=["*"],
            conditions=params,
            confidence=0.3,  # Low confidence for fallback
            metadata={"fallback": True, "notes": "Generated using fallback method"}
        ) 