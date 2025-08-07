"""
Intent Parser Agent Module

This module contains the IntentParserAgent that converts natural language
queries into structured SQL intents using OpenAI GPT-3.5-turbo with
detailed schema context for accurate parsing.
"""

from openai import OpenAI
from pydantic import BaseModel
import json
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntentParserAgent:
    """
    Parses natural language queries into structured SQL intents.
    
    Uses OpenAI GPT-3.5-turbo with detailed schema context to accurately
    interpret user queries and generate structured intents for SQL operations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the IntentParserAgent.
        
        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
        logger.info("IntentParserAgent initialized with OpenAI GPT-3.5-turbo")
        
    def create_schema_prompt(self, schema_info: dict) -> str:
        """
        Create detailed schema context for the LLM prompt.
        
        Args:
            schema_info: Detailed schema information from extract_schema()
            
        Returns:
            str: Formatted schema context for LLM prompt
        """
        schema_prompt = "DATABASE SCHEMA:\n"
        schema_prompt += "=" * 50 + "\n"
        
        for table_name, table_info in schema_info["tables"].items():
            schema_prompt += f"\nTABLE: {table_name}\n"
            schema_prompt += "-" * 30 + "\n"
            
            for col in table_info["columns"]:
                pk_marker = " (PRIMARY KEY)" if col["primary_key"] else ""
                null_marker = " NOT NULL" if col["not_null"] else ""
                schema_prompt += f"  - {col['name']}: {col['type']}{pk_marker}{null_marker}\n"
            
            if table_info["foreign_keys"]:
                schema_prompt += "\n  FOREIGN KEYS:\n"
                for fk in table_info["foreign_keys"]:
                    schema_prompt += f"    - {fk['column']} -> {fk['references_table']}.{fk['references_column']}\n"
        
        if schema_info["relationships"]:
            schema_prompt += "\nRELATIONSHIPS:\n"
            schema_prompt += "-" * 30 + "\n"
            for rel in schema_info["relationships"]:
                schema_prompt += f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}\n"
        
        return schema_prompt
    
    def parse(self, user_query: str, schema_info: dict) -> dict:
        """
        Parse natural language query into structured SQL intent.
        
        Args:
            user_query: Natural language query (e.g., "Show bookings from New York")
            schema_info: Database schema information
            
        Returns:
            dict: Structured intent with action, entity, and parameters
        """
        schema_prompt = self.create_schema_prompt(schema_info)
        
        system_prompt = f"""You are an expert SQL query parser. Your job is to convert natural language queries into structured SQL intents.

{schema_prompt}

INSTRUCTIONS:
1. Analyze the user query against the provided schema
2. Identify the intended SQL operation (filter, aggregate, join, etc.)
3. Extract the target table/entity
4. Parse any parameters (filters, grouping, etc.)
5. Return a JSON object with the following structure:
   {{
     "action": "filter|aggregate|join|select",
     "entity": "table_name",
     "params": {{"key": "value"}},
     "confidence": 0.0-1.0
   }}

EXAMPLES:
Query: "Show all bookings"
Output: {{"action": "select", "entity": "bookings", "params": {{}}, "confidence": 0.9}}

Query: "Count bookings by city"
Output: {{"action": "aggregate", "entity": "bookings", "params": {{"group_by": "city", "function": "count"}}, "confidence": 0.95}}

Query: "Show bookings from New York"
Output: {{"action": "filter", "entity": "bookings", "params": {{"city": "New York"}}, "confidence": 0.9}}

IMPORTANT: Only reference tables and columns that exist in the schema. If the query references non-existent entities, use the closest match or return confidence < 0.5."""

        try:
            logger.info(f"Parsing query: {user_query}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this query: {user_query}"}
                ],
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=200
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            intent_data = json.loads(content)
            
            logger.info(f"Successfully parsed intent: {intent_data}")
            return intent_data
            
        except Exception as e:
            # Fallback to basic parsing if LLM fails
            logger.error(f"Error parsing intent: {e}")
            return {
                "action": "select",
                "entity": "unknown",
                "params": {},
                "confidence": 0.1
            } 