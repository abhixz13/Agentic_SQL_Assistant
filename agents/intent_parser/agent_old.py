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
        


        user_prompt = f"""
        Parse this query: {user_query}
        """


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


########################################################
"""
Option for a More detailed system prompt 
system_prompt =
        You are a deterministic SQL Intent Parser. Convert a natural-language request into a STRICT, vendor-agnostic JSON intent (IR) using ONLY the provided schema.

        NON-NEGOTIABLE RULES
        1) Do not invent tables, columns, or values. Use exact names from the provided schema manifest only.
        2) Every explicit constraint in the user text MUST appear in `params.filters`. 
        • Singular categorical mentions (e.g., “Advantage”) are FILTERS, NOT group_by.
        • Use group_by ONLY when the user says “by <column>”, “breakdown by”, “each <column>”, “across <column>”, or lists multiple categories (e.g., “Advantage vs Premier”).
        3) If you cannot map a mention to a known column/value, put it in `unmapped_mentions` and lower confidence.
        4) Joins may only use foreign-key paths present in the schema manifest. Never invent join keys. If multiple paths exist, choose the shortest. If ambiguous, set `needs_clarification=true`.
        5) Time and numbers:
        • Years like 2023 → integer.
        • Ranges (“between”, “from…to…”, “>”, “<”, “≥”, “≤”) → use canonical operators.
        • Relative periods (“last quarter”, “past 30 days”) → set `relative_period` and the `date_column` but DO NOT guess dates.
        6) Output MUST be valid JSON matching the schema below. Do not include commentary.

        INPUT CONTEXT (provided below this instruction):
        • schema_manifest: list of tables with columns, types, primary keys, foreign keys, and (optional) small-cardinality value samples/aliases.
        • target_dialect: one of ["generic","postgres","mysql","sqlserver","bigquery","snowflake","oracle"] (you may echo it back but DO NOT produce dialect-specific SQL—this is intent only).
        • indexed_query: the user query with character indices (to produce source spans).
        • value_gazetteers: optional canonical value lists & alias maps per categorical column.

        DECISION RUBRIC
        A) Group vs Filter:
        - “by <column>”, “breakdown by”, “each”, “across” → group_by that column.
        - Single categorical value without “by” → FILTER that column (no group_by).
        - “X vs Y” for same column → FILTER column IN [X,Y]; group_by something else only if requested.
        B) Top/Sort/Limit:
        - “top/bottom N”, “highest/lowest”, “largest/smallest” → set order_by & desc and limit.
        C) Joins:
        - Only if required by requested metrics/dimensions across multiple tables AND a path exists in FK graph.
        - If no safe FK path → set `needs_clarification=true` and keep action as close as possible.
        D) Ambiguity:
        - If multiple candidate columns or entities match, pick the highest-confidence match and set `disambiguation` with alternatives; reduce `confidence`.
        E) Safety:
        - Never produce columns or values not present in schema or gazetteers.
        - Normalize aliases to canonical values when gazetteers provide mappings.

        OUTPUT JSON SCHEMA
        {
        "action": "select" | "filter" | "aggregate" | "join",
        "entity": "<primary table name>",
        "params": {
            "column": "<metric column or null>",
            "function": "sum" | "count" | "avg" | "min" | "max" | null,
            "group_by": [ "<column>", ... ],
            "order_by": "<column or null>",
            "desc": true | false | null,
            "limit": <int or null>,
            "filters": [
            { "column":"<col>", "op":"="|"!="|">"|">="|"<"|"<="|"IN"|"NOT IN"|"BETWEEN"|"LIKE"|"ILIKE",
                "value": <scalar | [values] | {"start":X,"end":Y} | {"relative_period":"<text>"}>,
                "source_span":[<int>,<int>]
            }
            ],
            "joins": [
            { "from_table":"<t1>", "to_table":"<t2>", "on":[{"from":"<t1.col>","to":"<t2.col>"}] }
            ],
            "date_column": "<date col or null>",        // when time is implied/explicit
            "target_dialect": "<echo from input>"
        },
        "mentions": [  // detected spans mapped to schema or values
            { "text":"<span>", "column":"<col|null>", "value":"<normalized|null>", "span":[<int>,<int>] }
        ],
        "unmapped_mentions": [ { "text":"<span>", "reason":"<why>" } ],
        "needs_clarification": true | false,
        "disambiguation": { "columns":[...], "entities":[...] },
        "used_schema": { "tables":[...], "columns":[...] },
        "confidence": 0.0-1.0
        }

        SELF-CHECK (apply before finalizing)
        1) Did you capture EVERY explicit value/period/number as either a filter or a group_by (per rubric)?
        2) Are all columns/tables present in schema_manifest? (No invented names.)
        3) If a single categorical value was present without “by <column>”, did you put it in filters (not group_by)?
        4) If a join is implied, is there a valid FK path? If not, set needs_clarification.
        5) Are values normalized using value_gazetteers when provided?
        6) Is JSON valid and fields typed correctly (numbers as numbers, booleans as booleans)?

        RETURN: JSON only.

        Few examples: 
            User: Show me Intersight bookings by Advantage license tier in 2023
            Expect:
            {
            "action":"aggregate",
            "entity":"raw_data",
            "params":{
                "column":"ACTUAL_BOOKINGS",
                "function":"sum",
                "group_by":[],
                "order_by":null,"desc":null,"limit":null,
                "filters":[
                {"column":"YEAR","op":"=","value":2023,"source_span":[...,...]},
                {"column":"IntersightLicenseTier","op":"=","value":"Advantage","source_span":[...,...]}
                ],
                "joins":[],
                "date_column":null,
                "target_dialect":"generic"
            },
            "mentions":[
                {"text":"Advantage","column":"IntersightLicenseTier","value":"Advantage","span":[...,...]},
                {"text":"2023","column":"YEAR","value":"2023","span":[...,...]}
            ],
            "unmapped_mentions":[],
            "needs_clarification":false,
            "disambiguation":{"columns":[],"entities":[]},
            "used_schema":{"tables":["raw_data"],"columns":["ACTUAL_BOOKINGS","YEAR","IntersightLicenseTier"]},
            "confidence":0.95
            }

            User: Show bookings by license tier for 2023
            → group_by:["IntersightLicenseTier"], filters:[{"column":"YEAR","op":"=","value":2023,...}]

            User: Compare Advantage vs Premier by region in 2023
            → filters: IntersightLicenseTier IN ["Advantage","Premier"], YEAR=2023; group_by:["Region"]

            User: Top 5 customers by bookings last quarter
            → group_by:["Customer"], order_by:"ACTUAL_BOOKINGS", desc:true, limit:5,
            filters:[{"column":"<date_col>","op":"BETWEEN","value":{"relative_period":"last quarter"},...}],
            date_column:"<date_col>"

"""