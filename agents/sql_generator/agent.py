"""
SQL Generator Agent Module

This module contains the SQLGeneratorAgent that converts structured intents
into executable SQL queries using OpenAI GPT-3.5-turbo with schema-aware
context for accurate and optimized SQL generation.
"""

from openai import OpenAI
from pydantic import BaseModel
import json
from typing import Optional, Dict, Any, List
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .schemas import SQLQuery, SQLGenerationRequest
from utils.token_tracker import record_openai_usage_from_response

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
        - Strict JSON contract
        - Few-shot
        - Retries with tightened prompt if parsing fails
        """
        schema_context = self._create_schema_context(request.schema_info)

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert SQL generator for various SQLite schemas. "
                "Follow these rules strictly: \n"
                "1) Use ONLY table and column names that appear in the provided schema context. \n"
                "2) Do NOT invent, paraphrase, or modify names. Use exact spelling. \n"
                "3) Choose the correct table(s) based on the intent and the schema. If the intent's entity does not exist, infer the best table(s) from the schema; never use names outside the schema. \n"
                "4) Validate the query logically (e.g., GROUP BY when using aggregates with non-aggregated columns). \n"
                "5) SQLite dialect only. \n"
                "6) ALWAYS return STRICT JSON only (no prose, no markdown, no code fences)."
            )
        }

        # Break the prompt into parts
        prompt_parts = [
            f"DATABASE SCHEMA (only use names listed here):\n{schema_context}\n\n",
            f"STRUCTURED INTENT:\n- action: {request.action}\n- entity: {request.entity}\n",
            f"- params: {json.dumps(request.params, ensure_ascii=False)}\n\n",
            """OUTPUT FORMAT (STRICT JSON):
            {
            "sql": "<SQL>",
            "query_type": "SELECT|INSERT|UPDATE|DELETE",
            "table_name": "<primary table>",
            "columns": ["<selected>", "..."],
            "conditions": {},
            "confidence": <float between 0.0 and 1.0 indicating how likely the SQL is correct given the schema>,
            "metadata": {
                "estimated_rows": "low|medium|high",
                "complexity": "simple|medium|complex",
                "notes": ""
            }
            }
            \n\n""",
            """RULES:
            - Use only tables/columns that appear in the schema above.
            - Do not guess names. If a referenced table/column is not in the schema, select alternatives only from the schema that align with the intent and columns mentioned in params.
            - For aggregation: if SELECT mixes aggregates and non-aggregates, include GROUP BY for all non-aggregated columns.
            - For "top N" or ranking: include ORDER BY <metric> DESC and LIMIT N.
            - Prefer numeric columns for SUM/AVG; use COUNT(*) for counts.
            - SQLite dialect only.
            - Reflect the chosen primary table in "table_name". If different from intent.entity due to schema mismatch, place a short note in metadata.notes explaining the choice.
            - Confidence scoring guidance:
              * 0.85–1.00 when table and all columns exactly match the schema and logic is straightforward.
              * 0.60–0.80 when there is minor ambiguity resolved using schema hints but still conforms.
              * 0.10–0.50 when unsure or when making heuristic choices among multiple plausible tables.
              * 0.00 only if you cannot produce a valid SQL for the provided schema.
            - Temporal handling: Never output placeholders like 'last quarter start date'. Always convert natural-language time ranges to concrete SQLite date expressions using date()/strftime(). Prefer half-open ranges [start, next_start) to avoid off-by-one errors.
              * Last month: order_date >= date('now','start of month','-1 month') AND order_date < date('now','start of month')
              * This year: order_date >= date('now','start of year') AND order_date < date('now','start of year','+1 year')
              * Last quarter (previous 3 full months): order_date >= date('now','start of month','-3 months') AND order_date < date('now','start of month')
            \n\n""",
            """FEW-SHOT EXAMPLES:

            Example of invalid table (do NOT do this):
            SELECT * FROM products;  -- if 'products' is not in schema

            Example of valid selection based on schema:
            SELECT region, SUM(total_price) AS total_price FROM product_sales GROUP BY region;

            Example conversion of temporal phrase (last quarter):
            SELECT region, AVG(price_per_unit) AS avg_price
            FROM product_sales
            WHERE order_date >= date('now','start of month','-3 months')
              AND order_date <  date('now','start of month')
            GROUP BY region;

            Example: Top 5 customers by total sales (with ordering and LIMIT):
            SELECT customer_id, SUM(total_price) AS total_sales
            FROM product_sales
            GROUP BY customer_id
            ORDER BY total_sales DESC
            LIMIT 5;

            Example: Aggregation with filter threshold (exclude low values):
            SELECT region, SUM(total_price) AS total_sales
            FROM product_sales
            WHERE total_price > 100
            GROUP BY region;

            Example: Distinct counts (unique customers per region):
            SELECT region, COUNT(DISTINCT customer_id) AS unique_customers
            FROM product_sales
            GROUP BY region;

            Example: Handle nulls when averaging numeric columns:
            SELECT region, AVG(price_per_unit) AS avg_unit_price
            FROM product_sales
            WHERE price_per_unit IS NOT NULL
            GROUP BY region;

            Now produce the STRICT JSON for the INTENT above.
            """
        ]

        user_prompt = "".join(prompt_parts)

        # Lightweight debug for prompt sizes (avoid logging full schema)
        try:
            logger.debug(f"LLM prompt sizes - schema:{len(schema_context)} intent:{len(json.dumps(request.params))} total:{len(user_prompt)}")
        except Exception:
            pass

        def _extract_json(text: str) -> str:
            # Strip code fences if any
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
            # Grab the largest {...} block
            start = text.find("{")
            end = text.rfind("}")
            return text[start:end+1] if start != -1 and end != -1 and end > start else text

        # Try up to 2 attempts with slightly different pressure
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[system_msg, {"role": "user", "content": user_prompt}],
                    temperature=0.0 if attempt == 0 else 0.1,
                    max_tokens=400
                )
                # Record token usage
                record_openai_usage_from_response(resp)
                raw = resp.choices[0].message.content
                data = json.loads(_extract_json(raw))
                # Pydantic will validate shape; if columns/fields missing, raise ValueError
                return SQLQuery(**data)
            except Exception as e:
                logger.warning(f"LLM gen/parse attempt {attempt+1} failed: {e}")

        # If still failing, throw to outer handler so fallback kicks in
        raise ValueError("Unable to parse LLM JSON for SQL generation")

    def _create_schema_context(self, schema_info: dict) -> str:
        """
        Build a compact, LLM-friendly schema context.
        - No assumption that 'foreign_keys' exists.
        - Adds semantic hints (numeric/categorical/temporal).
        - Adds soft join hints by same-named columns across tables.
        """
        # Index columns by name for soft join hints
        col_to_tables: Dict[str, List[str]] = {}
        tables = schema_info.get("tables", {})
        for tname, tinfo in tables.items():
            for col in tinfo.get("columns", []):
                col_to_tables.setdefault(col["name"], []).append(tname)

        lines = []
        for tname, tinfo in tables.items():
            lines.append(f"TABLE {tname}")
            cols = []
            for col in tinfo.get("columns", []):
                dtype = (col.get("type") or "").upper()
                role = []
                if any(x in dtype for x in ["INT", "REAL", "NUMERIC", "DEC"]):
                    role.append("numeric")
                if any(x in dtype for x in ["TEXT", "CHAR", "CLOB", "VARCHAR"]):
                    role.append("categorical")
                if "DATE" in dtype or "TIME" in dtype or "date" in col["name"].lower():
                    role.append("temporal")
                pk = " PK" if col.get("primary_key") else ""
                nn = " NN" if col.get("not_null") else ""
                hint = f" ({','.join(role)})" if role else ""
                cols.append(f"- {col['name']} {dtype}{pk}{nn}{hint}")

                # soft join hint
                others = [t for t in col_to_tables.get(col['name'], []) if t != tname]
                if others:
                    cols.append(f"  ↳ possibly joins with: {', '.join(sorted(set(others)))}")

            lines.extend(cols)

            # Foreign keys if present
            fks = tinfo.get("foreign_keys") or []
            if fks:
                lines.append("FOREIGN KEYS:")
                for fk in fks:
                    lines.append(f"- {fk.get('column')} -> {fk.get('references_table')}.{fk.get('references_column')}")

            # Indexes if present
            idxs = tinfo.get("indexes") or []
            if idxs:
                lines.append("INDEXES:")
                for idx in idxs:
                    uniq = " UNIQUE" if idx.get("unique") else ""
                    lines.append(f"- {idx.get('name')}{uniq}: {', '.join(idx.get('columns', []))}")

            lines.append("")  # spacer

        return "\n".join(lines)

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
            original_sql = sql_query.sql
            original_confidence = float(sql_query.confidence) if sql_query.confidence is not None else 0.0
            any_fixes_applied = False
            # Basic SQL validation
            if not self._validate_sql_syntax(sql_query.sql):
                logger.warning("SQL syntax validation failed, attempting to fix")
                sql_query.sql = self._fix_sql_syntax(sql_query.sql)
                sql_query.confidence *= 0.8  # Reduce confidence for fixed queries
                any_fixes_applied = True
            
            # Schema validation
            if not self._validate_against_schema(sql_query, schema_info):
                logger.warning("Schema validation failed, attempting to fix")
                fixed = self._fix_schema_issues(sql_query, schema_info)
                sql_query.sql = fixed
                sql_query.confidence = max(0.0, float(sql_query.confidence) * 0.7)  # Reduce confidence for schema fixes
                any_fixes_applied = True
            
            # Query optimization
            optimized_sql = self._optimize_query(sql_query.sql)
            if optimized_sql != sql_query.sql:
                sql_query.sql = optimized_sql
                sql_query.metadata["optimized"] = True
 
            # If no fixes were needed and validation passed, ensure confidence isn't unrealistically low
            if not any_fixes_applied and self._validate_against_schema(sql_query, schema_info):
                if original_confidence <= 0.05 or float(sql_query.confidence) <= 0.05:
                    sql_query.confidence = 0.9
                    notes = sql_query.metadata.get("notes", "") if isinstance(sql_query.metadata, dict) else ""
                    add_note = "Confidence raised because SQL fully matched schema with no fixes."
                    sql_query.metadata["notes"] = (notes + ("; " if notes else "") + add_note)

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
        Shallow validation:
        - table exists
        - referenced columns exist (best-effort scan)
        - if GROUP BY is needed, non-aggregated selected columns appear in GROUP BY
        """
        tables = set(schema_info.get("tables", {}).keys())
        if sql_query.table_name not in tables:
            return False

        # Collect set of valid columns for the table
        valid_cols = {c["name"] for c in schema_info["tables"][sql_query.table_name]["columns"]}

        # Quick column usage extraction (best-effort)
        used_cols = set()
        sql_up = sql_query.sql
        # naive find tokens like table.col or bare col
        for col in valid_cols:
            pattern = rf"(?<!\w){re.escape(col)}(?!\w)"
            if re.search(pattern, sql_up, flags=re.I):
                used_cols.add(col)

        # If the model selected columns that don't exist, fail
        for c in sql_query.columns:
            if c != "*" and c not in valid_cols:
                return False

        # GROUP BY sanity: if SELECT has aggregates + non-aggregates, enforce GROUP BY presence
        has_agg = bool(re.search(r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(", sql_up, flags=re.I))
        if has_agg:
            # Detect non-aggregated direct selects (very rough)
            m = re.search(r"select\s+(.*?)\s+from\s", sql_up, flags=re.I | re.S)
            if m:
                select_expr = m.group(1)
                non_aggs = []
                for c in valid_cols:
                    if re.search(rf"\b{re.escape(c)}\b", select_expr, flags=re.I) and not re.search(
                        rf"(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*{re.escape(c)}\s*\)", select_expr, flags=re.I
                    ):
                        non_aggs.append(c)
                if non_aggs and "group by" not in sql_up.lower():
                    return False

        return True

    def _fix_schema_issues(self, sql_query: SQLQuery, schema_info: dict) -> str:
        """
        Heuristic fixes:
        - replace close column names
        - add missing GROUP BY for non-aggregated columns when aggregates present
        """
        from difflib import get_close_matches

        sql = sql_query.sql
        table = sql_query.table_name
        valid_cols = [c["name"] for c in schema_info["tables"].get(table, {}).get("columns", [])]

        # 1) Column name correction (closest match)
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)
        for tok in tokens:
            if tok.lower() in {"select","from","where","group","by","order","limit","asc","desc","as","sum","avg","count","min","max"}:
                continue
            if tok not in valid_cols:
                match = get_close_matches(tok, valid_cols, n=1, cutoff=0.8)
                if match:
                    sql = re.sub(rf"\b{re.escape(tok)}\b", match[0], sql)

        # 2) Add GROUP BY if needed
        has_agg = bool(re.search(r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(", sql, flags=re.I))
        if has_agg and " group by " not in sql.lower():
            m = re.search(r"select\s+(.*?)\s+from\s", sql, flags=re.I | re.S)
            if m:
                select_expr = m.group(1)
                non_aggs = []
                for c in valid_cols:
                    if re.search(rf"\b{re.escape(c)}\b", select_expr, flags=re.I) and not re.search(
                        rf"(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*{re.escape(c)}\s*\)", select_expr, flags=re.I
                    ):
                        non_aggs.append(c)
                if non_aggs:
                    sql = sql.rstrip().rstrip(";")
                    sql += " GROUP BY " + ", ".join(non_aggs)

        return sql
    
    def _optimize_query(self, sql: str) -> str:
        """
        Light, safe optimizations:
        - normalize whitespace
        - ensure ORDER BY for LIMIT patterns when a metric is obvious
        """
        s = re.sub(r"\s+", " ", sql.strip())

        # If LIMIT present but no ORDER BY, add a likely metric desc (common top-N pitfall)
        if re.search(r"\blimit\s+\d+\b", s, flags=re.I) and " order by " not in s.lower():
            metric_candidates = ["total_price", "revenue", "sales", "amount", "total", "quantity"]
            for m in metric_candidates:
                if re.search(rf"\b{re.escape(m)}\b", s, flags=re.I):
                    s = s.rstrip(";") + f" ORDER BY {m} DESC;"
                    break
        if not s.endswith(";"):
            s += ";"
        return s

    def _fallback_sql_generation(self, intent_data: dict, schema_info: dict) -> SQLQuery:
        """Enhanced fallback SQL generation"""
        logger.warning("Using enhanced fallback SQL generation")
        
        action = intent_data.get("action", "select")
        entity = intent_data.get("entity", "product_sales")
        params = intent_data.get("params", {})
        
        # Enhanced fallback logic
        if action == "aggregate" and "group_by" in params:
            column = params.get("column", "*")
            if column != "*":
                sql = f"SELECT {params['group_by']}, SUM({column}) as total FROM {entity} GROUP BY {params['group_by']}"
            else:
                sql = f"SELECT {params['group_by']}, COUNT(*) as count FROM {entity} GROUP BY {params['group_by']}"
        elif action == "aggregate":
            column = params.get("column", "*")
            if column != "*":
                sql = f"SELECT SUM({column}) as total FROM {entity}"
            else:
                sql = f"SELECT COUNT(*) as count FROM {entity}"
        else:
            sql = f"SELECT * FROM {entity} LIMIT 100"
            
        return SQLQuery(
            sql=sql,
            query_type="SELECT",
            table_name=entity,
            columns=["*"],
            conditions=params,
            confidence=0.5,
            metadata={"fallback": True, "notes": "Generated using enhanced fallback method"}
        ) 