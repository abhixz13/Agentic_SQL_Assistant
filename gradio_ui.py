import gradio as gr
import sqlite3
import pandas as pd
from pathlib import Path
import sys
import os
import json, re
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.workflow import SQLWorkflow
from agents.query_executor.agent import QueryExecutorAgent
from agents.sql_generator.agent import SQLGeneratorAgent
from agents.visualization.agent import VisualizationAgent, VisualizationOptions
from utils.token_tracker import token_tracker, record_openai_usage_from_response
from agents.schema_loader.semantic_schema import SemanticSchemaManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradio.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_workflow():
    """Initialize the AI workflow with the correct database"""
    workflow = SQLWorkflow("data/product_sales.db")  # Use the correct database
    sql_generator = SQLGeneratorAgent()
    viz_agent = VisualizationAgent()
    return workflow, sql_generator, viz_agent

def parse_natural_language_to_intent(natural_query: str, schema: dict = None) -> dict:
    """LLM-driven intent parsing that works with any dataset schema"""
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  No OpenAI API key found. Using fallback intent parsing.")
            return create_fallback_intent(natural_query, schema)
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Get schema if not provided
        if schema is None:
            schema = get_formatted_schema()
        
        # Create schema context for the LLM
        schema_context = create_schema_context(schema)
        
        # LLM prompt for intent parsing
        prompt = f"""You are an Intent Parser that converts a natural-language question into a STRICT JSON intent for SQL generation.

            ## Database Schema (human-readable)
            {schema_context}

            ## Output format (STRICT JSON, no markdown, no prose)
            {{
            "action": "select|aggregate|filter|top_n",
            "entity": "<table_name>",
            "params": {{
                "column": "<single column or *>",
                "function": "sum|count|avg|max|min|null",
                "group_by": "<column or null>",
                "order_by": "<column or null>",
                "desc": true|false|null,
                "limit": <integer or null>,
                "filters": [{{"column":"<col>","op":"="|">"|"<"|">="|"<="|"LIKE","value":"<val>"}}]
            }}
            }}

            ## Rules (follow exactly)
            - Use only columns and tables that appear in the schema above.
            - If the user says “by X” or “per X”, set params.group_by = "X".
            - For revenue/sales/amount/price, prefer numeric columns containing ["total","revenue","sales","amount","price"].
            - For “top N …”, set action="top_n", set params.limit=N, and include params.order_by=<metric> and params.desc=true.
            - For aggregates, params.column MUST be a single string (not a list). Use "*" only for COUNT(*).
            - If unsure, choose the most plausible table from the schema and set params.limit=20.

            ## Few-shot examples
            Q: "Show me revenue by region"
            A:
            {{"action":"aggregate","entity":"product_sales","params":{{"column":"total_price","function":"sum","group_by":"region","order_by":null,"desc":null,"limit":null,"filters":[]}}}}

            Q: "Top 5 customers by sales"
            A:
            {{"action":"aggregate","entity":"product_sales","params":{{"column":"total_price","function":"sum","group_by":"customer_id","order_by":"total_price","desc":true,"limit":5,"filters":[]}}}}

            Q: "How many orders per category?"
            A:
            {{"action":"aggregate","entity":"product_sales","params":{{"column":"*","function":"count","group_by":"category","order_by":null,"desc":null,"limit":null,"filters":[]}}}}

            ## User question
            "{natural_query}"

            ## Return the STRICT JSON only:
            """

        # Call LLM for intent parsing
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a careful SQL intent parser. "
                    "Always return STRICT JSON that matches the required schema. "
                    "Do not include markdown, code fences, or explanations."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        # Record token usage
        record_openai_usage_from_response(response)
        
        # Parse the response
        intent_text = response.choices[0].message.content.strip()
        
        # ---- JSON extraction ----
        def extract_json(s: str) -> str:
            s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
            start, end = s.find("{"), s.rfind("}")
            return s[start:end+1] if start != -1 and end != -1 else s

        try:
            intent = json.loads(extract_json(intent_text))
            if not isinstance(intent, dict) or "action" not in intent or "entity" not in intent:
                raise ValueError("Invalid intent structure")
            print(f"🤖 LLM Intent Parsing: {natural_query} → {intent}")
            return intent

        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ LLM Intent Parsing Error: {e}")
            print(f"Raw response: {intent_text}")
            return create_fallback_intent(natural_query, schema)

    except Exception as e:
        print(f"❌ Intent Parsing Error: {e}")
        return create_fallback_intent(natural_query, schema)

# Creating the schema context for the LLM
"""
The schema context is a string that describes the database schema in a way that is easy for the LLM to understand.
It is used to help the LLM understand the database schema and generate SQL queries.

1. No FK dependence: join suggestions come from shared column names and the common *_id convention.
2. Engine-agnostic: only uses the schema dict (table → columns with name/type), so it works with SQLite/MySQL/Postgres as long as your extractor fills that shape.
3. Semantic cues: numeric/temporal/categorical hints push the model toward sane SUM/AVG, GROUP BY, and time filters.
4. Token control: truncates overly large schemas to avoid blowing prompt size.
"""
def create_schema_context(schema: dict, max_tables: int = 30, max_cols_per_table: int = 30) -> str:
    """
    Turn the extracted schema dict into an LLM-friendly description that:
      - works even when foreign keys are missing,
      - infers simple semantic roles from names/types (numeric / categorical / temporal / boolean),
      - suggests soft join hints based on shared column names and *_id conventions,
      - truncates very large schemas to keep prompts small.

    The function is agnostic to DB engine; it only relies on the `schema` dict shape produced by get_formatted_schema().
    """
    # Defensive defaults
    tables = schema.get("tables", {}) or {}
    if not isinstance(tables, dict):
        return "No schema available."

    # Build global index of column name -> [tables...]
    all_columns = {}
    for tname, tinfo in tables.items():
        for col in (tinfo.get("columns") or []):
            cname = str(col.get("name", "")).strip()
            if not cname:
                continue
            all_columns.setdefault(cname, set()).add(tname)

    def role_hint(col_name: str, col_type: str) -> str:
        """Heuristic semantic role from name/type only (engine-agnostic)."""
        name = (col_name or "").lower()
        ctype = (col_type or "").upper()

        # temporal first (either name or type suggests date/time)
        if "date" in name or "time" in name or ctype in {"DATE", "DATETIME", "TIMESTAMP"}:
            return "temporal (date filters/trends)"

        # boolean-ish
        if name.startswith(("is_", "has_")) or ctype in {"BOOLEAN", "BOOL"}:
            return "boolean (true/false filtering)"

        # numeric metrics (by type or name cues)
        metric_keywords = ("price", "amount", "revenue", "sales", "qty", "quantity", "total", "cost")
        if ctype in {"INTEGER", "INT", "BIGINT", "SMALLINT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"}:
            if any(k in name for k in metric_keywords):
                return "numeric_metric (use in SUM/AVG)"
            return "numeric (aggregations allowed)"

        # text / categorical
        if ctype in {"TEXT", "CHAR", "NCHAR", "VARCHAR", "NVARCHAR", "STRING"} or ctype == "" or ctype is None:
            return "categorical (GROUP BY / filtering)"

        # default
        return "other"

    def soft_join_targets(current_table: str, col_name: str) -> list:
        """
        Guess join opportunities:
          - same column name exists in other tables,
          - *_id appears across multiple tables,
          - id columns across tables (very common in dumps).
        """
        peers = set(all_columns.get(col_name, set()))
        if current_table in peers:
            peers.remove(current_table)

        hints = set(peers)

        # *_id convention: suggest tables sharing the same *_id
        if col_name.lower().endswith("_id"):
            for t, tinfo in tables.items():
                if t == current_table:
                    continue
                for c in (tinfo.get("columns") or []):
                    if str(c.get("name", "")).lower() == col_name.lower():
                        hints.add(t)

        # generic "id" columns (common but noisy) – only hint if many tables share it
        if col_name.lower() in {"id", "pk", "key"} and len(all_columns.get(col_name, [])) > 1:
            hints.update(all_columns[col_name])
            hints.discard(current_table)

        # limit how many we print to keep prompt short
        return sorted(list(hints))[:3]

    # Compose context string with truncation
    context_lines = []
    context_lines.append("📚 Available Tables and Columns (semantic hints & soft join suggestions):")

    shown_tables = 0
    for tname, tinfo in list(tables.items())[:max_tables]:
        context_lines.append(f"\n📊 Table: {tname}")
        context_lines.append("Columns:")

        cols = (tinfo.get("columns") or [])[:max_cols_per_table]
        for col in cols:
            cname = str(col.get("name", ""))
            ctype = str(col.get("type", "") or "")
            # base line
            line = f"  - {cname} ({ctype})"

            # role
            r = role_hint(cname, ctype)
            if r:
                line += f" → {r}"

            # primary / not null flags if present
            if col.get("primary_key"):
                line += " [PRIMARY KEY]"
            if col.get("not_null"):
                line += " [NOT NULL]"

            # soft join hints
            joins = soft_join_targets(tname, cname)
            if joins:
                line += f" → possibly joins with: {', '.join(joins)}"

            context_lines.append(line)

        # indexes (if available)
        idxs = tinfo.get("indexes") or []
        if idxs:
            context_lines.append("Indexes:")
            for idx in idxs[:8]:  # guardrail
                iname = idx.get("name", "")
                icols = ", ".join(idx.get("columns", []) or [])
                uniq = " (UNIQUE)" if idx.get("unique") else ""
                context_lines.append(f"  - {iname}: {icols}{uniq}")

        shown_tables += 1

    # Truncation warnings
    total_tables = len(tables)
    if shown_tables < total_tables:
        context_lines.append(f"\n… ({total_tables - shown_tables} more tables omitted for brevity)")

    return "\n".join(context_lines)


def create_fallback_intent(natural_query: str, schema: dict) -> dict:
    """Create an intelligent fallback intent when LLM parsing fails"""
    query_lower = natural_query.lower()
    
    # Get the first table from schema
    tables = list(schema.get("tables", {}).keys())
    if not tables:
        return {"action": "select", "entity": "unknown", "params": {"limit": 10}}
    
    table_name = tables[0]
    table_info = schema.get("tables", {}).get(table_name, {})
    columns = table_info.get("columns", [])
    
    # Find numeric columns for aggregation
    numeric_columns = [col["name"] for col in columns if col["type"] in ["INTEGER", "REAL"]]
    text_columns = [col["name"] for col in columns if col["type"] == "TEXT"]
    
    # Enhanced pattern matching with schema awareness
    if any(word in query_lower for word in ["count", "how many", "number"]):
        # Look for grouping by text columns
        for col in text_columns:
            if col in query_lower:
                return {
                    "action": "aggregate",
                    "entity": table_name,
                    "params": {"function": "count", "column": "*", "group_by": col}
                }
        return {
            "action": "aggregate",
            "entity": table_name,
            "params": {"function": "count", "column": "*"}
        }
    
    elif any(word in query_lower for word in ["sum", "total", "revenue", "sales", "amount"]):
        # Look for numeric columns to sum
        # Prioritize columns that match revenue/sales concepts
        revenue_columns = [col for col in numeric_columns if any(word in col.lower() for word in ["total", "revenue", "sales", "amount"])]
        price_columns = [col for col in numeric_columns if "price" in col.lower()]
        
        # Use the best matching column
        target_column = None
        if revenue_columns:
            target_column = revenue_columns[0]  # Prefer total_price, revenue, etc.
        elif price_columns:
            target_column = price_columns[0]    # Fallback to price columns
        elif numeric_columns:
            target_column = numeric_columns[0]  # Last resort
            
        if target_column:
            # Check for grouping
            for group_col in text_columns:
                if group_col in query_lower:
                    return {
                        "action": "aggregate",
                        "entity": table_name,
                        "params": {"function": "sum", "column": target_column, "group_by": group_col}
                    }
            return {
                "action": "aggregate",
                "entity": table_name,
                "params": {"function": "sum", "column": target_column}
            }
    
    elif any(word in query_lower for word in ["average", "avg", "mean"]):
        # Look for numeric columns to average
        for col in numeric_columns:
            if col in query_lower:
                # Check for grouping
                for group_col in text_columns:
                    if group_col in query_lower:
                        return {
                            "action": "aggregate",
                            "entity": table_name,
                            "params": {"function": "avg", "column": col, "group_by": group_col}
                        }
                return {
                    "action": "aggregate",
                    "entity": table_name,
                    "params": {"function": "avg", "column": col}
                }
    
    elif any(word in query_lower for word in ["top", "highest", "maximum"]):
        # Look for numeric columns to order by
        for col in numeric_columns:
            if col in query_lower:
                return {
                    "action": "top_n",
                    "entity": table_name,
                    "params": {"limit": 5, "order_by": col, "desc": True}
                }
    
    elif any(word in query_lower for word in ["by", "group", "per"]):
        # Look for grouping patterns
        for col in text_columns:
            if col in query_lower:
                # Find a numeric column to aggregate
                if numeric_columns:
                    return {
                        "action": "aggregate",
                        "entity": table_name,
                        "params": {"function": "sum", "column": numeric_columns[0], "group_by": col}
                    }
    
    # Default fallback
    return {
        "action": "select",
        "entity": table_name,
        "params": {"limit": 20}
    }

def get_formatted_schema():
    """Fetch complete schema with semantic enhancement if available."""
    try:
        # First try to load semantic schema from correct path
        semantic_manager = SemanticSchemaManager("agents/schema_loader/config/semantic_schemas")
        semantic_schema = semantic_manager.load_schema("product_sales_semantic.json")
        
        # Get standard schema structure
        standard_schema = _get_standard_schema()
        
        if semantic_schema:
            logger.info("✅ Using enhanced semantic schema for Planner → Validator → Generator")
            # Enhanced schema with full semantic context
            return {
                **standard_schema,
                "semantic_context": semantic_manager.get_enhanced_schema_context(semantic_schema),
                "semantic_schema": semantic_schema,  # Full semantic schema object for agents
                "has_semantic": True
            }
        else:
            logger.info("📊 Using standard schema (no semantic enhancement)")
            return {**standard_schema, "has_semantic": False, "semantic_schema": None}
            
    except Exception as e:
        print(f"Schema Error: {str(e)}")
        return {"tables": {}, "error": str(e), "has_semantic": False, "semantic_schema": None}

def _get_standard_schema():
    """Fetch basic schema metadata from the database."""
    try:
        with sqlite3.connect("data/product_sales.db") as conn:
            cursor = conn.cursor()
            schema = {"tables": {}}
            
            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall() if table[0] != "sqlite_sequence"]
            
            for table_name in tables:
                # Get columns
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = []
                for col in cursor.fetchall():
                    columns.append({
                        "name": col[1],  # column name
                        "type": col[2],   # data type
                        "not_null": bool(col[3]),
                        "default": col[4],
                        "primary_key": bool(col[5])
                    })
                
                # Get indexes
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = []
                for idx in cursor.fetchall():
                    index_name = idx[1]
                    cursor.execute(f"PRAGMA index_info({index_name})")
                    index_cols = [col[2] for col in cursor.fetchall()]  # column names
                    indexes.append({
                        "name": index_name,
                        "columns": index_cols,
                        "unique": bool(idx[2])
                    })
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = []
                for fk in cursor.fetchall():
                    foreign_keys.append({
                        "column": fk[3],  # local column
                        "references_table": fk[2],  # foreign table
                        "references_column": fk[4]  # foreign column
                    })
                
                schema["tables"][table_name] = {
                    "columns": columns,
                    "indexes": indexes,
                    "foreign_keys": foreign_keys
                }
            
            return schema
        
    except Exception as e:
        print(f"Schema Error: {str(e)}")
        return {"tables": {}, "error": str(e)}

def process_natural_language_query(natural_query, chart_type):
    """Convert natural language to SQL and execute with visualization"""
    logger.debug(f"Processing query: {natural_query}")
    try:
        # Reset token tracker at the start of one end-to-end request
        token_tracker.reset()
        # Initialize AI agents
        workflow, sql_generator, viz_agent = setup_workflow()
        
        # Get database schema for context
        schema = get_formatted_schema()
        schema_context = create_schema_context(schema)
        
        # Parse natural language to structured intent using LLM
        intent = parse_natural_language_to_intent(natural_query, schema)
        
        # Generate SQL using LLM-based generator
        sql_result = sql_generator.generate_sql(intent, schema)
        sql_query = sql_result.sql
        
        # Execute the generated SQL using workflow
        # Create a JSON-serializable version of schema for workflow
        schema_for_workflow = {k: v for k, v in schema.items() if k != 'semantic_schema'}
        result = workflow.run(sql_query, schema_context=json.dumps(schema_for_workflow))
        result_df = pd.DataFrame(result.data)
        
        # Build explanation
        # Collect model metadata if available
        confidence = getattr(sql_result, "confidence", None)
        metadata = getattr(sql_result, "metadata", {}) or {}
        complexity = metadata.get("complexity")
        estimated_rows = metadata.get("estimated_rows")
        flags = [k for k in ("optimized", "fallback") if metadata.get(k)]
        # Token usage totals
        usage = token_tracker.get_totals()

        # Attempt to get SQLite query plan
        plan_lines = []
        try:
            with sqlite3.connect("data/product_sales.db") as plan_conn:
                # Prefer the exact executed SQL, if provided by the executor
                executed_sql = getattr(result, "sql", sql_query)
                cur = plan_conn.execute(f"EXPLAIN QUERY PLAN {executed_sql.rstrip(';')}")
                rows = cur.fetchall()
                for r in rows:
                    try:
                        plan_lines.append(str(r[-1]))
                    except Exception:
                        plan_lines.append(str(r))
        except Exception as plan_err:
            plan_lines.append(f"Could not retrieve query plan: {plan_err}")

        # Compose richer explanation
        explanation_parts = []
        explanation_parts.append("**Intent**:")
        explanation_parts.append(f"- action: {intent.get('action')}\n- entity: {intent.get('entity')}\n- params: {json.dumps(intent.get('params', {}))}")

        explanation_parts.append("\n**Generated SQL**:\n```sql\n" + sql_query + "\n```")
        # If the executed SQL differs (e.g., repaired by reasoning), show it
        executed_sql_differs = getattr(result, "sql", sql_query) != sql_query
        if executed_sql_differs:
            explanation_parts.append("\n**Executed SQL (corrected)**:\n```sql\n" + getattr(result, "sql", sql_query) + "\n```")
            explanation_parts.append("\n**Correction Details**:")
            explanation_parts.append(f"- Initial error: Incorrect table/column reference")
            explanation_parts.append(f"- Automatic correction: System detected and fixed schema mismatch")

        model_bits = []
        if confidence is not None:
            model_bits.append(f"confidence: {confidence:.2f}")
            if confidence < 0.5 and executed_sql_differs:  # Only show note if correction actually happened
                model_bits.append("\n**Note**: Low confidence indicates the initial query required correction. The system automatically fixed the issues below:")
                if "product_name" in sql_query and "product_sales" in getattr(result, "sql", ""):
                    model_bits.append("- Fixed table name from 'product_name' to 'product_sales'")
        if complexity:
            model_bits.append(f"complexity: {complexity}")
        if estimated_rows:
            model_bits.append(f"estimated_rows: {estimated_rows}")
            # include notes from model metadata if present
            notes = metadata.get("notes") if isinstance(metadata, dict) else None
            if notes:
                model_bits.append(f"notes: {notes}")
        if flags:
            model_bits.append("flags: " + ", ".join(flags))
        if usage and (usage.get("total_tokens") or usage.get("prompt_tokens") or usage.get("completion_tokens")):
            model_bits.append(
                f"tokens: total={usage.get('total_tokens', 0)}, prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}"
            )
        if model_bits:
            explanation_parts.append("\n**Model assessment**:\n- " + "\n- ".join(model_bits))

        explanation_parts.append(
            "\n**Execution**:" +
            f"\n- rows: {len(result.data)}" +
            (f"\n- time: {getattr(result, 'execution_time', None):.3f}s" if getattr(result, 'execution_time', None) is not None else "")
        )

        if plan_lines:
            explanation_parts.append("\n**Query plan**:\n- " + "\n- ".join(plan_lines))

        explanation = "\n".join(explanation_parts)
        
        # Create visualization if requested
        if chart_type != "none":
            # Infer axes from actual result
            cols = list(result_df.columns)
            numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()

            # Prefer common categorical dimensions for x
            preferred_x_order = ["region", "category", "customer_id", "order_date", "product_name"]
            x_axis = next((c for c in preferred_x_order if c in cols), None)
            if x_axis is None and cols:
                # fallback to first non-numeric column
                non_numeric_cols = [c for c in cols if c not in numeric_cols]
                x_axis = non_numeric_cols[0] if non_numeric_cols else (cols[0] if cols else None)

            # Prefer meaningful numeric metrics for y
            preferred_y_order = [
                "total_price", "total_sales", "revenue", "amount", "price_per_unit",
                "avg_price", "count", "quantity", "sum", "avg"
            ]
            y_axis = next((c for c in preferred_y_order if c in cols), None)
            if y_axis is None and numeric_cols:
                # pick first numeric that's not the x-axis
                y_axis = next((c for c in numeric_cols if c != x_axis), numeric_cols[0])

            # If we still cannot determine axes, fall back to table view
            if not x_axis or (chart_type != "table" and not y_axis):
                logger.debug(f"Unable to determine axes for chart. Falling back to table. cols={cols}, numeric={numeric_cols}")
                viz_options = VisualizationOptions(
                    chart_type="table",
                    title=f"Results: {natural_query}"
                )
            else:
                viz_options = VisualizationOptions(
                    chart_type=chart_type,
                    title=f"Results: {natural_query}",
                    x_axis=x_axis,
                    y_axis=y_axis
                )

            fig = viz_agent.visualize(result.data, viz_options)
            return result_df, fig, explanation
        else:
            return result_df, None, explanation
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

def get_db_schema():
    """Get database schema for reference"""
    try:
        with sqlite3.connect("data/product_sales.db") as conn:  # Use correct database
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            schema = {}
            for table in tables:
                table_name = table[0]
                cursor.execute("PRAGMA table_info({})".format(table_name))
                columns = [col[1] for col in cursor.fetchall()]
                schema[table_name] = columns
            return schema
    except Exception as e:
        return {"Error": str(e)}

# Custom CSS for modern styling
custom_css = """
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header-section {
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    color: white;
}

.query-section {
    background: white;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.results-section {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.example-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}

.schema-display {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    font-family: 'Monaco', 'Menlo', monospace;
}

.gradient-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
}

.gradient-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}
"""

# Create the interface with modern styling
with gr.Blocks(
    title="QueryGPT-style SQL Assistant", 
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:
    
    # Header section
    with gr.Column(elem_classes="header-section"):
        gr.Markdown(
            """
            # 🚀 AI-Powered SQL Assistant
            ### Transform natural language into SQL queries with AI
            
            Ask questions about your data in plain English and get instant SQL queries with visualizations
            """,
            elem_classes="header-text"
        )
    
    # Main interface
    with gr.Column(elem_classes="main-container"):
        
        # Query input section
        with gr.Column(elem_classes="query-section"):
            gr.Markdown("## 💬 Ask Your Question")
            
            with gr.Row():
                with gr.Column(scale=4):
                    query_input = gr.Textbox(
                        label="",
                        placeholder="What would you like to know? (e.g., 'Show me revenue by region' or 'Which customers bought the most?')",
                        lines=3,
                        elem_classes="query-input"
                    )
                with gr.Column(scale=1):
                    chart_type = gr.Dropdown(
                        choices=["bar", "line", "pie", "table", "none"],
                        value="bar",
                        label="📊 Visualization",
                        elem_classes="chart-selector"
                    )
            
            with gr.Row():
                run_btn = gr.Button(
                    "🔮 Generate SQL & Analyze", 
                    variant="primary", 
                    size="lg",
                    elem_classes="gradient-btn"
                )
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        # Results section
        with gr.Column(elem_classes="results-section"):
            gr.Markdown("## 📊 Results & Analysis")
            
            # SQL output section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Generated SQL")
                    sql_output = gr.Code(
                        label="",
                        language="sql",
                        lines=5,
                        interactive=False
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 📈 Token Usage & Performance")
                    performance_output = gr.Markdown()
            
            # Data results and visualization
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Query Results")
                    results = gr.Dataframe(
                        label="",
                        wrap=True,
                        height=400
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Visualization")
                    plot_output = gr.Plot(label="")
            
            # Explanation section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧠 AI Analysis & Explanation")
                    explanation_output = gr.Markdown(elem_classes="explanation-panel")
    
    # Sidebar with examples and schema
    with gr.Accordion("💡 Example Questions", open=False):
        gr.Markdown(
            """
            <div class="example-card">
            <strong>Revenue Analysis:</strong><br>
            • "Show me revenue by region"<br>
            • "What's the total sales for each category?"<br>
            • "Which payment method generates the most revenue?"
            </div>
            
            <div class="example-card">
            <strong>Customer Insights:</strong><br>
            • "Top 5 customers by total purchases"<br>
            • "How many orders per customer segment?"<br>
            • "Which customers haven't ordered recently?"
            </div>
            
            <div class="example-card">
            <strong>Time-based Analysis:</strong><br>
            • "Revenue trend over time"<br>
            • "Monthly sales comparison"<br>
            • "Best performing months"
            </div>
            """
        )
        
        gr.Examples(
            examples=[
                ["Show me revenue by region", "bar"],
                ["How many orders per category?", "pie"],
                ["Top 5 customers by total purchases", "bar"],
                ["Revenue trend over time", "line"],
                ["Which payment method is most popular?", "pie"],
                ["Average order value by sales channel", "bar"]
            ],
            inputs=[query_input, chart_type],
            label="Click to try these examples:"
        )
    
    with gr.Accordion("🗃️ Database Schema", open=False):
        with gr.Column(elem_classes="schema-display"):
            schema_output = gr.JSON(
                get_db_schema(), 
                label="Available Tables and Columns"
            )
    
    with gr.Accordion("🧠 Semantic Schema Editor", open=False):
        gr.Markdown(
            """
            ### Enhance AI Understanding with Business Context
            Add business descriptions and context for each column to improve SQL generation accuracy.
            """
        )
        
        # Load current semantic schema
        semantic_manager = SemanticSchemaManager("agents/schema_loader/config/semantic_schemas")
        current_schema = semantic_manager.load_schema("product_sales_semantic.json")
        
        if current_schema:
            # Create editable interface for each table
            for table in current_schema.tables:
                with gr.Accordion(f"📊 Table: {table.name}", open=False):
                    gr.Markdown(f"**Purpose:** {table.business_purpose}")
                    
                    # Create input fields for each column
                    column_inputs = {}
                    
                    for col in table.columns:
                        with gr.Group():
                            gr.Markdown(f"**Column: {col.name}** ({col.data_type})")
                            
                            # Create unique component names for this column
                            col_key = f"{table.name}_{col.name}"
                            
                            description_input = gr.Textbox(
                                label="Description",
                                value=col.description,
                                placeholder="Enter a clear description of this column...",
                                lines=2
                            )
                            
                            business_meaning_input = gr.Textbox(
                                label="Business Meaning",
                                value=col.business_meaning,
                                placeholder="Explain what this column means in business terms...",
                                lines=2
                            )
                            
                            unit_input = gr.Textbox(
                                label="Unit (Optional)",
                                value=col.unit or "",
                                placeholder="e.g., USD, percentage, count, etc."
                            )
                            
                            calculation_note_input = gr.Textbox(
                                label="Calculation Note (Optional)",
                                value=col.calculation_note or "",
                                placeholder="How is this value calculated?",
                                lines=2
                            )
                            
                            # Store inputs for later use
                            column_inputs[col_key] = {
                                'description': description_input,
                                'business_meaning': business_meaning_input,
                                'unit': unit_input,
                                'calculation_note': calculation_note_input,
                                'original_col': col
                            }
            
            # Save button and status
            with gr.Row():
                save_semantic_btn = gr.Button(
                    "💾 Save Semantic Schema", 
                    variant="primary",
                    elem_classes="gradient-btn"
                )
                semantic_status = gr.Markdown("Ready to save changes...")
            
            def save_semantic_schema(*inputs):
                """Save the updated semantic schema"""
                try:
                    # Reconstruct the schema with updated values
                    input_index = 0
                    for table in current_schema.tables:
                        for col in table.columns:
                            col_key = f"{table.name}_{col.name}"
                            
                            # Update column with new values (4 inputs per column)
                            col.description = inputs[input_index] if inputs[input_index] else col.description
                            col.business_meaning = inputs[input_index + 1] if inputs[input_index + 1] else col.business_meaning
                            col.unit = inputs[input_index + 2] if inputs[input_index + 2] else None
                            col.calculation_note = inputs[input_index + 3] if inputs[input_index + 3] else None
                            
                            input_index += 4
                    
                    # Save the updated schema
                    semantic_manager.save_schema(current_schema, "product_sales_semantic.json")
                    
                    return "✅ Semantic schema saved successfully! The AI will now use your enhanced descriptions."
                    
                except Exception as e:
                    return f"❌ Error saving semantic schema: {str(e)}"
            
            # Collect all inputs for the save function
            all_inputs = []
            for table in current_schema.tables:
                for col in table.columns:
                    col_key = f"{table.name}_{col.name}"
                    if col_key in column_inputs:
                        all_inputs.extend([
                            column_inputs[col_key]['description'],
                            column_inputs[col_key]['business_meaning'],
                            column_inputs[col_key]['unit'],
                            column_inputs[col_key]['calculation_note']
                        ])
            
            # Connect save button
            save_semantic_btn.click(
                fn=save_semantic_schema,
                inputs=all_inputs,
                outputs=semantic_status
            )
        
        else:
            gr.Markdown("⚠️ No semantic schema found. Please create one first.")
    
    # Event handlers
    def process_query_with_sql_display(natural_query, chart_type):
        """Enhanced processing that separates SQL display from results"""
        try:
            result_df, fig, explanation = process_natural_language_query(natural_query, chart_type)
            
            # Extract SQL from explanation
            sql_match = re.search(r'```sql\n(.*?)\n```', explanation, re.DOTALL)
            sql_code = sql_match.group(1) if sql_match else "No SQL generated"
            
            # Extract performance info
            token_match = re.search(r'tokens: (.*?)\n', explanation)
            time_match = re.search(r'time: (.*?)s', explanation)
            confidence_match = re.search(r'confidence: (.*?)\n', explanation)
            
            performance_info = "### Performance Metrics\n"
            if token_match:
                performance_info += f"**🎯 Token Usage:** {token_match.group(1)}\n"
            if time_match:
                performance_info += f"**⚡ Execution Time:** {time_match.group(1)}s\n"
            if confidence_match:
                performance_info += f"**🎯 Confidence:** {confidence_match.group(1)}\n"
            else:
                performance_info += "**🎯 Confidence:** N/A\n"
            
            return result_df, fig, explanation, sql_code, performance_info
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return None, None, error_msg, error_msg, error_msg
        
    def clear_all():
        """Clear all outputs"""
        return "", "", None, None, "", "", ""
    
    # Connect event handlers
    run_btn.click(
        fn=process_query_with_sql_display,
        inputs=[query_input, chart_type],
        outputs=[results, plot_output, explanation_output, sql_output, performance_output]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[query_input, chart_type, results, plot_output, explanation_output, sql_output, performance_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Use different port to avoid conflict
        share=False,
        show_error=True
    )
