import gradio as gr
import sqlite3
import pandas as pd
from pathlib import Path
import sys
import os
import json, re

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.workflow import SQLWorkflow
from agents.query_executor.agent import QueryExecutorAgent
from agents.sql_generator.agent import SQLGeneratorAgent
from agents.visualization.agent import VisualizationAgent, VisualizationOptions


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
    """Fetch complete and accurate schema metadata from the database."""
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
                
                schema["tables"][table_name] = {
                    "columns": columns,
                    "indexes": indexes
                }
            
            # Print schema to console for debugging
            import json
            print("\n===== ACTUAL DATABASE SCHEMA =====")
            print(json.dumps(schema, indent=2))
            print("=================================\n")
            
            return schema
            
    except Exception as e:
        print(f"Schema Error: {str(e)}")
        return {"tables": {}, "error": str(e)}

def process_natural_language_query(natural_query, chart_type):
    """Convert natural language to SQL and execute with visualization"""
    try:
        # Initialize AI agents
        workflow, sql_generator, viz_agent = setup_workflow()
        
        # Get database schema for context
        schema = get_formatted_schema()
        schema_context = create_schema_context(schema)   # <-- add this
        
        # Parse natural language to structured intent using LLM
        intent = parse_natural_language_to_intent(natural_query, schema)
        
        # Generate SQL using LLM-based generator
        sql_result = sql_generator.generate_sql(intent, schema)
        sql_query = sql_result.sql  # Extract the SQL string from the SQLQuery object
        
        # Execute the generated SQL using workflow
        result = workflow.run(sql_query, schema_context=json.dumps(schema))
        
        # Create visualization if requested
        if chart_type != "none":
            viz_options = VisualizationOptions(
                chart_type=chart_type,
                title=f"Results: {natural_query}",
                x_axis="region" if "region" in sql_query.lower() else "category",
                y_axis="total_price" if "total_price" in sql_query.lower() else "count"
            )
            fig = viz_agent.visualize(result.data, viz_options)
            return pd.DataFrame(result.data), fig, f"Generated SQL: {sql_query}"
        else:
            return pd.DataFrame(result.data), None, f"Generated SQL: {sql_query}"
            
    except Exception as e:
        error_df = pd.DataFrame({"Error": [str(e)]})
        return error_df, None, f"Error: {str(e)}"

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

# Create the interface
with gr.Blocks(title="AI SQL Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI SQL Assistant")
    gr.Markdown("Ask questions in natural language and let AI convert them to SQL!")
    
    with gr.Tab("Natural Language Query"):
        with gr.Row():
            query_input = gr.Textbox(
                label="Ask your question", 
                placeholder="e.g., Show me revenue by region",
                lines=2
            )
            chart_type = gr.Dropdown(
                choices=["none", "bar", "line", "pie", "table"],
                value="bar",
                label="Visualization"
            )
        
        run_btn = gr.Button("Ask AI", variant="primary", size="lg")
        
        with gr.Row():
            results = gr.Dataframe(label="Results")
            plot_output = gr.Plot(label="Visualization")
        
        sql_output = gr.Textbox(label="Generated SQL", lines=3, interactive=False)
        
        run_btn.click(
            fn=process_natural_language_query,
            inputs=[query_input, chart_type],
            outputs=[results, plot_output, sql_output]
        )
    
    with gr.Tab("Database Schema"):
        gr.JSON(get_db_schema(), label="Available Tables and Columns")
    
    with gr.Tab("Example Questions"):
        gr.Examples(
            examples=[
                ["Show me revenue by region", "bar"],
                ["How many orders per category", "pie"],
                ["Top 3 customers by region", "bar"],
                ["Sales by payment method", "bar"],
                ["Revenue trend over time", "line"]
            ],
            inputs=[query_input, chart_type]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Use different port to avoid conflict
        share=False,
        show_error=True
    )
