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
from agents.intent_parser.agent import IntentParserAgent
from agents.sql_generator.agent import SQLGeneratorAgent
from agents.visualization.agent import VisualizationAgent, VisualizationOptions
from utils.token_tracker import token_tracker

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

# Global variables to store current database info (set dynamically)
current_db_path = None
current_table_name = None


def setup_workflow():
    """Initialize the AI workflow with automatic database setup"""
    from utils.file_converter import prepare_sqlite_db
    import glob
    from pathlib import Path
    
    # Look for data files first to determine database name
    data_files = []
    for pattern in ['data/*.csv', 'data/*.xlsx', 'data/*.xls']:
        data_files.extend(glob.glob(pattern))
    
    if not data_files:
        # Fallback to default name if no data files found
        db_path = "data/product_sales.db"
        table_name = "product_sales"
        logger.warning("⚠️  No data files found, using default database name")
    else:
        # Use the first data file to determine names
        source_file = data_files[0]
        file_stem = Path(source_file).stem  # Gets filename without extension
        
        # Convert to lowercase and replace spaces/special chars with underscores
        clean_name = file_stem.lower().replace(' ', '_').replace('-', '_')
        
        db_path = f"data/{clean_name}.db"
        table_name = clean_name
        
        logger.info(f"📁 Using data file: {source_file}")
        logger.info(f"🗄️  Database will be: {db_path}")
        logger.info(f"📊 Table will be: {table_name}")
    
    # Check if database exists and has tables
    db_needs_setup = True
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            if tables:  # Database has tables
                db_needs_setup = False
                logger.info(f"✅ Database already exists with {len(tables)} table(s)")
        except Exception as e:
            logger.warning(f"Database exists but couldn't check tables: {e}")
    
    # If database needs setup, convert the data file
    if db_needs_setup and data_files:
        logger.info("🔄 Creating database from data file...")
        source_file = data_files[0]
        
        try:
            # Convert to SQLite database with dynamic names
            result_db = prepare_sqlite_db(source_file, table_name=table_name)
            if result_db and os.path.exists(result_db):
                # If the created database is not at the expected location, move it
                if result_db != db_path:
                    import shutil
                    shutil.move(result_db, db_path)
                    logger.info(f"✅ Database created and moved from {result_db} to {db_path}")
                else:
                    logger.info(f"✅ Database created successfully at {db_path}")
            else:
                logger.error(f"❌ Failed to create database from data file. Expected: {result_db}, Found: {os.path.exists(result_db) if result_db else 'None'}")
        except Exception as e:
            logger.error(f"❌ Error creating database: {e}")
    
    # Store the database info globally for other functions to use
    global current_db_path, current_table_name
    current_db_path = db_path
    current_table_name = table_name
    
    workflow = SQLWorkflow(db_path)
    intent_parser = IntentParserAgent()
    sql_generator = SQLGeneratorAgent()
    viz_agent = VisualizationAgent()
    return workflow, intent_parser, sql_generator, viz_agent



def generate_dynamic_examples():
    """Generate dynamic examples based on actual database schema"""
    try:
        schema = get_db_schema()
        if not schema.get("tables"):
            return [], "", ""
        
        # Get first table and its columns for example generation
        table_name = list(schema["tables"].keys())[0]
        columns = [col["name"] for col in schema["tables"][table_name]["columns"]]
        
        # Categorize columns for smart example generation
        numeric_cols = []
        categorical_cols = []
        date_cols = []
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'year', 'month']):
                date_cols.append(col)
            elif any(keyword in col_lower for keyword in ['amount', 'price', 'cost', 'value', 'total', 'count', 'quantity', 'booking', 'revenue', 'sales']):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        examples = []
        placeholder_text = f"Ask questions about your {table_name} data"
        
        # Generate examples based on available columns
        if numeric_cols and categorical_cols:
            examples.append([f"Show me {numeric_cols[0]} by {categorical_cols[0]}", "bar"])
            if len(categorical_cols) > 1:
                examples.append([f"How many records per {categorical_cols[1]}?", "pie"])
        
        if numeric_cols:
            examples.append([f"What is the total {numeric_cols[0]}?", "table"])
            if len(numeric_cols) > 1:
                examples.append([f"Compare {numeric_cols[0]} and {numeric_cols[1]}", "bar"])
        
        if date_cols and numeric_cols:
            examples.append([f"{numeric_cols[0]} trend over {date_cols[0]}", "line"])
        
        if categorical_cols:
            examples.append([f"Top 5 records by {categorical_cols[0]}", "bar"])
            placeholder_text = f"Ask questions about your {table_name} data (e.g., 'Show me {numeric_cols[0] if numeric_cols else 'data'} by {categorical_cols[0]}')"
        
        # Generate example cards text
        example_cards = f"""
        <div class="example-card">
        <strong>Data Analysis:</strong><br>
        • "Show me data grouped by {categorical_cols[0] if categorical_cols else 'category'}"<br>
        • "What is the total {numeric_cols[0] if numeric_cols else 'amount'}?"<br>
        • "Compare different {categorical_cols[1] if len(categorical_cols) > 1 else 'groups'}"
        </div>
        
        <div class="example-card">
        <strong>Insights:</strong><br>
        • "Top 5 records by {numeric_cols[0] if numeric_cols else 'value'}"<br>
        • "How many records per {categorical_cols[0] if categorical_cols else 'category'}?"<br>
        • "Which {categorical_cols[0] if categorical_cols else 'item'} has the highest {numeric_cols[0] if numeric_cols else 'value'}?"
        </div>
        """
        
        if date_cols:
            example_cards += f"""
            <div class="example-card">
            <strong>Time-based Analysis:</strong><br>
            • "{numeric_cols[0] if numeric_cols else 'Data'} trend over {date_cols[0]}"<br>
            • "Monthly {numeric_cols[0] if numeric_cols else 'summary'}"<br>
            • "Best performing {date_cols[0].replace('_', ' ')}"
            </div>
            """
        
        return examples, placeholder_text, example_cards
            
    except Exception as e:
        logger.warning(f"Could not generate dynamic examples: {e}")
        return [], "Ask questions about your data", ""

def get_current_db_info():
    """Get current database info dynamically"""
    global current_db_path, current_table_name
    
    # If not set, determine dynamically
    if current_db_path is None or current_table_name is None:
        import glob
        from pathlib import Path
        
        # Look for data files
        data_files = []
        for pattern in ['data/*.csv', 'data/*.xlsx', 'data/*.xls']:
            data_files.extend(glob.glob(pattern))
        
        if data_files:
            source_file = data_files[0]
            file_stem = Path(source_file).stem
            clean_name = file_stem.lower().replace(' ', '_').replace('-', '_')
            current_db_path = f"data/{clean_name}.db"
            current_table_name = clean_name
        else:
            # Ultimate fallback - check for any .db files
            db_files = glob.glob('data/*.db')
            if db_files:
                current_db_path = db_files[0]
                current_table_name = Path(db_files[0]).stem
            else:
                # No data found
                current_db_path = "data/unknown.db"
                current_table_name = "unknown"
    
    return current_db_path, current_table_name

def get_db_schema():
    """Get detailed database schema in format expected by IntentParserAgent"""
    try:
        db_path, table_name = get_current_db_info()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            schema = {
                "tables": {},
                "relationships": []
            }
            
            for table in tables:
                table_name = table[0]
                cursor.execute("PRAGMA table_info({})".format(table_name))
                table_info = cursor.fetchall()
                
                columns = []
                for col in table_info:
                    columns.append({
                        "name": col[1],
                        "type": col[2],
                        "not_null": bool(col[3]),
                        "default": col[4],
                        "primary_key": bool(col[5])
                    })
                
                # Get foreign keys
                cursor.execute("PRAGMA foreign_key_list({})".format(table_name))
                foreign_keys = []
                for fk in cursor.fetchall():
                    foreign_keys.append({
                        "column": fk[3],
                        "references_table": fk[2],
                        "references_column": fk[4]
                    })
                
                schema["tables"][table_name] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys
                }
            
            # Try to load and integrate semantic schema if available
            try:
                from agents.schema_loader.semantic_schema import SemanticSchemaManager
                semantic_manager = SemanticSchemaManager("agents/schema_loader/config/semantic_schemas")
                semantic_file = f"{table_name}_semantic.json"
                semantic_schema = semantic_manager.load_schema(semantic_file)
                
                if semantic_schema:
                    # Enhance schema with semantic information
                    enhanced_context = semantic_manager.get_combined_schema_context(semantic_schema, schema)
                    schema["semantic_context"] = enhanced_context
                    logger.info("📊 Using enhanced semantic schema")
                else:
                    logger.info("📊 Using standard schema (no semantic enhancement)")
            except Exception as semantic_error:
                logger.info(f"📊 Using standard schema (semantic error: {semantic_error})")
            
            return schema
    except Exception as e:
        return {"tables": {}, "relationships": [], "Error": str(e)}

def process_natural_language_query(natural_query, chart_type):
    """Convert natural language to SQL and execute with visualization"""
    logger.debug(f"Processing query: {natural_query}")
    try:
        # Reset token tracker at the start of one end-to-end request
        token_tracker.reset()
        # Initialize AI agents
        workflow, intent_parser, sql_generator, viz_agent = setup_workflow()
        
        # Get database schema for context
        schema = get_db_schema()
        
        # Parse natural language to structured intent using agent
        intent = intent_parser.parse(natural_query, schema)
        
        # Generate SQL using LLM-based generator with enhanced schema
        sql_result = sql_generator.generate_sql(intent, schema)
        sql_query = sql_result.sql
        
        # Execute the generated SQL using workflow
        result = workflow.run(sql_query, schema_context=json.dumps(schema))
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
            with sqlite3.connect(current_db_path) as plan_conn:
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
                model_bits.append("- Fixed schema/table references automatically")
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

            # Dynamically determine categorical dimensions for x-axis
            # Prefer text/string columns, date columns, and ID columns
            categorical_cols = []
            date_cols = []
            id_cols = []
            
            for col in cols:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'year', 'month']):
                    date_cols.append(col)
                elif any(keyword in col_lower for keyword in ['id', 'name', 'type', 'category', 'region', 'channel']):
                    id_cols.append(col)
                elif col not in numeric_cols:
                    categorical_cols.append(col)
            
            # Priority order: dates, then IDs/categories, then other non-numeric
            preferred_x_candidates = date_cols + id_cols + categorical_cols
            x_axis = preferred_x_candidates[0] if preferred_x_candidates else (cols[0] if cols else None)

            # Dynamically determine numeric metrics for y-axis
            # Prefer columns with keywords suggesting aggregatable values
            aggregatable_cols = []
            for col in numeric_cols:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['total', 'sum', 'amount', 'price', 'cost', 'value', 'revenue', 'sales', 'count', 'quantity', 'booking']):
                    aggregatable_cols.append(col)
            
            # Use aggregatable columns first, then any numeric column
            preferred_y_candidates = aggregatable_cols + [c for c in numeric_cols if c not in aggregatable_cols]
            y_axis = next((c for c in preferred_y_candidates if c != x_axis), preferred_y_candidates[0] if preferred_y_candidates else None)

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
                        placeholder="Ask questions about your data",
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
        gr.Markdown("""
            <div class="example-card">
            <strong>Data Analysis:</strong><br>
            • "Show me data grouped by category"<br>
            • "What is the total amount?"<br>
            • "Compare different groups"
            </div>
            
            <div class="example-card">
            <strong>Insights:</strong><br>
            • "Top 5 records by value"<br>
            • "How many records per category?"<br>
            • "Which item has the highest value?"
            </div>
            """)
        
        # Static examples that work for any dataset
        gr.Examples(
            examples=[
                ["What is the total amount?", "table"],
                ["Show me data by category", "bar"],
                ["Top 5 records", "bar"],
                ["Compare different groups", "pie"]
            ],
            inputs=[query_input, chart_type],
            label="Click to try these examples:"
        )
    
    with gr.Accordion("🗃️ Database Schema", open=False):
        with gr.Column(elem_classes="schema-display"):
            schema_output = gr.JSON(
            get_db_schema().get("tables", {}), 
                label="Available Tables and Columns"
            )
    
    with gr.Accordion("🧠 Semantic Schema Editor", open=False):
        gr.Markdown(
            """
            ### Enhance AI Understanding with Business Context
            Add business descriptions and context for each column to improve SQL generation accuracy.
            """
        )
        
        # Semantic schema will be loaded dynamically when needed
        def load_semantic_schema_ui():
            """Load semantic schema dynamically"""
            try:
                from agents.schema_loader.semantic_schema import SemanticSchemaManager
                _, table_name = get_current_db_info()
                semantic_manager = SemanticSchemaManager("agents/schema_loader/config/semantic_schemas")
                semantic_file = f"{table_name}_semantic.json"
                return semantic_manager.load_schema(semantic_file)
            except Exception as e:
                logger.warning(f"Could not load semantic schema: {e}")
                return None
        
        current_schema = load_semantic_schema_ui()
        
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
                                placeholder="e.g., currency, percentage, count, etc."
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
                    from agents.schema_loader.semantic_schema import SemanticSchemaManager
                    _, table_name = get_current_db_info()
                    semantic_manager = SemanticSchemaManager("agents/schema_loader/config/semantic_schemas")
                    semantic_file = f"{table_name}_semantic.json"
                    semantic_manager.save_schema(current_schema, semantic_file)
                    
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
        server_port=7862,  # Use different port to avoid conflict
        share=False,
        show_error=True
    )
