import gradio as gr
import sqlite3
import pandas as pd
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.workflow import SQLWorkflow
from agents.query_executor.agent import QueryExecutorAgent
from agents.sql_generator.agent import SQLGeneratorAgent
from agents.visualization.agent import VisualizationAgent, VisualizationOptions
from custom_sql_generator import parse_query_to_sql

def setup_workflow():
    """Initialize the AI workflow with the correct database"""
    workflow = SQLWorkflow("data/product_sales.db")  # Use the correct database
    sql_generator = SQLGeneratorAgent()
    viz_agent = VisualizationAgent()
    return workflow, sql_generator, viz_agent

def parse_natural_language_to_intent(natural_query: str) -> dict:
    """Convert natural language to structured intent for product_sales dataset"""
    query_lower = natural_query.lower()
    
    # Enhanced intent parsing for product_sales schema
    if "revenue" in query_lower or "sales" in query_lower or "total" in query_lower:
        return {
            "action": "aggregate",
            "entity": "product_sales",
            "params": {"function": "sum", "column": "total_price"}
        }
    elif "count" in query_lower or "how many" in query_lower:
        return {
            "action": "aggregate", 
            "entity": "product_sales",
            "params": {"function": "count", "column": "*"}
        }
    elif "region" in query_lower:
        return {
            "action": "aggregate",
            "entity": "product_sales", 
            "params": {"function": "sum", "column": "total_price", "group_by": "region"}
        }
    elif "category" in query_lower:
        return {
            "action": "aggregate",
            "entity": "product_sales",
            "params": {"function": "sum", "column": "total_price", "group_by": "category"}
        }
    elif "top" in query_lower and ("customer" in query_lower or "customers" in query_lower):
        return {
            "action": "top_customers",
            "entity": "product_sales",
            "params": {"limit": 3, "group_by": "region"}
        }
    elif "describe" in query_lower or "show" in query_lower:
        return {
            "action": "select",
            "entity": "product_sales",
            "params": {"limit": 10}
        }
    else:
        return {
            "action": "select",
            "entity": "product_sales",
            "params": {"limit": 20}
        }

def get_formatted_schema():
    """Get database schema in the format expected by SQL generator"""
    try:
        with sqlite3.connect("data/product_sales.db") as conn:  # Use correct database
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            schema = {"tables": {}}
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()
                
                columns = []
                for col_info in columns_info:
                    col_id, col_name, col_type, not_null, default_val, primary_key = col_info
                    columns.append({
                        "name": col_name,
                        "type": col_type
                        # Removed primary_key and not_null to avoid errors
                    })
                
                schema["tables"][table_name] = {
                    "columns": columns,
                    "foreign_keys": []  # Simplified for now
                }
            
            return schema
    except Exception as e:
        return {"tables": {}, "error": str(e)}

def process_natural_language_query(natural_query, chart_type):
    """Convert natural language to SQL and execute with visualization"""
    try:
        # Initialize AI agents
        workflow, sql_generator, viz_agent = setup_workflow()
        
        # Use custom SQL generator instead of LLM-based one
        sql_query = parse_query_to_sql(natural_query)
        
        # Execute the generated SQL using workflow
        result = workflow.run(sql_query)
        
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
