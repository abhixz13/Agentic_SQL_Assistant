"""
Gradio UI for Semantic Schema Management

Allows users to view database columns in a table and optionally provide
business descriptions and context for better SQL generation.
"""

import gradio as gr
import pandas as pd
from typing import Dict, List, Any, Optional
from .semantic_schema import SemanticSchemaManager, SemanticSchema

class SemanticSchemaUI:
    """UI for managing semantic schemas"""
    
    def __init__(self, db_path: str = "data/product_sales.db"):
        self.manager = SemanticSchemaManager()
        self.db_path = db_path
        self.current_schema = None
    
    def create_ui(self) -> gr.Blocks:
        """Create the Gradio UI for semantic schema management"""
        
        with gr.Blocks(title="Database Schema Enhancement") as demo:
            gr.Markdown("# 🗃️ Database Schema Enhancement")
            gr.Markdown("""
            **Enhance your database schema with business context to improve AI SQL generation accuracy.**
            
            1. **View** your current database columns
            2. **Add** business descriptions and context (optional)
            3. **Save** enhanced schema for better SQL generation
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Database info section
                    with gr.Group():
                        gr.Markdown("### 📊 Database Information")
                        db_name = gr.Textbox(
                            label="Database Name",
                            value="product_sales",
                            interactive=True
                        )
                        db_description = gr.Textbox(
                            label="Database Description",
                            placeholder="Brief description of what this database contains...",
                            lines=2,
                            interactive=True
                        )
                
                with gr.Column(scale=1):
                    # Control buttons
                    with gr.Group():
                        gr.Markdown("### 🔧 Actions")
                        load_btn = gr.Button("📁 Load Database Schema", variant="primary")
                        save_btn = gr.Button("💾 Save Enhanced Schema", variant="secondary")
                        reset_btn = gr.Button("🔄 Reset to Basic", variant="secondary")
            
            # Status messages
            status_msg = gr.Markdown("")
            
            # Main schema editing area
            with gr.Group():
                gr.Markdown("### 🏷️ Column Descriptions")
                gr.Markdown("**Add business context to help AI understand your data better:**")
                
                # Schema display/editing table
                schema_df = gr.Dataframe(
                    headers=["Table", "Column", "Type", "Description", "Business Meaning", "Unit", "Sample Values"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    col_count=(7, "fixed"),
                    interactive=True,
                    wrap=True,
                    label="Database Schema - Edit descriptions and business meanings"
                )
            
            # Preview section
            with gr.Group():
                gr.Markdown("### 👁️ Enhanced Context Preview")
                gr.Markdown("This is what the AI will see with your enhanced descriptions:")
                
                context_preview = gr.Textbox(
                    label="LLM Context",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
            
            # Example section
            with gr.Accordion("💡 Examples & Tips", open=False):
                gr.Markdown("""
                ### How to write good business descriptions:
                
                **Column: `discount_percent`**
                - **Description**: "Percentage discount applied to this order"
                - **Business Meaning**: "Promotional discount given to customer, affects total price calculation"
                - **Unit**: "percentage"
                - **Sample Values**: "0, 10, 25, 50"
                
                **Column: `total_price`**
                - **Description**: "Final price paid by customer after all discounts"
                - **Business Meaning**: "Net revenue amount for financial reporting and profitability analysis"
                - **Unit**: "USD"
                
                ### Tips:
                - **Be specific**: Instead of "price", use "unit price before discounts"
                - **Include relationships**: "Links orders to customers for loyalty analysis"
                - **Add constraints**: "Values between 0-100" or "Format: YYYY-MM-DD"
                - **Mention calculations**: "total = quantity × unit_price × (1 - discount/100)"
                """)
            
            # Event handlers
            def load_database_schema(db_name_val: str, db_desc_val: str):
                try:
                    # Create basic schema from database
                    self.current_schema = self.manager.create_semantic_schema_from_database(
                        self.db_path, db_name_val
                    )
                    
                    # Update description if provided
                    if db_desc_val.strip():
                        self.current_schema.description = db_desc_val
                    
                    # Convert to dataframe for editing
                    df_data = self._schema_to_dataframe()
                    
                    # Generate initial context preview
                    context = self.manager.get_enhanced_schema_context(self.current_schema)
                    
                    return (
                        df_data,
                        "✅ Database schema loaded successfully! You can now edit the descriptions.",
                        context
                    )
                    
                except Exception as e:
                    return (
                        pd.DataFrame(),
                        f"❌ Error loading database: {str(e)}",
                        ""
                    )
            
            def save_enhanced_schema(df_data: pd.DataFrame, db_name_val: str, db_desc_val: str):
                try:
                    if self.current_schema is None:
                        return "❌ Please load database schema first"
                    
                    # Update schema from dataframe
                    self._update_schema_from_dataframe(df_data, db_name_val, db_desc_val)
                    
                    # Save to file
                    filename = f"{db_name_val}_enhanced.json"
                    self.manager.save_schema(self.current_schema, filename)
                    
                    # Generate updated context
                    context = self.manager.get_enhanced_schema_context(self.current_schema)
                    
                    return (
                        f"✅ Enhanced schema saved as {filename}",
                        context
                    )
                    
                except Exception as e:
                    return f"❌ Error saving schema: {str(e)}", ""
            
            def reset_to_basic():
                try:
                    if self.current_schema is None:
                        return pd.DataFrame(), "❌ Please load database schema first", ""
                    
                    # Recreate basic schema
                    db_name = self.current_schema.database_name
                    self.current_schema = self.manager.create_semantic_schema_from_database(
                        self.db_path, db_name
                    )
                    
                    df_data = self._schema_to_dataframe()
                    context = self.manager.get_enhanced_schema_context(self.current_schema)
                    
                    return (
                        df_data,
                        "🔄 Reset to basic schema. All custom descriptions removed.",
                        context
                    )
                    
                except Exception as e:
                    return pd.DataFrame(), f"❌ Error resetting: {str(e)}", ""
            
            def update_preview(df_data: pd.DataFrame, db_name_val: str, db_desc_val: str):
                try:
                    if self.current_schema is None:
                        return ""
                    
                    # Update schema from current dataframe
                    self._update_schema_from_dataframe(df_data, db_name_val, db_desc_val)
                    
                    # Generate preview
                    context = self.manager.get_enhanced_schema_context(self.current_schema)
                    return context
                    
                except Exception as e:
                    return f"Error generating preview: {str(e)}"
            
            # Connect events
            load_btn.click(
                load_database_schema,
                inputs=[db_name, db_description],
                outputs=[schema_df, status_msg, context_preview]
            )
            
            save_btn.click(
                save_enhanced_schema,
                inputs=[schema_df, db_name, db_description],
                outputs=[status_msg, context_preview]
            )
            
            reset_btn.click(
                reset_to_basic,
                outputs=[schema_df, status_msg, context_preview]
            )
            
            # Update preview when dataframe changes
            schema_df.change(
                update_preview,
                inputs=[schema_df, db_name, db_description],
                outputs=[context_preview]
            )
            
            # Auto-load on startup
            demo.load(
                load_database_schema,
                inputs=[db_name, db_description],
                outputs=[schema_df, status_msg, context_preview]
            )
        
        return demo
    
    def _schema_to_dataframe(self) -> pd.DataFrame:
        """Convert semantic schema to dataframe for editing"""
        if not self.current_schema:
            return pd.DataFrame()
        
        rows = []
        for table in self.current_schema.tables:
            for col in table.columns:
                sample_values = ", ".join(col.sample_values) if col.sample_values else ""
                
                rows.append({
                    "Table": table.name,
                    "Column": col.name,
                    "Type": col.data_type,
                    "Description": col.description,
                    "Business Meaning": col.business_meaning,
                    "Unit": col.unit or "",
                    "Sample Values": sample_values
                })
        
        return pd.DataFrame(rows)
    
    def _update_schema_from_dataframe(self, df: pd.DataFrame, db_name: str, db_desc: str):
        """Update semantic schema from edited dataframe"""
        if self.current_schema is None:
            return
        
        # Update database info
        self.current_schema.database_name = db_name
        self.current_schema.description = db_desc
        
        # Update columns from dataframe
        for _, row in df.iterrows():
            table_name = row["Table"]
            col_name = row["Column"]
            
            # Find the table and column
            for table in self.current_schema.tables:
                if table.name == table_name:
                    for col in table.columns:
                        if col.name == col_name:
                            # Update with dataframe values
                            col.description = row["Description"]
                            col.business_meaning = row["Business Meaning"]
                            col.unit = row["Unit"] if row["Unit"] else None
                            
                            # Parse sample values
                            sample_str = row["Sample Values"]
                            if sample_str and sample_str.strip():
                                col.sample_values = [s.strip() for s in sample_str.split(",")]
                            else:
                                col.sample_values = None
                            break
                    break

if __name__ == "__main__":
    # Create and launch the UI
    ui = SemanticSchemaUI()
    demo = ui.create_ui()
    demo.launch(share=True)
