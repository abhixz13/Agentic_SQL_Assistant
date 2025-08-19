"""
Semantic Schema Enhancement Module

Allows users to provide business descriptions and context for database columns
to improve LLM understanding and SQL generation accuracy.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ColumnSemantic:
    """Semantic description for a database column"""
    name: str
    description: str
    business_meaning: str
    data_type: str
    sample_values: Optional[List[str]] = None
    unit: Optional[str] = None  # e.g., "USD", "percentage", "count"
    constraints: Optional[Dict[str, Any]] = None  # e.g., {"min": 0, "max": 100}
    related_columns: Optional[List[str]] = None
    calculation_note: Optional[str] = None

@dataclass
class TableSemantic:
    """Semantic description for a database table"""
    name: str
    description: str
    primary_key: str
    columns: List[ColumnSemantic]
    business_purpose: str
    common_queries: Optional[List[str]] = None

@dataclass
class SemanticSchema:
    """Complete semantic schema with business context"""
    database_name: str
    description: str
    tables: List[TableSemantic]
    relationships: Optional[List[Dict[str, str]]] = None
    business_glossary: Optional[Dict[str, str]] = None

class SemanticSchemaManager:
    """Manages semantic schema definitions and integration with SQL generation"""
    
    def __init__(self, schema_dir: str = "config/semantic_schemas"):
        self.schema_dir = Path(schema_dir)
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
    def create_semantic_schema(self, database_name: str) -> SemanticSchema:
        """Create a new semantic schema template"""
        return SemanticSchema(
            database_name=database_name,
            description=f"Semantic schema for {database_name}",
            tables=[]
        )
    
    def add_table_semantic(self, schema: SemanticSchema, table_name: str, 
                          description: str, business_purpose: str, 
                          primary_key: str) -> TableSemantic:
        """Add semantic information for a table"""
        table_semantic = TableSemantic(
            name=table_name,
            description=description,
            business_purpose=business_purpose,
            primary_key=primary_key,
            columns=[]
        )
        schema.tables.append(table_semantic)
        return table_semantic
    
    def add_column_semantic(self, table: TableSemantic, name: str, 
                           description: str, business_meaning: str,
                           data_type: str, **kwargs) -> ColumnSemantic:
        """Add semantic information for a column"""
        column_semantic = ColumnSemantic(
            name=name,
            description=description,
            business_meaning=business_meaning,
            data_type=data_type,
            **kwargs
        )
        table.columns.append(column_semantic)
        return column_semantic
    
    def save_schema(self, schema: SemanticSchema, filename: Optional[str] = None):
        """Save semantic schema to JSON file"""
        if not filename:
            filename = f"{schema.database_name}_semantic.json"
        
        filepath = self.schema_dir / filename
        with open(filepath, 'w') as f:
            json.dump(asdict(schema), f, indent=2)
        
        print(f"✅ Semantic schema saved to: {filepath}")
    
    def load_schema(self, filename: str) -> Optional[SemanticSchema]:
        """Load semantic schema from JSON file"""
        filepath = self.schema_dir / filename
        if not filepath.exists():
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct objects from dict
        schema = SemanticSchema(**data)
        schema.tables = [
            TableSemantic(
                **{**table_data, 'columns': [
                    ColumnSemantic(**col_data) for col_data in table_data['columns']
                ]}
            ) for table_data in data['tables']
        ]
        return schema
    
    def get_enhanced_schema_context(self, schema: SemanticSchema) -> str:
        """Generate enhanced schema context for LLM with semantic information"""
        context_parts = []
        
        # Database overview
        context_parts.append(f"DATABASE: {schema.database_name}")
        context_parts.append(f"PURPOSE: {schema.description}")
        
        if schema.business_glossary:
            context_parts.append("\nBUSINESS TERMS:")
            for term, definition in schema.business_glossary.items():
                context_parts.append(f"- {term}: {definition}")
        
        context_parts.append("\nTABLES WITH SEMANTIC CONTEXT:")
        
        for table in schema.tables:
            context_parts.append(f"\n📊 TABLE: {table.name}")
            context_parts.append(f"Purpose: {table.business_purpose}")
            context_parts.append(f"Description: {table.description}")
            context_parts.append(f"Primary Key: {table.primary_key}")
            
            context_parts.append("Columns:")
            for col in table.columns:
                col_info = [f"  • {col.name} ({col.data_type})"]
                col_info.append(f"    Business Meaning: {col.business_meaning}")
                col_info.append(f"    Description: {col.description}")
                
                if col.unit:
                    col_info.append(f"    Unit: {col.unit}")
                if col.constraints:
                    col_info.append(f"    Constraints: {col.constraints}")
                if col.sample_values:
                    col_info.append(f"    Sample Values: {', '.join(col.sample_values)}")
                if col.calculation_note:
                    col_info.append(f"    Calculation: {col.calculation_note}")
                
                context_parts.append("\n".join(col_info))
            
            if table.common_queries:
                context_parts.append("Common Query Patterns:")
                for query in table.common_queries:
                    context_parts.append(f"  - {query}")
        
        if schema.relationships:
            context_parts.append("\nTABLE RELATIONSHIPS:")
            for rel in schema.relationships:
                context_parts.append(f"- {rel}")
        
        return "\n".join(context_parts)
    
    def get_combined_schema_context(self, schema: SemanticSchema, basic_schema_info: dict) -> str:
        """Generate combined schema context with both semantic and basic schema details"""
        context_parts = []
        
        # Start with semantic context
        semantic_context = self.get_enhanced_schema_context(schema)
        context_parts.append(semantic_context)
        
        # Add basic schema summary for completeness
        context_parts.append("\n" + "="*50)
        context_parts.append("BASIC SCHEMA SUMMARY:")
        context_parts.append("="*50)
        
        basic_summary = self._create_schema_summary(basic_schema_info)
        context_parts.append(basic_summary)
        
        return "\n".join(context_parts)
    
    def _create_schema_summary(self, schema_info: dict) -> str:
        """Create a concise basic schema summary for the LLM."""
        tables = schema_info.get("tables", {})
        summary_lines = []
        
        for table_name, table_info in tables.items():
            columns = table_info.get("columns", [])
            col_list = []
            
            for col in columns[:10]:  # Limit columns for token efficiency
                col_type = col.get("type", "")
                col_name = col.get("name", "")
                flags = []
                if col.get("primary_key"):
                    flags.append("PK")
                if col.get("not_null"):
                    flags.append("NN")
                
                flag_str = f" ({', '.join(flags)})" if flags else ""
                col_list.append(f"{col_name} {col_type}{flag_str}")
            
            summary_lines.append(f"Table {table_name}:")
            summary_lines.extend([f"  - {col}" for col in col_list])
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def create_semantic_schema_from_database(self, db_path: str, database_name: str) -> SemanticSchema:
        """
        Create a basic semantic schema from database structure.
        Users can then enhance it with business descriptions.
        """
        import sqlite3
        
        schema = self.create_semantic_schema(database_name)
        schema.description = f"Database schema for {database_name}"
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall() if table[0] != "sqlite_sequence"]
            
            for table_name in tables:
                # Get table info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()
                
                # Find primary key
                primary_key = None
                for col in columns_info:
                    if col[5]:  # is_primary_key
                        primary_key = col[1]
                        break
                
                # Create table semantic with basic info
                table = self.add_table_semantic(
                    schema,
                    table_name=table_name,
                    description=f"Table: {table_name}",
                    business_purpose=f"Data storage for {table_name} related information",
                    primary_key=primary_key or "id"
                )
                
                # Add columns with basic info
                for col in columns_info:
                    col_name = col[1]
                    col_type = col[2]
                    
                    self.add_column_semantic(
                        table,
                        name=col_name,
                        description=f"Column: {col_name}",
                        business_meaning=f"Data field for {col_name.replace('_', ' ')}",
                        data_type=col_type
                    )
        
        return schema
    
    def get_schema_template_for_editing(self, schema: SemanticSchema) -> Dict[str, Any]:
        """
        Get a simplified template that users can fill out with business descriptions.
        Returns a structure suitable for UI forms.
        """
        template = {
            "database_name": schema.database_name,
            "database_description": schema.description,
            "tables": []
        }
        
        for table in schema.tables:
            table_template = {
                "table_name": table.name,
                "table_description": table.description,
                "business_purpose": table.business_purpose,
                "primary_key": table.primary_key,
                "columns": []
            }
            
            for col in table.columns:
                col_template = {
                    "name": col.name,
                    "data_type": col.data_type,
                    "description": col.description,
                    "business_meaning": col.business_meaning,
                    "unit": col.unit or "",
                    "sample_values": col.sample_values or [],
                    "calculation_note": col.calculation_note or "",
                    "constraints": col.constraints or {}
                }
                table_template["columns"].append(col_template)
            
            template["tables"].append(table_template)
        
        return template
    
    def update_schema_from_user_input(self, schema: SemanticSchema, user_input: Dict[str, Any]) -> SemanticSchema:
        """
        Update semantic schema with user-provided business descriptions.
        """
        # Update database-level info
        if "database_description" in user_input:
            schema.description = user_input["database_description"]
        
        # Update table and column info
        for table_input in user_input.get("tables", []):
            table_name = table_input["table_name"]
            
            # Find the table in schema
            table = None
            for t in schema.tables:
                if t.name == table_name:
                    table = t
                    break
            
            if table:
                # Update table info
                if "table_description" in table_input:
                    table.description = table_input["table_description"]
                if "business_purpose" in table_input:
                    table.business_purpose = table_input["business_purpose"]
                
                # Update column info
                for col_input in table_input.get("columns", []):
                    col_name = col_input["name"]
                    
                    # Find the column
                    for col in table.columns:
                        if col.name == col_name:
                            # Update with user input (only if provided)
                            if col_input.get("description"):
                                col.description = col_input["description"]
                            if col_input.get("business_meaning"):
                                col.business_meaning = col_input["business_meaning"]
                            if col_input.get("unit"):
                                col.unit = col_input["unit"]
                            if col_input.get("sample_values"):
                                col.sample_values = col_input["sample_values"]
                            if col_input.get("calculation_note"):
                                col.calculation_note = col_input["calculation_note"]
                            if col_input.get("constraints"):
                                col.constraints = col_input["constraints"]
                            break
        
        return schema
    
    def create_example_schema_for_product_sales(self) -> SemanticSchema:
        """
        Create an example semantic schema for the product_sales database 
        to demonstrate the capability. This can be used as a reference.
        """
        schema = self.create_semantic_schema("product_sales")
        schema.description = "E-commerce product sales and customer transaction database"
        schema.business_glossary = {
            "List Price": "Original price before any discounts or promotions",
            "Discount": "Percentage reduction from list price offered to customer",
            "Sales Channel": "Method through which sale was made (online, retail, partner)",
            "Order Status": "Current state of customer order in fulfillment process"
        }
        
        # Add product_sales table semantic information
        table = self.add_table_semantic(
            schema,
            table_name="product_sales",
            description="Complete customer transaction records with product and pricing details",
            business_purpose="Track all product sales, customer behavior, pricing effectiveness, and regional performance",
            primary_key="order_id"
        )
        
        # Add key column semantics (just the most important ones as examples)
        self.add_column_semantic(
            table, "order_id", 
            "Unique identifier for each customer order transaction",
            "Primary key that uniquely identifies a complete customer purchase",
            "TEXT",
            sample_values=["ORD-2024-001", "ORD-2024-002"]
        )
        
        self.add_column_semantic(
            table, "discount_percent",
            "Percentage discount applied to this order",
            "Promotional discount given to customer, affects total price calculation",
            "INTEGER",
            unit="percentage",
            constraints={"min": 0, "max": 100},
            sample_values=["0", "10", "25", "50"],
            calculation_note="Discount amount = (price_per_unit * quantity) * (discount_percent / 100)"
        )
        
        self.add_column_semantic(
            table, "total_price",
            "Final price paid by customer after all discounts",
            "Net revenue amount for financial reporting and profitability analysis",
            "REAL",
            unit="USD",
            calculation_note="total_price = (price_per_unit * quantity) * (1 - discount_percent/100)",
            constraints={"min": 0}
        )
        
        # Add other columns with basic info
        basic_columns = [
            ("product_id", "TEXT", "Unique identifier for each product"),
            ("product_name", "TEXT", "Human-readable product name"),
            ("category", "TEXT", "Product classification"),
            ("quantity", "INTEGER", "Number of units purchased"),
            ("price_per_unit", "REAL", "List price per unit"),
            ("region", "TEXT", "Geographic region"),
            ("sales_channel", "TEXT", "Sales method"),
            ("order_date", "TEXT", "Date of order"),
            ("customer_id", "TEXT", "Customer identifier"),
            ("customer_name", "TEXT", "Customer full name"),
            ("payment_method", "TEXT", "Payment type"),
            ("order_status", "TEXT", "Order status"),
            ("sales_rep", "TEXT", "Sales representative")
        ]
        
        for col_name, col_type, basic_desc in basic_columns:
            self.add_column_semantic(
                table, col_name,
                basic_desc,
                f"Data field for {col_name.replace('_', ' ')}",
                col_type
            )
        
        return schema

if __name__ == "__main__":
    # Example usage - demonstrate flexible schema creation
    manager = SemanticSchemaManager()
    
    # Method 1: Create from any database file
    print("=== CREATING SCHEMA FROM DATABASE ===")
    basic_schema = manager.create_semantic_schema_from_database(
        "../../data/product_sales.db", 
        "product_sales"
    )
    
    # Get template for user editing
    template = manager.get_schema_template_for_editing(basic_schema)
    print(f"Created basic schema with {len(template['tables'])} tables")
    for table in template['tables']:
        print(f"  - {table['table_name']}: {len(table['columns'])} columns")
    
    # Method 2: Create example enhanced schema
    print("\n=== CREATING ENHANCED EXAMPLE ===")
    example_schema = manager.create_example_schema_for_product_sales()
    
    # Save the enhanced version (this is what users would actually use)
    manager.save_schema(example_schema, "product_sales_semantic.json")
    
    # Save basic version for comparison/demo purposes only
    manager.save_schema(basic_schema, "product_sales_basic_demo.json")
    
    # Show difference
    basic_context = manager.get_enhanced_schema_context(basic_schema)
    enhanced_context = manager.get_enhanced_schema_context(example_schema)
    
    print(f"\nBasic schema context length: {len(basic_context)} chars")
    print(f"Enhanced schema context length: {len(enhanced_context)} chars")
    print("Enhancement provides", len(enhanced_context) - len(basic_context), "additional context characters")
