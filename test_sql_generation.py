#!/usr/bin/env python3
"""
Test script to verify SQL generation with product_sales dataset
"""

import sqlite3
import pandas as pd
from agents.workflow import SQLWorkflow
from agents.sql_generator.agent import SQLGeneratorAgent

def test_sql_generation():
    """Test SQL generation with product_sales dataset"""
    
    # Initialize components
    workflow = SQLWorkflow("data/product_sales.db")
    sql_generator = SQLGeneratorAgent()
    
    # Test queries
    test_queries = [
        "Show me revenue by region",
        "How many orders per category", 
        "Top 3 customers by region",
        "Describe the table"
    ]
    
    print("Testing SQL Generation with Product Sales Dataset")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        try:
            # Parse intent
            intent = {
                "action": "aggregate" if "revenue" in query.lower() else "select",
                "entity": "product_sales",
                "params": {
                    "function": "sum" if "revenue" in query.lower() else "count",
                    "column": "total_price" if "revenue" in query.lower() else "*",
                    "group_by": "region" if "region" in query.lower() else "category"
                }
            }
            
            # Get schema
            with sqlite3.connect("data/product_sales.db") as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(product_sales)")
                columns_info = cursor.fetchall()
                
                schema = {
                    "tables": {
                        "product_sales": {
                            "columns": [
                                {"name": col[1], "type": col[2]} 
                                for col in columns_info
                            ]
                        }
                    }
                }
            
            # Generate SQL
            sql_result = sql_generator.generate_sql(intent, schema)
            print(f"Generated SQL: {sql_result.sql}")
            print(f"Confidence: {sql_result.confidence:.0%}")
            
            # Execute query
            result = workflow.run(sql_result.sql, schema_context=json.dumps(schema))
            print(f"Result rows: {len(result.data)}")
            if result.data:
                print(f"Sample result: {result.data[0]}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_sql_generation() 