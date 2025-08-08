#!/usr/bin/env python3
"""
Comprehensive test of the final SQL Assistant system
"""

import sqlite3
import pandas as pd
from custom_sql_generator import parse_query_to_sql
from agents.workflow import SQLWorkflow

def test_final_system():
    """Test the complete system with product_sales dataset"""
    
    print("🧪 Testing SQL Assistant with Product Sales Dataset")
    print("=" * 60)
    
    # Test queries and expected results
    test_cases = [
        {
            "query": "Show me revenue by region",
            "expected_sql": "SELECT region, SUM(total_price) as revenue FROM product_sales GROUP BY region ORDER BY revenue DESC",
            "description": "Revenue aggregation by region"
        },
        {
            "query": "How many orders per category",
            "expected_sql": "SELECT category, COUNT(*) as order_count FROM product_sales GROUP BY category ORDER BY order_count DESC",
            "description": "Order count by category"
        },
        {
            "query": "Top 3 customers by region",
            "expected_sql": "SELECT customer_name, region, SUM(total_price) as total_sales FROM product_sales GROUP BY customer_name, region ORDER BY total_sales DESC LIMIT 3",
            "description": "Top customers by region"
        },
        {
            "query": "Describe the table",
            "expected_sql": "SELECT * FROM product_sales LIMIT 10",
            "description": "Sample data display"
        }
    ]
    
    workflow = SQLWorkflow("data/product_sales.db")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['description']}")
        print("-" * 40)
        
        # Generate SQL
        generated_sql = parse_query_to_sql(test_case["query"])
        print(f"Query: {test_case['query']}")
        print(f"Generated SQL: {generated_sql}")
        
        # Verify SQL matches expected
        if generated_sql.strip() == test_case["expected_sql"].strip():
            print("✅ SQL generation: PASSED")
        else:
            print("❌ SQL generation: FAILED")
            print(f"Expected: {test_case['expected_sql']}")
        
        # Execute query
        try:
            # Get schema for error repair
            with sqlite3.connect("data/product_sales.db") as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(product_sales)")
                columns_info = cursor.fetchall()
                schema = {
                    "tables": {
                        "product_sales": {
                            "columns": [{"name": col[1], "type": col[2]} for col in columns_info]
                        }
                    }
                }
            result = workflow.run(generated_sql, schema_context=json.dumps(schema))
            print(f"✅ Query execution: PASSED ({len(result.data)} rows)")
            
            if result.data:
                print(f"Sample result: {result.data[0]}")
                
        except Exception as e:
            print(f"❌ Query execution: FAILED - {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 Final System Test Completed!")
    
    # Show sample data from database
    print("\n📊 Sample Data from Database:")
    with sqlite3.connect("data/product_sales.db") as conn:
        df = pd.read_sql("SELECT * FROM product_sales LIMIT 3", conn)
        print(df.to_string(index=False))

if __name__ == "__main__":
    test_final_system() 