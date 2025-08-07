#!/usr/bin/env python3
"""
Test script to verify Gradio integration uses correct SQL generator
"""

from gradio_ui import process_natural_language_query

def test_gradio_integration():
    """Test that the Gradio integration uses our custom SQL generator"""
    
    print("🧪 Testing Gradio Integration with Custom SQL Generator")
    print("=" * 60)
    
    test_queries = [
        "Show me revenue by region",
        "How many orders per category",
        "Top 3 customers by region",
        "Describe the table"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Call the same function that Gradio uses
            result_df, fig, sql_output = process_natural_language_query(query, "none")
            
            print(f"Generated SQL: {sql_output}")
            print(f"Result rows: {len(result_df)}")
            
            if not result_df.empty:
                print(f"Sample data: {result_df.iloc[0].to_dict()}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 Gradio Integration Test Completed!")

if __name__ == "__main__":
    test_gradio_integration() 