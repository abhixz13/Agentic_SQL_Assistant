"""
Test script for SQL Generator Agent

Validates SQL generation for:
1. Simple filters
2. Complex multi-condition filters
3. Aggregations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.sql_generator.agent import SQLGeneratorAgent

from dotenv import load_dotenv
load_dotenv()

# Mock database schema matching the 'bookings' table
TEST_SCHEMA = {
    "tables": {
        "bookings": {
            "columns": [
                {"name": "id", "type": "INTEGER", "primary_key": True, "not_null": True},
                {"name": "city", "type": "TEXT", "primary_key": False, "not_null": True},
                {"name": "amount", "type": "FLOAT", "primary_key": False, "not_null": False},
                {"name": "date", "type": "DATE", "primary_key": False, "not_null": True}
            ],
            "foreign_keys": []
        }
    }
}

def test_simple_filter():
    """Test basic filtering with single condition"""
    agent = SQLGeneratorAgent(api_key=os.getenv("OPENAI_API_KEY"))
    
    intent = {
        "action": "filter",
        "entity": "bookings",
        "params": {"city": "New York"}
    }
    
    result = agent.generate_sql(intent, TEST_SCHEMA)
    print("\n=== Simple Filter Test ===")
    print(f"Input: {intent}")
    print(f"Generated SQL: {result.sql}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Metadata: {result.metadata}")

def test_complex_filter():
    """Test filtering with multiple conditions"""
    agent = SQLGeneratorAgent(api_key=os.getenv("OPENAI_API_KEY"))
    
    intent = {
        "action": "filter",
        "entity": "bookings",
        "params": {"city": "NYC", "amount": "> 1000", "date": "2024-01-01"}
    }
    
    result = agent.generate_sql(intent, TEST_SCHEMA)
    print("\n=== Complex Filter Test ===")
    print(f"Input: {intent}")
    print(f"Generated SQL: {result.sql}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Metadata: {result.metadata}")

def test_aggregation():
    """Test aggregation with GROUP BY"""
    agent = SQLGeneratorAgent(api_key=os.getenv("OPENAI_API_KEY"))
    
    intent = {
        "action": "aggregate",
        "entity": "bookings",
        "params": {"group_by": "city", "function": "sum", "column": "amount"}
    }
    
    result = agent.generate_sql(intent, TEST_SCHEMA)
    print("\n=== Aggregation Test ===")
    print(f"Input: {intent}")
    print(f"Generated SQL: {result.sql}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Metadata: {result.metadata}")

if __name__ == "__main__":
    print("=== Starting SQL Generator Tests ===")
    test_simple_filter()
    test_complex_filter()
    test_aggregation()
    print("\n=== All tests completed successfully ===") 