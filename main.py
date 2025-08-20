# main.py
from agents.workflow import SQLWorkflow
from agents.sql_generator.agent import SQLGeneratorAgent
import sqlite3
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_test_db():
    """Initialize test database with sample data"""
    conn = sqlite3.connect('data/db.sqlite')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY,
            city TEXT NOT NULL,
            amount REAL,
            checkin_date TEXT
        )
    """)
    cursor.execute("DELETE FROM bookings")
    cursor.executemany(
        "INSERT INTO bookings (city, amount, checkin_date) VALUES (?, ?, ?)",
        [
            ('NYC', 100.50, '2024-01-01'),
            ('London', 200.75, '2024-01-02'),
            ('Paris', 300.00, '2024-01-03')
        ]
    )
    conn.commit()
    conn.close()

def get_test_schema() -> dict:
    """Return mock schema for testing"""
    return {
        "tables": {
            "bookings": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "city", "type": "TEXT", "not_null": True},
                    {"name": "amount", "type": "REAL"},
                    {"name": "checkin_date", "type": "TEXT"}
                ],
                "foreign_keys": []
            }
        }
    }

def main():
    # Initialize test environment
    setup_test_db()
    schema = get_test_schema()
    
    # Create workflow components
    workflow = SQLWorkflow("data/db.sqlite")
    sql_generator = SQLGeneratorAgent()
    
    # Test natural language to SQL conversion
    test_intents = [
        {
            "action": "filter",
            "entity": "bookings",
            "params": {"city": "NYC"}
        },
        {
            "action": "aggregate",
            "entity": "bookings", 
            "params": {"group_by": "city", "function": "sum", "column": "amount"}
        }
    ]
    
    for intent in test_intents:
        print(f"\nProcessing intent: {intent['action']} on {intent['entity']}")
        
        # Step 1: Generate SQL
        try:
            sql_query = sql_generator.generate_sql(intent, schema)
            print(f"Generated SQL: {sql_query.sql}")
            print(f"Confidence: {sql_query.confidence:.0%}")
            
            # Step 2: Execute query
            result = workflow.run(sql_query.sql, schema_context=json.dumps(schema))
            print("Execution successful. First row:", result.data[0])
            
        except Exception as e:
            print(f"Error processing intent: {str(e)}")

if __name__ == "__main__":
    main()