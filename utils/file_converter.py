import pandas as pd # type: ignore
import sqlite3
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_sqlite_db(
    input_file_path: str,
    table_name: str = "data",
    if_exists: str = "replace",
    index: bool = False,
) -> Optional[str]:
    """
    Convert a CSV/XLSX file to SQLite DB and save it in the root `data` folder.
    
    Args:
        input_file_path (str): Path to the input CSV/XLSX file.
        table_name (str): Name of the table in SQLite DB. Defaults to "data".
        if_exists (str): Action if table exists ("replace", "append", "fail"). Defaults to "replace".
        index (bool): Write DataFrame index as a column. Defaults to False.
    
    Returns:
        Optional[str]: Path to the generated SQLite DB (data/db.sqlite) if successful, None otherwise.
    """
    output_db_path = "data/db.sqlite"  # Hardcoded to root/data/
    
    try:
        # Validate input file
        if not os.path.exists(input_file_path):
            logger.error(f"Input file not found: {input_file_path}")
            return None

        # Read input file
        if input_file_path.endswith('.csv'):
            df = pd.read_csv(input_file_path)
        elif input_file_path.endswith('.xlsx'):
            df = pd.read_excel(input_file_path)
        else:
            logger.error("Unsupported file format. Only CSV/XLSX are supported.")
            return None

        # Ensure the `data` folder exists
        os.makedirs("data", exist_ok=True)

        # Save to SQLite
        with sqlite3.connect(output_db_path) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=index)

        logger.info(f"SQLite DB prepared at: {output_db_path}")
        return output_db_path

    except Exception as e:
        logger.error(f"Error preparing SQLite DB: {e}")
        return None


def extract_schema(db_path: str = "data/db.sqlite") -> dict:
    """
    Extract detailed schema information from SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file
        
    Returns:
        dict: Detailed schema with tables, columns, types, and relationships
    """
    schema_info = {
        "tables": {},
        "relationships": [],
        "summary": ""
    }
    
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logger.warning(f"Database not found at {db_path}. Returning empty schema.")
            return schema_info
            
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                schema_info["tables"][table_name] = {
                    "columns": [],
                    "primary_keys": [],
                    "foreign_keys": []
                }
                
                for col in columns:
                    column_info = {
                        "name": col[1],
                        "type": col[2],
                        "not_null": bool(col[3]),
                        "default": col[4],
                        "primary_key": bool(col[5])
                    }
                    schema_info["tables"][table_name]["columns"].append(column_info)
                    
                    if column_info["primary_key"]:
                        schema_info["tables"][table_name]["primary_keys"].append(col[1])
                
                # Get foreign key information
                cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                foreign_keys = cursor.fetchall()
                
                for fk in foreign_keys:
                    fk_info = {
                        "column": fk[3],
                        "references_table": fk[2],
                        "references_column": fk[4]
                    }
                    schema_info["tables"][table_name]["foreign_keys"].append(fk_info)
                    schema_info["relationships"].append({
                        "from_table": table_name,
                        "from_column": fk[3],
                        "to_table": fk[2],
                        "to_column": fk[4]
                    })
        
        # Generate summary
        table_summaries = []
        for table_name, table_info in schema_info["tables"].items():
            columns = [col["name"] for col in table_info["columns"]]
            table_summaries.append(f"Table '{table_name}': {', '.join(columns)}")
        
        schema_info["summary"] = "\n".join(table_summaries)
        
        logger.info(f"Schema extracted successfully. Found {len(schema_info['tables'])} tables.")
        return schema_info
        
    except Exception as e:
        logger.error(f"Error extracting schema: {e}")
        return {"tables": {}, "relationships": [], "summary": "Error loading schema"}


if __name__ == "__main__":
    # Example usage (replace with your input file)
    prepare_sqlite_db("data/raw_data.xlsx", table_name="bookings") 

    