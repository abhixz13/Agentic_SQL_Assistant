#!/usr/bin/env python3
"""
Setup script to load product_sales_dataset.csv into SQLite database
"""

import pandas as pd
import sqlite3
import os

def setup_product_sales_database():
    """Load product_sales_dataset.csv into SQLite database"""
    
    # Read the CSV file
    csv_path = "data/product_sales_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return False
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    
    # Create SQLite database
    db_path = "data/product_sales.db"
    conn = sqlite3.connect(db_path)
    
    # Load data into database
    df.to_sql("product_sales", conn, index=False, if_exists="replace")
    
    # Create indexes for better performance
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_region ON product_sales(region)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_date ON product_sales(order_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON product_sales(category)")
    
    # Get table info
    cursor.execute("PRAGMA table_info(product_sales)")
    columns = cursor.fetchall()
    print(f"\nDatabase schema:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Sample queries to verify data
    print(f"\nSample data verification:")
    cursor.execute("SELECT COUNT(*) FROM product_sales")
    total_rows = cursor.fetchone()[0]
    print(f"  Total rows: {total_rows}")
    
    cursor.execute("SELECT DISTINCT region FROM product_sales")
    regions = [row[0] for row in cursor.fetchall()]
    print(f"  Regions: {regions}")
    
    cursor.execute("SELECT DISTINCT category FROM product_sales")
    categories = [row[0] for row in cursor.fetchall()]
    print(f"  Categories: {categories}")
    
    conn.close()
    print(f"\nDatabase created successfully at {db_path}")
    return True

if __name__ == "__main__":
    setup_product_sales_database() 