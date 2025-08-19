#!/usr/bin/env python3
"""
Launch Semantic Schema Management UI

Standalone launcher for the semantic schema enhancement interface.
Users can view database columns and optionally provide business descriptions.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.schema_loader.semantic_ui import SemanticSchemaUI

if __name__ == "__main__":
    print("🚀 Launching Semantic Schema Management UI...")
    print("📊 This interface allows you to enhance your database schema with business context")
    print("🔗 Access the interface in your browser once it starts")
    
    # Create UI instance
    ui = SemanticSchemaUI(db_path="data/product_sales.db")
    
    # Create and launch the interface
    demo = ui.create_ui()
    
    # Launch with sharing enabled
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,  # Different port from main gradio_ui
        share=True,
        show_api=False
    )
