# 🛠️ Scripts Directory

This directory contains standalone utility scripts for managing the SQL Assistant system.

## 📋 Available Scripts

### 🗄️ **`setup_database.py`**
**Purpose**: Initial database setup from CSV data  
**Usage**: `python scripts/setup_database.py`  
**What it does**:
- Converts `data/product_sales_dataset.csv` to `data/product_sales.db`
- Creates performance indexes for faster queries
- Validates data integrity
- Shows database schema and sample data

**When to run**: First time setup or when refreshing database

---

### 📊 **`launch_semantic_ui.py`**
**Purpose**: Semantic Schema Management Interface  
**Usage**: `python scripts/launch_semantic_ui.py`  
**What it does**:
- Launches standalone UI for managing semantic schemas
- Allows users to add business descriptions to database columns
- Provides live preview of enhanced context for LLM
- Runs on port 7862 (separate from main UI)

**When to run**: When you want to customize business descriptions for better SQL generation

---

## 🚀 **Quick Start**

1. **Setup Database** (required):
   ```bash
   python scripts/setup_database.py
   ```

2. **Launch Main SQL Assistant**:
   ```bash
   python gradio_ui.py
   ```

3. **Optional: Customize Semantic Schemas**:
   ```bash
   python scripts/launch_semantic_ui.py
   ```

## 📁 **File Organization**

```
scripts/
├── README.md                 # This documentation
├── setup_database.py         # Database setup utility
└── launch_semantic_ui.py     # Semantic schema management UI
```

These scripts are **standalone utilities** that support the main SQL Assistant but are not part of the core pipeline.
