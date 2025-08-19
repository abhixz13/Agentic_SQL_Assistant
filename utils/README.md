# 🛠️ Utils Directory

Utility modules and standalone tools that support the SQL Assistant system.

## 📋 Core Utilities

### 🗄️ **`database_setup.py`**
**Purpose**: Database initialization and setup  
**Usage**: `python utils/database_setup.py`  
**What it does**:
- Converts CSV data to SQLite database
- Creates performance indexes
- Validates data integrity
- One-time setup utility

### 📊 **`semantic_ui_launcher.py`**
**Purpose**: Semantic schema management interface  
**Usage**: `python utils/semantic_ui_launcher.py`  
**What it does**:
- Launches standalone UI for editing semantic schemas
- Allows business context customization
- Runs on separate port from main UI

### 🔄 **`token_tracker.py`**
**Purpose**: OpenAI token usage monitoring  
**Usage**: Imported by other modules  
**What it does**:
- Tracks token consumption across requests
- Provides usage analytics
- Cost monitoring

### 📁 **`file_converter.py`**
**Purpose**: File format conversion utilities  
**Usage**: Imported by other modules  
**What it does**:
- Handles various file format conversions
- Data transformation utilities

## 🏗️ **Organization**

```
utils/
├── README.md                    # This documentation
├── __init__.py                  # Package initialization
├── database_setup.py           # 🗄️ Database setup utility
├── semantic_ui_launcher.py     # 📊 Semantic schema UI launcher
├── token_tracker.py            # 🔄 Token usage tracking
└── file_converter.py           # 📁 File conversion utilities
```

## 🎯 **Design Principle**

The `utils/` folder contains:
- ✅ **Standalone utilities** that can be run independently
- ✅ **Helper modules** imported by the main system
- ✅ **Setup and maintenance tools**
- ✅ **Supporting infrastructure**

This keeps the root directory clean with only the main entry points:
- `gradio_ui.py` - Main web interface
- `main.py` - CLI interface
