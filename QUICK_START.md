# 🚀 Quick Start Guide

## 📋 **Setup & Usage**

### **1. Database Setup** (Required - First Time Only)
```bash
python utils/database_setup.py
```
This converts the CSV data to SQLite database with proper indexes.

### **2. Launch Main SQL Assistant**
```bash
python gradio_ui.py
```
This starts the main web interface on `http://localhost:7860`

### **3. Optional: Customize Semantic Schemas**
```bash
python utils/semantic_ui_launcher.py
```
This launches the semantic schema editor on `http://localhost:7862`

---

## 📁 **Project Structure Overview**

```
SQL_Assistant_2/
├── 🎯 gradio_ui.py              # Main web interface
├── 🎯 main.py                   # CLI interface  
├── 📁 agents/                   # Core AI system
├── 📁 data/                     # Database files
├── 📁 documentation/            # All .md files
├── 📁 utils/                    # Utilities & setup tools
│   ├── database_setup.py       # Database initialization
│   └── semantic_ui_launcher.py # Semantic schema editor
└── 📁 tests/                    # Test suite
```

## 🎯 **Core Features**

- **🧠 Semantic Schema Enhancement**: Your brilliant idea - business context for better SQL
- **📋 Planner → Validator → Generator**: 3-stage SQL generation pipeline
- **🤖 Multi-Agent Architecture**: Intent parsing, SQL generation, execution, visualization
- **📊 Dynamic Visualizations**: Automatic chart generation
- **🔄 Token Tracking**: Monitor OpenAI API usage

## 🆘 **Need Help?**

- 📚 **Full Documentation**: See `documentation/` folder
- 🐛 **Debugging**: See `documentation/DEBUGGING_GUIDE.md`
- 🏗️ **Architecture**: See `documentation/PROJECT_STRUCTURE_AND_FLOW.md`
