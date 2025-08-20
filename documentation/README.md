# 🧠 Agentic SQL Assistant

An intelligent AI system that converts natural language queries into SQL and visualizations using a multi-agent architecture with a central reasoning coordinator.

## 🏗️ Architecture Overview

```
                   ┌─────────────────────────────┐
                   │  🧠 Reasoning Coordinator    │
                   │  - Thinks step by step       │
                   │  - Plans agent execution     │
                   └────────────┬────────────────┘
                                │
        ┌──────────────────────┼────────────────────────┐
        ▼                      ▼                        ▼
┌────────────────┐   ┌────────────────────┐    ┌────────────────────┐
│ IntentParser   │   │  SQLGeneratorAgent │    │  ChartGeneratorAgent │
│ - Parses query │   │  - Writes SQL      │    │  - Builds chart      │
└────────────────┘   └────────────────────┘    └────────────────────┘
        ▲                      ▼                        ▼
        └─────────────┬────────┴────────────┬────────────┘
                      │                     │
              ┌───────▼───────┐     ┌───────▼─────────┐
              │ SchemaLoader  │     │ QueryExecutor   │
              │ - Loads DB    │     │ - Runs SQL      │
              └───────────────┘     └─────────────────┘
```

## 🎯 Agent Responsibilities

### 🧠 **Reasoning Coordinator** (`agents/coordinator/`)
- **Primary Role**: Central orchestrator and decision maker
- **Responsibilities**:
  - Analyzes user queries for intent and complexity
  - Plans step-by-step execution strategy
  - Coordinates agent interactions
  - Manages workflow and dependencies
  - Provides final response synthesis

### 🔍 **Intent Parser Agent** (`agents/intent_parser/`)
- **Primary Role**: Natural language understanding
- **Responsibilities**:
  - Parses user queries into structured intent
  - Identifies query type (SELECT, INSERT, UPDATE, etc.)
  - Extracts entities, conditions, and requirements
  - Determines data visualization needs

### 📝 **SQL Generator Agent** (`agents/sql_generator/`)
- **Primary Role**: SQL code generation
- **Responsibilities**:
  - Converts structured intent to SQL queries
  - Optimizes queries for performance
  - Validates SQL syntax
  - Handles complex joins and aggregations

### 📊 **Chart Generator Agent** (`agents/chart_generator/`)
- **Primary Role**: Data visualization
- **Responsibilities**:
  - Analyzes query results for visualization potential
  - Generates appropriate chart types
  - Creates interactive visualizations
  - Provides chart customization options

### 🗄️ **Schema Loader Agent** (`agents/schema_loader/`)
- **Primary Role**: Database introspection
- **Responsibilities**:
  - Loads and caches database schemas
  - Provides table and column information
  - Maintains relationship mappings
  - Handles multiple database connections

### ⚡ **Query Executor Agent** (`agents/query_executor/`)
- **Primary Role**: SQL execution and results management
- **Responsibilities**:
  - Executes SQL queries safely
  - Manages database connections
  - Handles query results and formatting
  - Provides execution statistics

## 📁 Project Structure

```
SQL_Assistant_2/
├── agents/                          # All AI agents
│   ├── coordinator/                 # 🧠 Reasoning Coordinator
│   │   ├── __init__.py
│   │   ├── coordinator.py          # Main coordinator logic
│   │   ├── planner.py              # Execution planning
│   │   └── orchestrator.py         # Agent coordination
│   ├── intent_parser/              # 🔍 Intent Parser Agent
│   │   ├── __init__.py
│   │   ├── parser.py               # NL understanding
│   │   └── intent_models.py        # Pydantic models
│   ├── sql_generator/              # 📝 SQL Generator Agent
│   │   ├── __init__.py
│   │   ├── generator.py            # SQL generation
│   │   └── optimizer.py            # Query optimization
│   ├── chart_generator/            # 📊 Chart Generator Agent
│   │   ├── __init__.py
│   │   ├── chart_builder.py        # Chart creation
│   │   └── chart_types.py          # Chart type logic
│   ├── schema_loader/              # 🗄️ Schema Loader Agent
│   │   ├── __init__.py
│   │   ├── loader.py               # Schema extraction
│   │   └── cache.py                # Schema caching
│   └── query_executor/             # ⚡ Query Executor Agent
│       ├── __init__.py
│       ├── executor.py             # SQL execution
│       └── connection_manager.py   # DB connections
├── config/                         # Configuration
│   ├── __init__.py
│   ├── settings.py                 # App settings
│   └── database.py                 # DB configuration
├── shared/                         # Shared utilities
│   ├── __init__.py
│   ├── models.py                   # Common Pydantic models
│   ├── utils.py                    # Helper functions
│   └── constants.py                # Constants
├── data/                           # Data files
│   ├── raw_data.xlsx              # Source data
│   ├── db.sqlite                  # SQLite database
│   └── sample/                    # Sample data
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_coordinator.py        # Coordinator tests
│   ├── test_agents.py             # Agent tests
│   └── test_integration.py        # Integration tests
├── docs/                           # Documentation
│   ├── api.md                     # API documentation
│   └── usage.md                   # Usage guide
├── *.md                           # Project documentation
└── requirements.txt               # Dependencies
```

## 🚀 Key Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Central Coordination**: Reasoning coordinator manages complex workflows
- **Natural Language Processing**: Advanced query understanding
- **Smart SQL Generation**: Context-aware SQL with optimization
- **Dynamic Visualizations**: Automatic chart generation
- **Schema Intelligence**: Dynamic database introspection
- **Extensible Design**: Easy to add new agents and capabilities

## 🔄 Workflow Example

1. **User Query**: "Show me sales by region for Q3 2023 in a bar chart"
2. **Reasoning Coordinator**: 
   - Analyzes query complexity
   - Plans: Intent Parsing → Schema Loading → SQL Generation → Query Execution → Chart Generation
3. **Intent Parser**: Extracts intent (SELECT, GROUP BY region, WHERE date conditions)
4. **Schema Loader**: Provides sales table schema and relationships
5. **SQL Generator**: Creates optimized SQL query
6. **Query Executor**: Runs query and returns results
7. **Chart Generator**: Creates bar chart visualization
8. **Coordinator**: Synthesizes final response with chart and insights

## 🛠️ Technology Stack

- **Framework**: LangChain for agent orchestration
- **LLM**: OpenAI GPT-4 for natural language processing
- **Database**: SQLite with pandas integration
- **Visualization**: Plotly/Matplotlib for charts
- **Architecture**: Multi-agent system with central coordination

## 📖 Documentation

- [Development Plan](DEVELOPMENT_PLAN.md)
- [Product Considerations](Product_consideration.md)
- [Debugging Guide](DEBUGGING_GUIDE.md)
- [TODO List](TODO.md)
- [API Documentation](docs/api.md)
- [Usage Guide](docs/usage.md)

## 🎯 Next Steps

1. Implement Reasoning Coordinator with planning capabilities
2. Create specialized agents with clear interfaces
3. Build agent communication protocols
4. Add comprehensive testing framework
5. Implement monitoring and logging
6. Create user interface for interaction

---

**Built with ❤️ using AI agents that think, plan, and coordinate together.** 