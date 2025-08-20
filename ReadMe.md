# SQL Assistant

## Overview
SQL Assistant is an AI-powered tool that converts natural language queries into executable SQL statements. It leverages semantic schemas to enhance understanding and accuracy in query generation.

## Features
- Natural language to SQL intent parsing
- Schema-aware SQL generation
- Query execution with retry logic
- Error classification and recovery
- Multi-agent coordination for complex queries

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SQL_Assistant_2
   ```

2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the Gradio UI:
   ```bash
   python3 gradio_ui.py
   ```

2. Access the application in your web browser at `http://localhost:7862`.

3. Enter your natural language query in the input field and submit.

## System Analysis: Autonomy Assessment

### 💪 Key Strengths

#### 1. Robust Multi-Agent Architecture with Error Recovery
- **Execute→Observe→Repair→Re-execute cycles** with bounded retries (3 attempts)
- **Probing strategy** using `LIMIT 1` to catch errors cheaply before full execution
- **Automatic error classification** for syntax, schema, aggregate, and semantic issues

#### 2. Semantic Schema Integration for Context Awareness
- **Enhanced business context** through semantic mappings (e.g., "SaaS" → `IntersightConsumption`)
- **Dynamic schema-aware prompts** with business meanings and sample values
- **Intelligent column mapping** using semantic understanding

#### 3. Comprehensive SQL Generation Pipeline
- **Three-stage validation**: Planner → Validator → Generator with quality assurance
- **AST parsing and validation** with function allow-lists and auto-rewrite capabilities
- **Few-shot learning** with cached examples for consistent SQL patterns

#### 4. Built-in Observability and Monitoring
- **Thread-safe token tracking** for OpenAI usage monitoring
- **Performance metrics** including query execution time, confidence scores, complexity assessment
- **Comprehensive debug logging** with schema flow tracing

#### 5. Modular and Extensible Design
- **Clean agent separation** for intent parsing, SQL generation, execution, and visualization
- **Configuration-driven architecture** with pluggable policies and validation rules
- **Easy extensibility** through well-defined interfaces

### 🚀 Areas for Complete Autonomy

#### 1. Intelligent Query Disambiguation & Clarification
**Current Gap**: System flags ambiguity but doesn't actively resolve it.
**Improvement**: Implement conversational clarification with context inference and pattern-based disambiguation.

#### 2. Continuous Learning & Adaptive Optimization
**Current Gap**: No learning mechanism from user feedback or query patterns.
**Improvement**: Add adaptive learning from user interactions, pattern mining, and dynamic optimization.

#### 3. Proactive Schema Evolution & Maintenance
**Current Gap**: Manual semantic schema creation and static schema assumptions.
**Improvement**: Auto-discover semantics, detect schema drift, and maintain schema versioning.

#### 4. Autonomous Error Classification & Resolution
**Current Gap**: Error repair uses static checklists rather than intelligent diagnosis.
**Improvement**: ML-based error classification, predictive error prevention, and sophisticated repair strategies.

#### 5. Self-Monitoring & Autonomous Optimization
**Current Gap**: Monitoring exists but no autonomous action on performance issues.
**Improvement**: Real-time performance optimization, self-healing capabilities, and autonomous scaling.

### 🎯 Autonomy Roadmap

**Phase 1: Enhanced Intelligence** (Immediate)
- Query disambiguation with conversational clarification
- Pattern-based learning from user corrections
- Enhanced error classification with LLM-based diagnosis

**Phase 2: Adaptive Behavior** (3-6 months)
- Continuous learning from user interactions
- Autonomous schema evolution detection
- Predictive error prevention capabilities

**Phase 3: Full Autonomy** (6-12 months)
- Complete self-monitoring and optimization
- Autonomous performance tuning and resource management
- Zero-touch operation with intelligent fallback strategies

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.