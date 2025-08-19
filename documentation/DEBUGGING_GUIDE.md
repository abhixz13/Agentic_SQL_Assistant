# SQL Assistant Debugging Guide

## Overview
This document chronicles all the issues encountered while building the AI-powered SQL Assistant, their root causes, solutions, and prevention strategies. This serves as a reference for future implementations and troubleshooting.

---

## 🚨 **Issues Encountered & Solutions**

### **1. Repository Structure & Path Issues**

#### **Issue 1.1: Directory Creation Errors**
```bash
mkdir: ../../src: Permission denied
rmdir: sql_assistant: Directory not empty
```

**Root Cause:**
- Incorrect path navigation (`../../` instead of `../`)
- Hidden files preventing directory removal
- Permission issues in parent directories

**Solution:**
```bash
# Correct path navigation
mkdir -p ../src/agents  # One level up, not two

# Handle hidden files
mv sql_assistant/.* Server/  # Move hidden files
mv sql_assistant/* Server/   # Move visible files

# Fix permissions
chmod u+w ../src/
```

**Prevention:**
- Always verify current directory with `pwd`
- Use `ls -la` to check for hidden files
- Test paths with `ls` before using in commands

---

### **2. Python Environment & Dependencies**

#### **Issue 2.1: Missing LangChain Packages**
```bash
ModuleNotFoundError: No module named 'langchain'
ModuleNotFoundError: No module named 'langchain_community'
```

**Root Cause:**
- LangChain packages not installed
- Incorrect package structure assumptions

**Solution:**
```bash
# Install core packages
pip3 install langchain langchain-openai langchain-core

# Verify installation
python3 -c "import langchain; print(f'LangChain version: {langchain.__version__}')"
```

**Prevention:**
- Create `requirements.txt` with specific versions
- Use virtual environments for isolation
- Document all dependencies

#### **Issue 2.2: Deprecated Import Paths**
```python
# OLD (Deprecated)
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

# NEW (Current)
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
```

**Root Cause:**
- LangChain reorganized package structure
- `langchain.chat_models` moved to `langchain_openai`

**Solution:**
- Update all imports to use new package structure
- Check LangChain documentation for current imports

**Prevention:**
- Pin package versions in requirements.txt
- Regularly check for deprecation warnings
- Test imports after package updates

---

### **3. Python Version Compatibility**

#### **Issue 3.1: Union Type Syntax Error**
```python
# ERROR in Python 3.9
limit: int | None = Field(default=None)

# TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

**Root Cause:**
- Python 3.9 doesn't support `|` union syntax (introduced in 3.10)

**Solution:**
```python
from typing import Optional

# Use Optional for older Python versions
limit: Optional[int] = Field(default=None)
```

**Prevention:**
- Check Python version requirements
- Use `typing.Union` or `Optional` for compatibility
- Test code on target Python version

---

### **4. LangChain Agent Configuration**

#### **Issue 4.1: Empty Tools List Error**
```bash
ValueError: Got no tools for ZeroShotAgent. At least one tool must be provided.
```

**Root Cause:**
- Agent initialized with empty tools list `tools=[]`
- Tools were commented out during debugging

**Solution:**
```python
# Ensure at least one valid tool
tools = [parse_user_intent, load_database_schema]
assert all(isinstance(t, BaseTool) for t in tools), "Tools must inherit BaseTool"

agent = initialize_agent(
    tools=tools,  # Non-empty tools list
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
```

**Prevention:**
- Validate tools before agent initialization
- Use assertions to check tool types
- Test agent creation in isolation

#### **Issue 4.2: API Key Configuration**
```bash
OpenAIError: The api_key client option must be set
```

**Root Cause:**
- Missing `OPENAI_API_KEY` environment variable
- Hardcoded fallback key was insufficient for real API calls

**Solution:**
```python
# Method 1: Environment variable
export OPENAI_API_KEY="your-api-key"

# Method 2: Fallback in code
api_key = os.getenv("OPENAI_API_KEY", "your-fallback-key")
llm = ChatOpenAI(openai_api_key=api_key)
```

**Prevention:**
- Use environment variables for sensitive data
- Provide clear error messages for missing keys
- Document environment setup requirements

---

### **5. File Path & Import Resolution**

#### **Issue 5.1: Module Import Errors**
```python
# ERROR
from tools.intent_parser_tool import parse_user_intent
# ModuleNotFoundError: No module named 'tools'

# SOLUTION
from src.core.intent_parser_tool import parse_user_intent
```

**Root Cause:**
- Incorrect import paths after repository restructuring
- Missing `__init__.py` files
- Wrong assumptions about package structure

**Solution:**
```python
# Create missing __init__.py files
touch src/__init__.py
touch src/core/__init__.py
touch src/database/__init__.py

# Use correct relative imports
from src.core.intent_parser_tool import parse_user_intent
from src.database.database_tools import load_database_schema
```

**Prevention:**
- Maintain consistent package structure
- Always create `__init__.py` files for packages
- Test imports after restructuring

#### **Issue 5.2: Database Path Resolution**
```bash
ToolException: Database not found at 'data/db.sqlite'
```

**Root Cause:**
- Relative paths failing when working directory changes
- Agent tools running from different contexts

**Solution:**
```python
# Use absolute paths
import os
db_path = os.path.abspath("data/db.sqlite")

# Wrapper function with path resolution
def load_schema_with_absolute_path(input_text: str = "") -> dict:
    db_path = os.path.abspath("data/db.sqlite")
    return load_database_schema.run(db_path)
```

**Prevention:**
- Use absolute paths for file operations
- Test tools in different working directories
- Create path resolution utilities

---

### **6. Tool Integration & Compatibility**

#### **Issue 6.1: Tool Method Call Errors**
```bash
TypeError: run() missing 1 required positional argument: 'tool_input'
```

**Root Cause:**
- Incorrect tool method invocation
- Schema mismatch between tool definition and usage

**Solution:**
```python
# WRONG
schema = load_database_schema.run()

# CORRECT
schema = load_database_schema.run("data/db.sqlite")

# BETTER - with wrapper
def load_schema_wrapper(input_text: str = "") -> dict:
    return load_database_schema.run(os.path.abspath("data/db.sqlite"))
```

**Prevention:**
- Check tool signatures before calling
- Create wrapper functions for complex tools
- Test tools in isolation before agent integration

#### **Issue 6.2: Schema Context Not Passed**
```python
# ISSUE: Schema loaded but not used
schema = load_schema_as_text.run()  # Loaded but unused

# SOLUTION: Inject schema into tools
def parse_user_intent_with_schema(query: str) -> dict:
    return parse_user_intent.run(query, schema_info=schema)
```

**Root Cause:**
- Schema preloaded but not passed to intent parser
- Missing context injection

**Solution:**
- Create wrapper functions to inject context
- Pass schema as parameter to relevant tools

---

### **7. Command Line & Terminal Issues**

#### **Issue 7.1: Command Syntax Errors**
```bash
# ERROR
pwd  # Should show: /Users/...
# pwd: too many arguments

# CAUSE: Comments included in command
```

**Root Cause:**
- Comments passed as command arguments
- Copy-paste including explanatory text

**Solution:**
```bash
# Run commands without trailing comments
pwd
ls -l
```

**Prevention:**
- Always run clean commands
- Separate documentation from executable code

#### **Issue 7.2: Python Indentation in Terminal**
```python
# ERROR in terminal
python3 -c "
   from src.agents.SQL_generator_agent import agent
   print(agent)
"
# IndentationError: unexpected indent
```

**Root Cause:**
- Leading spaces in multi-line terminal commands

**Solution:**
```python
# Remove leading spaces
python3 -c "
from src.agents.SQL_generator_agent import agent
print(agent)
"
```

**Prevention:**
- Use consistent indentation in terminal commands
- Test commands before execution

---

## 📋 **Debugging Methodology**

### **Step-by-Step Approach**

1. **Isolate the Problem**
   ```bash
   # Test components individually
   python3 -c "from src.core.intent_parser_tool import parse_user_intent"
   python3 -c "from src.database.database_tools import load_database_schema"
   ```

2. **Check Dependencies**
   ```bash
   pip3 list | grep langchain
   python3 --version
   which python3
   ```

3. **Verify Paths**
   ```bash
   pwd
   ls -la
   find . -name "*.py" -type f
   ```

4. **Test Tools Separately**
   ```python
   # Test each tool in isolation before agent integration
   result = tool.run(test_input)
   print(result)
   ```

5. **Incremental Integration**
   ```python
   # Start with minimal agent, add tools one by one
   agent = initialize_agent(tools=[], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
   ```

---

## 🛠 **Best Practices for Prevention**

### **Environment Setup**
```bash
# Create isolated environment
python3 -m venv venv
source venv/bin/activate

# Install with pinned versions
pip install langchain==0.3.27 langchain-openai==0.3.28

# Create requirements.txt
pip freeze > requirements.txt
```

### **Code Organization**
```python
# Always include __init__.py
project/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── intent_parser_tool.py
│   └── database/
│       ├── __init__.py
│       └── database_tools.py
```

### **Error Handling**
```python
# Robust error handling
try:
    result = tool.run(input_data)
except ToolException as e:
    logger.error(f"Tool failed: {e}")
    return {"error": str(e)}
except Exception as e:
    logger.exception("Unexpected error")
    raise
```

### **Testing Strategy**
```python
# Test each component
def test_tool():
    result = parse_user_intent.run("test query")
    assert isinstance(result, dict)
    
def test_agent():
    agent = create_agent()
    assert len(agent.tools) > 0
```

---

## 📊 **Issue Categories & Frequency**

| Category | Count | Severity |
|----------|-------|----------|
| Import/Path Issues | 6 | High |
| Environment/Dependencies | 4 | High |
| Configuration Errors | 3 | Medium |
| Tool Integration | 3 | Medium |
| Command Line Issues | 2 | Low |

---

## 🔄 **Quick Recovery Checklist**

When encountering issues:

- [ ] Check Python version compatibility
- [ ] Verify all packages installed
- [ ] Confirm file paths are correct
- [ ] Test imports individually
- [ ] Validate tool signatures
- [ ] Check working directory
- [ ] Verify environment variables
- [ ] Test with minimal examples
- [ ] Check for deprecation warnings
- [ ] Validate file permissions

---

## 📚 **Resources for Future Reference**

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Migration Guide](https://python.langchain.com/docs/versions/migrating_agents/)
- [Python Typing Documentation](https://docs.python.org/3/library/typing.html)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

---

## 💡 **Key Takeaways**

1. **Environment Isolation**: Always use virtual environments
2. **Path Management**: Use absolute paths for file operations
3. **Incremental Development**: Test components individually before integration
4. **Error Handling**: Implement robust error handling and logging
5. **Documentation**: Keep track of changes and configurations
6. **Version Pinning**: Pin dependency versions to avoid compatibility issues

This guide should help prevent similar issues in future implementations and provide quick solutions when problems arise. 