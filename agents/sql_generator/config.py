"""
Configuration for the SQL Generator Agent.

This module contains configuration settings for the SQL Generator Agent,
including model parameters, validation settings, and optimization options.
"""

# OpenAI Configuration
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.1  # Low temperature for consistent SQL generation
OPENAI_MAX_TOKENS = 300

# SQL Validation Settings
ENABLE_SQL_VALIDATION = True
ENABLE_SCHEMA_VALIDATION = True
ENABLE_QUERY_OPTIMIZATION = True

# Confidence Thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5
FALLBACK_CONFIDENCE = 0.3

# Query Complexity Levels
COMPLEXITY_LEVELS = {
    "simple": "Basic SELECT queries",
    "medium": "Queries with WHERE, GROUP BY, or simple joins",
    "complex": "Queries with subqueries, CTEs, or multiple joins"
}

# Supported SQL Operations
SUPPORTED_OPERATIONS = [
    "select",
    "filter", 
    "aggregate",
    "join",
    "insert",
    "update",
    "delete"
] 