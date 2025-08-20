"""Core orchestrator for SQL assistant pipeline.

This module defines the :class:`Orchestrator` which wires together the
individual agents involved in answering a natural language question with
SQL.  The orchestration flow is:

    schema retrieval → intent parsing → planning → validation → SQL
    generation → execution → visualization.

Each step can be swapped out by providing a different implementation via a
configuration file (YAML or dictionary).  The configuration uses dotted
Python paths to locate classes and optional constructor parameters for each
step.  Example YAML::

    steps:
      schema_retriever:
        class: core.orchestrator.SQLiteSchemaRetriever
        params:
          db_path: data/db.sqlite
      intent_parser:
        class: agents.intent_parser.agent.IntentParserAgent
      planner:
        class: agents.sql_generator.planner.SQLPlannerAgent
      validator:
        class: services.validation_engine.ValidationEngine
      sql_generator:
        class: agents.sql_generator.agent.SQLGeneratorAgent
      executor:
        class: agents.query_executor.agent.QueryExecutorAgent
        params:
          db_path: data/db.sqlite
      visualizer:
        class: agents.visualization.agent.VisualizationAgent

The orchestrator dynamically imports and instantiates the components based on
this configuration and exposes a :meth:`run` method to execute the pipeline.
"""
from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _import_from_path(path: str):
    """Import ``path`` of the form ``module.submodule:Class`` or
    ``module.submodule.Class`` and return the class."""
    if ":" in path:
        module_path, class_name = path.split(":", 1)
    else:
        module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class SQLiteSchemaRetriever:
    """Simple schema loader for SQLite databases."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_schema(self) -> Dict[str, Any]:
        """Return basic table/column information for the configured database."""
        schema: Dict[str, Any] = {"tables": {}}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table_name,) in cursor.fetchall():
                info_cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = [
                    {
                        "name": row[1],
                        "type": row[2],
                        "not_null": bool(row[3]),
                        "primary_key": bool(row[5]),
                    }
                    for row in info_cursor.fetchall()
                ]
                schema["tables"][table_name] = {"columns": columns}
        return schema


class Orchestrator:
    """Configurable pipeline orchestrator."""

    STEP_ORDER = [
        "schema_retriever",
        "intent_parser",
        "planner",
        "validator",
        "sql_generator",
        "executor",
        "visualizer",
    ]

    def __init__(self, config: Dict[str, Any] | str):
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        self.config = config
        self.steps: Dict[str, Any] = {}
        step_cfg = config.get("steps", {})
        for name in self.STEP_ORDER:
            cfg = step_cfg.get(name)
            if not cfg:
                raise ValueError(f"Missing configuration for step '{name}'")
            cls = _import_from_path(cfg["class"])
            params = cfg.get("params", {})
            self.steps[name] = cls(**params)

    # Convenience properties for typed access
    @property
    def schema_retriever(self):
        return self.steps["schema_retriever"]

    @property
    def intent_parser(self):
        return self.steps["intent_parser"]

    @property
    def planner(self):
        return self.steps["planner"]

    @property
    def validator(self):
        return self.steps["validator"]

    @property
    def sql_generator(self):
        return self.steps["sql_generator"]

    @property
    def executor(self):
        return self.steps["executor"]

    @property
    def visualizer(self):
        return self.steps["visualizer"]

    def run(self, question: str, viz_options: Optional[Dict[str, Any]] = None):
        """Execute the orchestrated workflow for ``question``.

        Parameters
        ----------
        question:
            Natural language question to answer with SQL.
        viz_options:
            Optional visualization configuration passed to the visualizer.

        Returns
        -------
        Any
            Raw execution result or a mapping with ``data`` and
            ``visualization`` when ``viz_options`` are supplied.
        """
        schema = self.schema_retriever.get_schema()
        intent = self.intent_parser.parse(question, schema)
        plan = self.planner.create_plan(intent, schema)
        normalized_plan, _report = self.validator.validate(plan, schema)
        sql_query = self.sql_generator.generate_sql(normalized_plan, schema)
        result = self.executor.execute(sql_query.sql)

        if viz_options is not None:
            fig = self.visualizer.visualize(result.data, viz_options)
            return {"data": result, "visualization": fig}
        return result
