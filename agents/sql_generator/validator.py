"""
SQL Validator Agent Module

This module contains the SQLValidatorAgent that validates and auto-fixes DSL plans 
from the SQL Planner before they are used by the SQL Generator. Adapted from the 
external validator code to integrate with the SQL_Assistant_2 architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import json
import logging

logger = logging.getLogger(__name__)

# ========== Public Data Models ==========

@dataclass
class ValidationIssue:
    severity: str           # "ERROR" | "WARN" | "FIX"
    code: str               # e.g., "UNKNOWN_COLUMN", "AUTO_SAFE_DIV", ...
    message: str
    path: Optional[str] = None  # e.g., "expressions.discount"

@dataclass
class ValidationReport:
    ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, path: Optional[str] = None):
        self.issues.append(ValidationIssue(severity, code, message, path))

    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARN"]

    def fixes(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "FIX"]

@dataclass
class ValidatorPolicy:
    """
    Configuration for SQL validation with SQLite-specific settings.
    """
    allowed_functions: List[str]                          # From Planner function catalog
    rate_name_hints: List[str] = field(default_factory=lambda: [
        "discount", "rate", "ratio", "share", "pct", "percent", "margin"
    ])
    auto_rewrite_division: bool = True                    # turn "a/b" into SAFE_DIV(a,b)
    require_clamp_for_rates: bool = True
    allow_avg_of_ratios: bool = False                     # unless explicitly requested
    warehouse_dialect: str = "sqlite"                     # SQLite dialect
    # Optional: units mapping for type checking
    units: Dict[str, str] = field(default_factory=dict)   # keys = fully-qualified columns
    # Optional: treat these metric names as rates even if name hints don't match
    explicit_rate_metrics: List[str] = field(default_factory=list)

# ========== Expression AST (Parser + Printer) ==========

class Node: ...

@dataclass
class Ident(Node):
    name: str  # "table.column" or function-less symbol

@dataclass
class Number(Node):
    value: str

@dataclass
class Func(Node):
    name: str
    args: List[Node]

@dataclass
class BinOp(Node):
    op: str       # "+", "-", "*", "/"
    left: Node
    right: Node

@dataclass
class Paren(Node):
    inner: Node

# ---- Tokenizer ----
_TOKEN_SPEC = [
    ("WS",     r"[ \t\r\n]+"),
    ("NUMBER", r"(?:\d+(?:\.\d*)?|\.\d+)"),
    ("IDENT",  r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"),
    ("COMMA",  r","),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("OP",     r"[+\-*/]"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))

@dataclass
class Token:
    kind: str
    text: str

def tokenize(s: str) -> List[Token]:
    out: List[Token] = []
    for m in _TOKEN_RE.finditer(s):
        k = m.lastgroup
        t = m.group()
        if k == "WS":
            continue
        out.append(Token(k, t))
    return out

# ---- Pratt Parser (precedence-aware) ----
class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[Token]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, kind: Optional[str] = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression.")
        if kind and tok.kind != kind:
            raise ValueError(f"Expected {kind}, got {tok.kind} ({tok.text})")
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.parse_expr(0)
        if self.peek() is not None:
            raise ValueError(f"Unexpected trailing token: {self.peek().text}")
        return node

    # binding powers
    BP = {
        "+": 10,
        "-": 10,
        "*": 20,
        "/": 20,
    }

    def parse_expr(self, min_bp: int) -> Node:
        # prefix
        tok = self.eat()
        if tok.kind == "NUMBER":
            left: Node = Number(tok.text)
        elif tok.kind == "IDENT":
            # function call or identifier
            if self.peek() and self.peek().kind == "LPAREN":
                fn_name = tok.text
                self.eat("LPAREN")
                args: List[Node] = []
                if self.peek() and self.peek().kind != "RPAREN":
                    while True:
                        # Special case: handle * as a valid function argument (e.g., COUNT(*))
                        if self.peek() and self.peek().kind == "OP" and self.peek().text == "*":
                            star_tok = self.eat("OP")
                            args.append(Ident("*"))  # Treat * as a special identifier
                        else:
                            args.append(self.parse_expr(0))
                        
                        if self.peek() and self.peek().kind == "COMMA":
                            self.eat("COMMA")
                            continue
                        break
                self.eat("RPAREN")
                left = Func(fn_name, args)
            else:
                left = Ident(tok.text)
        elif tok.kind == "LPAREN":
            inner = self.parse_expr(0)
            self.eat("RPAREN")
            left = Paren(inner)
        elif tok.kind == "OP" and tok.text == "-":
            # unary minus
            right = self.parse_expr(self.BP["*"])  # high precedence
            left = BinOp("*", Number("-1"), right)
        else:
            raise ValueError(f"Unexpected token: {tok.kind} ({tok.text})")

        # infix
        while self.peek() and self.peek().kind == "OP":
            op_tok = self.peek()
            op = op_tok.text
            lbp = self.BP[op]
            if lbp < min_bp:
                break
            # consume op
            self.eat("OP")
            # right operand with right-binding power
            rbp = lbp + 1
            right = self.parse_expr(rbp)
            left = BinOp(op, left, right)
        return left

# ---- AST Utilities ----
def ast_walk(n: Node, fn):
    """Generic visitor."""
    fn(n)
    if isinstance(n, Func):
        for a in n.args:
            ast_walk(a, fn)
    elif isinstance(n, BinOp):
        ast_walk(n.left, fn)
        ast_walk(n.right, fn)
    elif isinstance(n, Paren):
        ast_walk(n.inner, fn)

def ast_map(n: Node, fn) -> Node:
    """Map/transform every node; return new root."""
    n2 = fn(n)
    if isinstance(n2, Func):
        return Func(n2.name, [ast_map(a, fn) for a in n2.args])
    if isinstance(n2, BinOp):
        return BinOp(n2.op, ast_map(n2.left, fn), ast_map(n2.right, fn))
    if isinstance(n2, Paren):
        return Paren(ast_map(n2.inner, fn))
    return n2

def ast_to_str(n: Node) -> str:
    if isinstance(n, Number):
        return n.value
    if isinstance(n, Ident):
        return n.name
    if isinstance(n, Func):
        return f"{n.name}(" + ", ".join(ast_to_str(a) for a in n.args) + ")"
    if isinstance(n, BinOp):
        # parenthesize to be safe
        return f"({ast_to_str(n.left)} {n.op} {ast_to_str(n.right)})"
    if isinstance(n, Paren):
        return "(" + ast_to_str(n.inner) + ")"
    raise TypeError(n)

# ========== SQL Validator Agent ==========

class SQLValidatorAgent:
    """
    Validates & normalizes SQL plans from the Planner.
    
    Features:
    - Schema validation (columns exist, tables joinable)
    - Function allow-list checks
    - Expression parsing to AST (no regex hacks)
    - Auto-rewrite raw divisions a / b → SAFE_DIV(a,b)
    - Optional clamping for rate-like metrics
    - Time alignment sanity checks
    - Returns normalized plan + validation report
    """

    def __init__(self, policy: Optional[ValidatorPolicy] = None):
        """
        Initialize the SQL Validator Agent.
        
        Args:
            policy: Validation policy configuration
        """
        # Default policy for SQLite with product_sales schema
        if policy is None:
            policy = ValidatorPolicy(
                allowed_functions=[
                    "SUM", "AVG", "COUNT", "COUNT_DISTINCT", "MIN", "MAX",
                    "SAFE_DIV", "RATIO_OF_SUMS", "PCT_OF_TOTAL", "CLAMP_0_1",
                    "TIME_BUCKET", "DATE_FILTER", "RUNNING_SUM", "LAG", "RANK",
                    "CONCAT", "UPPER", "LOWER"
                ],
                rate_name_hints=["discount", "rate", "ratio", "share", "pct", "percent", "margin"],
                auto_rewrite_division=True,
                require_clamp_for_rates=True,
                allow_avg_of_ratios=False,
                warehouse_dialect="sqlite",
                units={
                    "product_sales.total_price": "currency",
                    "product_sales.price_per_unit": "currency", 
                    "product_sales.quantity": "count",
                    "product_sales.discount_percent": "percent"
                }
            )
        
        self.policy = policy
        logger.info("SQLValidatorAgent initialized with SQLite policy")
    
    def _update_policy_with_semantic_info(self, semantic_schema):
        """Update validation policy with semantic schema information."""
        try:
            # Extract semantic units and constraints for better validation
            semantic_units = {}
            rate_keywords = set(self.policy.rate_name_hints)
            
            for table in semantic_schema.tables:
                for col in table.columns:
                    col_key = f"{table.name}.{col.name}"
                    
                    # Add unit information
                    if col.unit:
                        semantic_units[col_key] = col.unit
                        
                    # Detect rate-like columns from business meaning
                    if col.unit == "percentage" or any(word in col.business_meaning.lower() 
                                                    for word in ["rate", "ratio", "percent", "discount", "share"]):
                        rate_keywords.add(col.name)
            
            # Update policy with semantic information
            self.policy.units.update(semantic_units)
            self.policy.rate_name_hints = list(rate_keywords)
            
            logger.info(f"Updated validation policy with {len(semantic_units)} semantic units")
            
        except Exception as e:
            logger.warning(f"Could not update policy with semantic info: {e}")

    def validate_plan(self, plan: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], ValidationReport]:
        """
        Validate and normalize a SQL execution plan.
        
        Args:
            plan: SQL execution plan from planner
            schema: Database schema information (may include semantic_schema)
            
        Returns:
            Tuple of (normalized_plan, validation_report)
        """
        try:
            # Check if semantic schema is available for enhanced validation
            if schema.get("has_semantic") and schema.get("semantic_schema"):
                logger.info("🧠 Validating SQL plan with semantic schema context")
                # Use semantic schema for enhanced validation rules
                self._update_policy_with_semantic_info(schema["semantic_schema"])
            else:
                logger.info("📊 Validating SQL plan with basic schema")
            
            report = ValidationReport(ok=True)

            # 1) Required shape
            self._assert_shape(plan, report)

            # 2) Collect allowed columns & join graph
            allowed_cols = self._collect_allowed_columns(schema)
            join_graph = self._build_join_graph(schema)

            # 3) Validate group_by & filters column references
            self._validate_group_by(plan, allowed_cols, schema, join_graph, report)
            self._validate_filters(plan, allowed_cols, report)

            # 4) Validate & normalize expressions (parse → checks → rewrites)
            normalized_exprs = self._validate_and_normalize_expressions(plan, allowed_cols, schema, report)

            # 5) Validate time block presence/alignment
            self._validate_time(plan, schema, report)

            # Finalize
            normalized_plan = dict(plan)
            normalized_plan["expressions"] = normalized_exprs
            report.ok = len(report.errors()) == 0
            
            logger.info(f"Validation complete. OK: {report.ok}, Issues: {len(report.issues)}")
            return normalized_plan, report
            
        except Exception as e:
            logger.error(f"Error in plan validation: {e}")
            report = ValidationReport(ok=False)
            report.add("ERROR", "VALIDATION_FAILURE", f"Validation failed: {e}")
            return plan, report

    def _validate_and_normalize_expressions(self, plan: Dict[str, Any], allowed_cols: set, 
                                          schema: Dict[str, Any], report: ValidationReport) -> Dict[str, str]:
        """Validate and normalize all expressions in the plan."""
        normalized_exprs: Dict[str, str] = {}
        
        for metric_name, expr in plan.get("expressions", {}).items():
            try:
                ast = Parser(tokenize(expr)).parse()
            except Exception as e:
                report.add("ERROR", "PARSE_ERROR",
                          f"Cannot parse expression '{metric_name}': {e}", path=f"expressions.{metric_name}")
                continue

            # 4a) Ensure only allowed functions
            bad_funcs = self._find_disallowed_functions(ast)
            for fn in bad_funcs:
                report.add("ERROR", "DISALLOWED_FUNCTION",
                          f"Function '{fn}' not in allowed catalog.", path=f"expressions.{metric_name}")

            # 4b) Ensure identifiers (table.column) exist
            unknown_cols = self._find_unknown_columns(ast, allowed_cols)
            for fq in unknown_cols:
                report.add("ERROR", "UNKNOWN_COLUMN",
                          f"Unknown column '{fq}' in expression.", path=f"expressions.{metric_name}")

            # 4c) Auto-rewrite raw division to SAFE_DIV if policy allows
            if self.policy.auto_rewrite_division:
                before = ast_to_str(ast)
                ast = self._rewrite_divisions(ast, report, metric_name)
                after = ast_to_str(ast)
                if before != after:
                    report.add("FIX", "AUTO_SAFE_DIV",
                              "Rewrote '/' to SAFE_DIV(…, …).", path=f"expressions.{metric_name}")
            else:
                raw_div = self._has_raw_div(ast)
                if raw_div:
                    report.add("ERROR", "RAW_DIVISION",
                              "Raw '/' not allowed. Use SAFE_DIV(x,y).", path=f"expressions.{metric_name}")

            # 4d) Rate clamping (by name hints or explicit list)
            if self._looks_like_rate(metric_name, ast):
                if not self._is_top_level_clamped(ast):
                    ast = Func("CLAMP_0_1", [ast])
                    report.add("FIX", "AUTO_CLAMP",
                              "Wrapped rate-like metric with CLAMP_0_1.", path=f"expressions.{metric_name}")

            # 4e) AVG_OF_RATIOS policy
            if (not self.policy.allow_avg_of_ratios) and self._contains_func(ast, "AVG_OF_RATIOS"):
                report.add("WARN", "AVG_OF_RATIOS_USED",
                          "AVG_OF_RATIOS present; prefer RATIO_OF_SUMS unless explicitly required.",
                          path=f"expressions.{metric_name}")

            # 4f) Unit/type checks
            unit_issue = self._unit_check(ast, schema)
            if unit_issue:
                sev, msg = unit_issue
                report.add(sev, "UNIT_CHECK", msg, path=f"expressions.{metric_name}")

            normalized_exprs[metric_name] = ast_to_str(ast)

        return normalized_exprs

    # ----- Shape & schema helpers -----
    def _assert_shape(self, plan: Dict[str, Any], report: ValidationReport):
        if not isinstance(plan, dict):
            report.add("ERROR", "PLAN_NOT_OBJECT", "Plan must be a JSON object.")
            return
        if "expressions" not in plan or not isinstance(plan["expressions"], dict) or not plan["expressions"]:
            report.add("ERROR", "MISSING_EXPRESSIONS", "Plan must include non-empty 'expressions' dict.")
        # Note: time block is optional in our simplified schema
        if "time" in plan:
            time_value = plan["time"]
            if time_value is None:
                # Convert None to empty dict for compatibility
                plan["time"] = {}
            elif not isinstance(time_value, dict):
                report.add("ERROR", "BAD_TIME_BLOCK", "'time' must be a dict with 'col' and 'grain' or empty dict.")

    def _collect_allowed_columns(self, schema: Dict[str, Any]) -> set:
        """Extract all valid table.column combinations from schema."""
        allowed = set()
        tables = schema.get("tables", {})
        for tbl_name, tbl_info in tables.items():
            columns = tbl_info.get("columns", [])
            for col in columns:
                col_name = col.get("name") if isinstance(col, dict) else str(col)
                allowed.add(f"{tbl_name}.{col_name}")
        return allowed

    def _build_join_graph(self, schema: Dict[str, Any]) -> Dict[str, set]:
        """
        Build join graph for multi-table queries.
        For single-table scenarios (like product_sales), this returns empty graph.
        """
        g: Dict[str, set] = {}
        # For now, assume single table - can be extended for multi-table schemas
        return g

    def _reachable(self, g: Dict[str, set], src: str, dst: str) -> bool:
        if src == dst:
            return True
        seen = {src}
        stack = [src]
        while stack:
            t = stack.pop()
            for n in g.get(t, []):
                if n == dst:
                    return True
                if n not in seen:
                    seen.add(n); stack.append(n)
        return False

    # ----- Group by / filter validation -----
    def _validate_group_by(self, plan, allowed_cols, schema, join_graph, report: ValidationReport):
        for g in plan.get("group_by", []):
            if g not in allowed_cols:
                report.add("ERROR", "UNKNOWN_GROUP_BY", f"Unknown group_by column '{g}'.", path="group_by")

    def _validate_filters(self, plan, allowed_cols, report: ValidationReport):
        for i, f in enumerate(plan.get("filters", [])):
            sql = f.get("sql", "")
            # extract table.column tokens
            for t, c in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", sql):
                fq = f"{t}.{c}"
                if fq not in allowed_cols:
                    report.add("ERROR", "UNKNOWN_FILTER_COLUMN",
                              f"Unknown column in filter: '{fq}'", path=f"filters[{i}]")

    # ----- Expression analysis & rewrites -----
    def _find_disallowed_functions(self, ast: Node) -> List[str]:
        disallowed = []
        allowed = set(fn.upper() for fn in self.policy.allowed_functions)
        def visit(n: Node):
            if isinstance(n, Func):
                if n.name.upper() not in allowed:
                    disallowed.append(n.name)
        ast_walk(ast, visit)
        return disallowed

    def _find_unknown_columns(self, ast: Node, allowed_cols: set) -> List[str]:
        unknown = []
        def visit(n: Node):
            if isinstance(n, Ident):
                # Only flag if it looks like table.col and not in allowed set
                if "." in n.name and n.name not in allowed_cols:
                    unknown.append(n.name)
        ast_walk(ast, visit)
        return unknown

    def _has_raw_div(self, ast: Node) -> bool:
        found = False
        def visit(n: Node):
            nonlocal found
            if isinstance(n, BinOp) and n.op == "/":
                found = True
        ast_walk(ast, visit)
        return found

    def _rewrite_divisions(self, ast: Node, report: ValidationReport, metric_name: str) -> Node:
        def transform(n: Node) -> Node:
            if isinstance(n, BinOp) and n.op == "/":
                return Func("SAFE_DIV", [n.left, n.right])
            return n
        return ast_map(ast, transform)

    def _contains_func(self, ast: Node, fn_name: str) -> bool:
        target = fn_name.upper()
        hit = False
        def visit(n: Node):
            nonlocal hit
            if isinstance(n, Func) and n.name.upper() == target:
                hit = True
        ast_walk(ast, visit)
        return hit

    def _is_top_level_clamped(self, ast: Node) -> bool:
        return isinstance(ast, Func) and ast.name.upper() == "CLAMP_0_1"

    def _looks_like_rate(self, metric_name: str, ast: Node) -> bool:
        # 1) explicit list
        if metric_name in self.policy.explicit_rate_metrics:
            return True
        # 2) name hints
        hints = [h.lower() for h in self.policy.rate_name_hints]
        if any(h in metric_name.lower() for h in hints):
            return True
        # 3) expression contains ratio-ish constructs
        if (self._contains_func(ast, "RATIO_OF_SUMS") or 
            self._contains_func(ast, "AVG_OF_RATIOS") or 
            self._contains_func(ast, "SAFE_DIV")):
            return True
        return False

    # ----- Unit/type checker -----
    def _unit_check(self, ast: Node, schema: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """
        Basic unit checking for SQLite schema.
        Returns (severity, message) or None.
        """
        units_map = self.policy.units
        def infer(n: Node) -> Optional[str]:
            if isinstance(n, Number):
                return "number"
            if isinstance(n, Ident):
                return units_map.get(n.name, None)
            if isinstance(n, Paren):
                return infer(n.inner)
            if isinstance(n, BinOp):
                ul = infer(n.left); ur = infer(n.right)
                if n.op in ("+", "-"):
                    if ul and ur and ul != ur:
                        return "incompatible"
                    return ul or ur
                if n.op in ("*", "/"):
                    if ul == "currency" and ur == "currency" and n.op == "/":
                        return "unitless"
                    return None
            if isinstance(n, Func):
                name = n.name.upper()
                if name in ("SUM", "AVG", "MIN", "MAX"):
                    return infer(n.args[0]) if n.args else None
                if name in ("COUNT", "COUNT_DISTINCT"):
                    return "count"
                if name == "RATIO_OF_SUMS":
                    return "unitless"
                if name == "SAFE_DIV":
                    ul = infer(n.args[0]); ur = infer(n.args[1]) if len(n.args) > 1 else None
                    if ul == "currency" and ur == "currency":
                        return "unitless"
                    if ul == "currency" and ur == "count":
                        return "currency"
                    return None
                if name in ("PCT_OF_TOTAL", "PCT_OF_TOTAL_BY"):
                    return "unitless"
                if name == "CLAMP_0_1":
                    return "unitless"
                return None
            return None

        u = infer(ast)
        if u == "incompatible":
            return ("ERROR", "Incompatible units in + / - operation.")
        return None

    # ----- Time validation -----
    def _validate_time(self, plan: Dict[str, Any], schema: Dict[str, Any], report: ValidationReport):
        t = plan.get("time", {})
        if not t:  # time block is optional
            return
            
        tcol = t.get("col")
        grain = t.get("grain")
        if not tcol or not grain:
            report.add("ERROR", "TIME_INCOMPLETE", "'time.col' and 'time.grain' are required.", path="time")
            return

        # Check column exists
        tables = schema.get("tables", {})
        if "." in tcol:
            tbl, col = tcol.split(".", 1)
            if tbl not in tables:
                report.add("ERROR", "UNKNOWN_TIME_TABLE", f"Unknown table '{tbl}' in time column.", path="time")
            else:
                table_cols = [c.get("name") if isinstance(c, dict) else str(c) 
                             for c in tables[tbl].get("columns", [])]
                if col not in table_cols:
                    report.add("ERROR", "UNKNOWN_TIME_COLUMN", f"Unknown time column '{tcol}'.", path="time")

        # Check for window functions requiring time
        uses_window = False
        for expr in plan.get("expressions", {}).values():
            try:
                ast = Parser(tokenize(expr)).parse()
                if any(self._contains_func(ast, fn) for fn in 
                      ("LAG", "ROLLING_SUM_K", "ROLLING_AVG_K", "RUNNING_SUM")):
                    uses_window = True
            except Exception:
                continue
        
        if uses_window and not tcol:
            report.add("ERROR", "MISSING_TIME_FOR_WINDOW",
                      "Expressions use window functions but 'time.col' missing.", path="time")

    def get_validation_summary(self, report: ValidationReport) -> str:
        """Generate a human-readable validation summary."""
        summary_parts = []
        
        summary_parts.append(f"=== SQL Plan Validation Report ===")
        summary_parts.append(f"Status: {'✅ VALID' if report.ok else '❌ INVALID'}")
        summary_parts.append(f"Total Issues: {len(report.issues)}")
        
        if report.errors():
            summary_parts.append(f"\nErrors ({len(report.errors())}):")
            for err in report.errors():
                summary_parts.append(f"  ❌ {err.code}: {err.message}")
                if err.path:
                    summary_parts.append(f"     Path: {err.path}")
        
        if report.warnings():
            summary_parts.append(f"\nWarnings ({len(report.warnings())}):")
            for warn in report.warnings():
                summary_parts.append(f"  ⚠️  {warn.code}: {warn.message}")
                if warn.path:
                    summary_parts.append(f"     Path: {warn.path}")
        
        if report.fixes():
            summary_parts.append(f"\nAuto-Fixes Applied ({len(report.fixes())}):")
            for fix in report.fixes():
                summary_parts.append(f"  🔧 {fix.code}: {fix.message}")
                if fix.path:
                    summary_parts.append(f"     Path: {fix.path}")
        
        summary_parts.append("=== End Report ===")
        
        return "\n".join(summary_parts)
