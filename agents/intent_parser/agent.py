"""
Intent Parser Agent Module

This module contains the IntentParserAgent that converts natural language
queries into structured SQL intents using OpenAI GPT-3.5-turbo with
detailed schema context for accurate parsing.
"""

from openai import OpenAI
import json, logging, re
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class IntentParserAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = model
        # Compact, strict system prompt
        self.system_prompt = (
            "You are a deterministic SQL Intent Parser. Output STRICT JSON for an IR. Never invent tables/columns/values; use only names in schema_manifest. JSON only.\n"
            
            "Rules: DO NOT invent names and columns; capture every explicit value/number/time in params.filters or group_by.\n"
            "Single categorical mention (e.g., Advantage) ⇒ FILTER on that column (not group_by).\n"
            "CRITICAL: SaaS/vApp/pApp/Misc values MUST use IntersightConsumption column, NOT IntersightProduct.\n"
            "Use group_by only for 'by <column>/breakdown/each/across' or multiple values ('A vs B').\n"
            "Joins only via FK paths in schema_manifest; if none/ambiguous set needs_clarification=true.\n"
            "Relative periods => {'relative_period': '...'} and set date_column; do not guess dates.\n"
            "Numbers are numbers, booleans booleans.\n"
            "Entity MUST ALWAYS be the table that owns the metric/columns used. If params.column (or any filtered column) exists in exactly one table in schema_manifest, set entity to that table.\n"
            "DO NOT emit aggregations/custom fields. Use exactly: params.column, params.function, params.group_by (array), params.filters (array).\n"

            """            
            RANKING & DEFAULT METRIC
            - Words like top/bottom/most/fewest/highest/lowest => ranking task.
            - Default metric = COUNT of rows unless a numeric metric (e.g., revenue) is explicitly named.
            - If COUNT and a primary key is unknown, COUNT(*) if allowed; else COUNT of any grouped, non-null column.

            PERIODS & DIMENSIONS (schema-agnostic)
            - If the text implies a period (“each/per/by year/quarter/month”):
            • Prefer an actual period column whose name contains that token (case-insensitive).
            • Else, if exactly one date/timestamp column exists, set date_column to it and use YEAR(<date_column>) / QUARTER(<date_column>) / MONTH(<date_column>) as needed.
            - If the text names an entity like “customer/product/account/region”, prefer a dimension column whose name contains that token.

            GROUP vs FILTER
            - Single categorical mention without “by …” => FILTER (not group_by).
            - “X vs Y” for the same column => FILTER IN [X,Y]; only add group_by if the user asks for a breakdown.

            TOP-N PER GROUP
            - For “top … in each <period>”, after aggregation emit:
            params.post_aggregation.top_n_per_group = {
                "group_field": "<the chosen period dimension>",
                "order_by": "<metric alias or column>",
                "desc": true, "n": <N or 1>, "include_ties": true
            }

            JOINS & AMBIGUITY
            - Use only FK paths present in schema_manifest; if ambiguous, set needs_clarification=true.
            - If multiple plausible columns/entities match, pick the best and list alternatives in disambiguation; lower confidence.

            OUTPUT SHAPE (and nothing else)
            {
            "intent":"IR",
            "action":"select"|"filter"|"aggregate"|"join",
            "entity":"<primary table>",
            "params":{
                "column": "<metric column or null>",
                "function": "sum"|"count"|"avg"|"min"|"max"|null,
                "alias": "<metric_alias_or_null>",
                "group_by": ["<dim>", ...],
                "order_by": null,
                "desc": null,
                "limit": null,
                "filters": [ { "column":"<col>","operator":"="|...,"value":<scalar|obj> } ],
                "joins": [],
                "date_column": "<date col or null>",
                "target_dialect":"generic",
                "post_aggregation": {
                "top_n_per_group": {
                    "group_field":"<dim>",
                    "order_by":"<metric or alias>",
                    "desc":true, "n":1, "include_ties":true
                }
                }
            },
            "mentions":[],
            "unmapped_mentions":[],
            "needs_clarification":false,
            "disambiguation":{"columns":[],"entities":[]},
            "used_schema":{"tables":[],"columns":[]},
            "confidence":0.0-1.0
            }

            SELF-CHECK before answering
            1) Every explicit value/number/period captured as filter or group_by.
            2) Ranking present when “top/most/…” appears (order/limit or top_n_per_group).
            3) No invented names; types correct (numbers as numbers).
        """
        )

        logger.info(f"IntentParserAgent initialized with model={self.model}")

    

    # -------- static helpers (small, generic) --------
    @staticmethod
    def adapt_schema_for_llm(schema_info: Dict[str, Any]) -> Dict[str, Any]:
        tables = []
        for tname, tinfo in schema_info.get("tables", {}).items():
            cols = [{
                "name": c["name"],
                "type": c.get("type"),
                "is_pk": bool(c.get("primary_key")),
                "not_null": bool(c.get("not_null"))
            } for c in tinfo.get("columns", [])]
            fks = [{
                "from": fk["column"],
                "to_table": fk["references_table"],
                "to_column": fk["references_column"]
            } for fk in tinfo.get("foreign_keys", [])]
            pk = [c["name"] for c in tinfo.get("columns", []) if c.get("primary_key")]
            tables.append({"name": tname, "columns": cols, "primary_key": pk, "foreign_keys": fks})
        
        result = {"tables": tables}
        
        # Include semantic context if available
        if "semantic_context" in schema_info:
            result["semantic_context"] = schema_info["semantic_context"]
            
        return result

    @staticmethod
    def _tokens_from_name(name: str) -> List[str]:
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").split()
        return [p for p in parts if p]

    @staticmethod
    def _alias_phrases_for_column(col_name: str) -> List[str]:
        toks = IntentParserAgent._tokens_from_name(col_name)
        if not toks:
            return []
        phrases = []
        phrases.append(" ".join(toks).lower())
        if len(toks) >= 2:
            phrases.append(" ".join(toks[-2:]).lower())
        phrases.append(toks[-1].lower())
        drop = {"id", "ids", "key", "keys", "code", "codes"}
        if toks[-1].lower() in drop and len(toks) >= 2:
            phrases.append(" ".join(toks[:-1]).lower())
        dedup = []
        for p in phrases:
            if p and p not in dedup:
                dedup.append(p)
        return dedup[:5]

    @staticmethod
    def _column_alias_index(text: str, column_aliases: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        hits, low = [], text.lower()
        for col, aliases in column_aliases.items():
            for a in aliases:
                for m in re.finditer(rf"\b{re.escape(a)}\b", low):
                    hits.append({"column": col, "alias": a, "span": [m.start(), m.end()]})
        return hits

    
    @staticmethod
    def _detect_mentions_generic(user_query: str) -> List[Dict[str, Any]]:
        mentions: List[Dict[str, Any]] = []

        # Years (existing)
        for m in re.finditer(r"\b(19|20)\d{2}\b", user_query):
            mentions.append({"text": m.group(0), "type": "year", "span": [m.start(), m.end()]})

        # Rank with explicit N (existing)
        for m in re.finditer(r"\b(top|bottom)\s+(\d{1,3})\b", user_query, re.I):
            mentions.append({"text": m.group(0), "type": "rank", "n": int(m.group(2)), "span": [m.start(), m.end()]})

        # ✅ Enhanced Rank words WITHOUT N (enhanced)
        rank_patterns = [
            (r"\b(top|best|highest|maximum)\b", "DESC"),
            (r"\b(bottom|worst|lowest|minimum)\b", "ASC"),
            (r"\b(most)\s+(\w+)", "DESC"),  # "most bookings"
            (r"\b(fewest|least)\s+(\w+)", "ASC"),  # "fewest bookings"
        ]
        
        for pattern, direction in rank_patterns:
            for m in re.finditer(pattern, user_query, re.I):
                # avoid duplicating spans already captured as "top/bottom N"
                if not any(m.start() >= r["span"][0] and m.end() <= r["span"][1]
                        for r in mentions if r["type"] == "rank"):
                    mentions.append({
                        "text": m.group(0), 
                        "type": "rank", 
                        "n": None, 
                        "direction": direction,
                        "metric_hint": m.group(2) if len(m.groups()) > 1 else None,
                        "span": [m.start(), m.end()]
                    })

        # Comparators (existing)
        for m in re.finditer(r"([<>]=?|=)\s*\$?\d[\d,]*(\.\d+)?", user_query):
            mentions.append({"text": m.group(0), "type": "comparator", "span": [m.start(), m.end()]})

        # ✅ Enhanced Period hints with per-group detection (enhanced)
        per_group_patterns = [
            (r"\bfor\s+each\s+(\w+)", "for_each"),     # "for each year"
            (r"\bin\s+each\s+(\w+)", "in_each"),       # "in each year"
            (r"\bper\s+(\w+)", "per"),                 # "per year"
            (r"\bby\s+(\w+)", "by"),                   # "by year" (when used with ranking)
            (r"\bwithin\s+each\s+(\w+)", "within"),    # "within each year"
            (r"\bacross\s+(\w+)", "across"),           # "across years"
        ]
        
        for pattern, per_type in per_group_patterns:
            for m in re.finditer(pattern, user_query, re.I):
                mentions.append({
                    "text": m.group(0), 
                    "type": "per_group", 
                    "per_type": per_type,
                    "group_hint": m.group(1),
                    "span": [m.start(), m.end()]
                })
        
        # Simple period mentions (fallback)
        for m in re.finditer(r"\b(year|years|quarter|quarters|month|months)\b", user_query, re.I):
            # Only add if not already captured in per_group
            if not any(m.start() >= r["span"][0] and m.end() <= r["span"][1]
                    for r in mentions if r["type"] == "per_group"):
                mentions.append({"text": m.group(0), "type": "period", "span": [m.start(), m.end()]})

        # ✅ Entity hints (new)
        for m in re.finditer(r"\b(customer|customers|account|accounts|product|products|region|regions|country|countries|market|markets)\b",
                            user_query, re.I):
            mentions.append({"text": m.group(0), "type": "entity_hint", "span": [m.start(), m.end()]})

        return mentions


    @staticmethod
    def _build_context_generic(user_query: str, schema_manifest: Dict[str, Any]) -> Dict[str, Any]:
        column_aliases: Dict[str, List[str]] = {}
        for t in schema_manifest.get("tables", []):
            for c in t.get("columns", []):
                col = c.get("name")
                if col:
                    column_aliases[col] = IntentParserAgent._alias_phrases_for_column(col)
        column_hits = IntentParserAgent._column_alias_index(user_query, column_aliases)
        detected_mentions = IntentParserAgent._detect_mentions_generic(user_query)
        return {
            "schema_manifest": schema_manifest,
            "target_dialect": "generic",
            "user_query": user_query,
            "column_aliases": column_aliases,
            "column_hits": column_hits,
            "detected_mentions": detected_mentions
        }       

    @staticmethod
    def _safe_json_from_text(s: str) -> Any:
        s = (s or "").strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL).strip()
        try:
            return json.loads(s)
        except Exception:
            # fallback: longest {...}
            starts = [m.start() for m in re.finditer(r"\{", s)]
            ends = [m.start() for m in re.finditer(r"\}", s)]
            for i in range(len(starts)):
                for j in range(len(ends) - 1, -1, -1):
                    if ends[j] >= starts[i]:
                        cand = s[starts[i]:ends[j] + 1]
                        try:
                            obj = json.loads(cand)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            pass
            raise ValueError("Could not parse JSON from model output")

    @staticmethod
    def _validate_and_fill_defaults(ir: Dict[str, Any]) -> Dict[str, Any]:
        ir.setdefault("action", "select")
        ir.setdefault("entity", "unknown")
        p = ir.setdefault("params", {})
        p.setdefault("column", None)
        p.setdefault("function", None)
        p.setdefault("group_by", [])
        p.setdefault("order_by", None)
        p.setdefault("desc", None)
        p.setdefault("limit", None)
        p.setdefault("filters", [])
        p.setdefault("joins", [])
        p.setdefault("date_column", None)
        p.setdefault("target_dialect", "generic")
        p.setdefault("post_aggregation", None)
        p.setdefault("alias", None)
        ir.setdefault("mentions", [])
        ir.setdefault("unmapped_mentions", [])
        ir.setdefault("needs_clarification", False)
        ir.setdefault("disambiguation", {"columns": [], "entities": []})
        ir.setdefault("used_schema", {"tables": [], "columns": []})
        ir.setdefault("confidence", 0.5)
        return ir

    def _normalize_ir(self, ir: dict, schema_manifest: dict) -> dict:
        p = ir.setdefault("params", {})

        # 0) Enhanced: Analyze query complexity using detected mentions
        context = ir.get("context", {})
        detected_mentions = context.get("detected_mentions", [])
        
        # Classify query complexity
        ranking_mentions = [m for m in detected_mentions if m.get("type") == "rank"]
        per_group_mentions = [m for m in detected_mentions if m.get("type") == "per_group"]
        
        query_complexity = "simple_aggregation"  # default
        if ranking_mentions and per_group_mentions:
            query_complexity = "ranked_aggregation"  # Top N per group
        elif ranking_mentions:
            query_complexity = "simple_ranking"      # Just top N overall
        elif per_group_mentions:
            query_complexity = "grouped_aggregation" # Group by without ranking
        
        ir["query_complexity"] = query_complexity

        # 1) Map accidental shapes -> expected shape
        #    e.g., {"aggregations": {"ACTUAL_BOOKINGS": "total"}}
        aggs = ir.pop("aggregations", None) or p.pop("aggregations", None)
        if aggs and not p.get("column"):
            # pick single metric; if many, prefer known booking column
            metric, func = next(iter(aggs.items()))
            p["column"] = metric
            p["function"] = "sum" if func in ("sum", "total") else func

        # 2) Enhanced Action correction: use complexity classification
        text_hints = " ".join([
            (ir.get("original_query") or ""),
            json.dumps(p, ensure_ascii=False)
        ]).lower()
        
        # Set action based on complexity
        if query_complexity == "ranked_aggregation":
            ir["action"] = "ranked_aggregation"
        elif p.get("function") or any(k in text_hints for k in ["total", "sum", "count", "avg", "minimum", "maximum"]):
            ir["action"] = "aggregate"
        else:
            ir.setdefault("action", "aggregate" if p.get("function") else "select")

        # 3) GROUP BY default: only if explicitly present (empty array otherwise)
        p["group_by"] = p.get("group_by") or []

        # --- Generic fallbacks (schema-agnostic) ---
        # A) If ranking language and no function => default to COUNT
        if not p.get("function") and re.search(r"\b(top|most|highest|fewest|lowest|bottom)\b", text_hints):
            p["function"] = "count"

        # B) Enhanced: Build window function parameters for ranked_aggregation
        if query_complexity == "ranked_aggregation" and not p.get("post_aggregation"):
            # Extract ranking info
            ranking_info = {"direction": "DESC", "n": 1}  # defaults
            for mention in ranking_mentions:
                if mention.get("n"):
                    ranking_info["n"] = mention["n"]
                if mention.get("direction"):
                    ranking_info["direction"] = mention["direction"]
            
            # Extract partition columns from per_group mentions
            partition_columns = []
            for mention in per_group_mentions:
                group_hint = mention.get("group_hint", "")
                # Map hints to actual columns
                mapped_column = self._map_hint_to_column(group_hint, schema_manifest)
                if mapped_column:
                    partition_columns.append(mapped_column)
            
            # Build window function parameters
            metric_ref = p.get("alias") or p.get("column") or (partition_columns[0] if partition_columns else "CUSTOMER_NAME")
            p["post_aggregation"] = {
                "top_n_per_group": {
                    "partition_by": partition_columns,
                    "order_by": metric_ref,
                    "desc": ranking_info["direction"] == "DESC",
                    "n": ranking_info["n"],
                    "include_ties": True
                }
            }
        
        # Fallback: Old logic for compatibility
        elif (
            re.search(r"\b(top|most|highest)\b", text_hints) and
            re.search(r"\b(each|per|in each)\b", text_hints) and
            p.get("group_by") and
            not p.get("post_aggregation")
        ):
            metric_ref = p.get("alias") or p.get("column") or p["group_by"][0]
            p["post_aggregation"] = {
                "top_n_per_group": {
                    "group_field": p["group_by"][0],
                    "order_by": metric_ref,
                    "desc": True,
                    "n": 1,
                    "include_ties": True
                }
            }

        # 4) Entity inference from schema: pick the table that owns `column` or filter columns
        if not ir.get("entity") or ir["entity"] == "unknown":
            candidate_counts = {}
            cols_to_locate = [p.get("column")] if p.get("column") else []
            cols_to_locate += [f.get("column") for f in p.get("filters", []) if isinstance(f, dict)]
            cols_to_locate = [c for c in cols_to_locate if c]

            for t in schema_manifest.get("tables", []):
                tcols = {c["name"] for c in t.get("columns", [])}
                hit = any(c in tcols for c in cols_to_locate)
                if hit:
                    candidate_counts[t["name"]] = sum(c in tcols for c in cols_to_locate)

            if candidate_counts:
                ir["entity"] = max(candidate_counts, key=candidate_counts.get)

        # C) If COUNT metric chosen but no column set, pick a safe target
        if p.get("function") == "count" and not p.get("column"):
            if p.get("group_by"):
                p["column"] = p["group_by"][0]
            elif ir.get("entity"):
                for t in schema_manifest.get("tables", []):
                    if t.get("name") == ir["entity"]:
                        cols = [c["name"] for c in t.get("columns", [])]
                        if cols:
                            p["column"] = cols[0]
                        break

        # 5) Final shape hygiene
        p.setdefault("order_by", None)
        p.setdefault("desc", None)
        p.setdefault("limit", None)
        p.setdefault("filters", [])
        p.setdefault("joins", [])
        p.setdefault("date_column", None)
        p.setdefault("post_aggregation", p.get("post_aggregation", None))
        p.setdefault("alias", p.get("alias", None))
        ir.setdefault("mentions", [])
        ir.setdefault("unmapped_mentions", [])
        ir.setdefault("needs_clarification", False)
        ir.setdefault("disambiguation", {"columns": [], "entities": []})
        ir.setdefault("used_schema", {"tables": [], "columns": []})
        ir.setdefault("confidence", ir.get("confidence", 0.7))
        return ir

    def _map_hint_to_column(self, hint: str, schema_manifest: dict) -> str:
        """Map natural language hints to actual column names"""
        hint_mappings = {
            'year': 'YEAR',
            'years': 'YEAR', 
            'quarter': 'QUARTER',
            'quarters': 'QUARTER',
            'customer': 'CUSTOMER_NAME',
            'customers': 'CUSTOMER_NAME',
            'client': 'CUSTOMER_NAME',
            'clients': 'CUSTOMER_NAME',
            'country': 'COUNTRY',
            'region': 'COUNTRY',  # Fallback if no region column
            'tier': 'IntersightLicenseTier',
            'license': 'IntersightLicenseTier',
            'consumption': 'IntersightConsumption',
            'service': 'IntersightConsumption',  # Map "service" to consumption
            'saas': 'IntersightConsumption',     # Map SaaS values to consumption
            'product': 'IntersightProduct',
            'bookings': 'ACTUAL_BOOKINGS',
            'booking': 'ACTUAL_BOOKINGS',
            'revenue': 'ACTUAL_BOOKINGS',  # Fallback to bookings
            'price': 'LIST_PRICE',
            'quantity': 'QUANTITY'
        }
        
        # Direct mapping
        if hint.lower() in hint_mappings:
            return hint_mappings[hint.lower()]
        
        # Fuzzy matching with schema columns
        for table in schema_manifest.get("tables", []):
            for col in table.get("columns", []):
                col_name = col["name"]
                # Exact match
                if hint.upper() == col_name:
                    return col_name
                # Partial match
                if hint.lower() in col_name.lower() or col_name.lower() in hint.lower():
                    return col_name
        
        return None  # No mapping found

    def simplify_question(self, user_question: str, schema_info: Dict[str, Any], model: str = "gpt-3.5-turbo") -> str:
        # Get available columns for context
        available_columns = []
        for table_name, table_info in schema_info.get("tables", {}).items():
            for col in table_info.get("columns", []):
                available_columns.append(col["name"])
        
        columns_context = "Available columns: " + ", ".join(available_columns[:10])  # Show first 10

        SYSTEM_PROMPT = f"""You rewrite analytics questions into ONE clear plain-English sentence.
        
        {columns_context}
        
        Constraints:
        - No SQL. No JSON. No code. One sentence only.
        - Default “most/top” to COUNT of records unless the user explicitly says amount/revenue.
        - If filters appear (e.g., X = Y), say “among rows where X = Y”.
        - If a time grain like year/quarter/month is implied, say “For each <period>…”.
        - If the query implies a per-group winner, say “Include ties for first place”.
        - If the period is year, end with “Sort by year ascending.”

        Example:
        User: Show the bookings of top most customer with higher bookings for pApp in each year
        You: For each year, among rows where IntersightConsumption = 'pApp', find the customer with the most bookings. Return the year, customer name, and the number of bookings. Include ties for first place and sort by year ascending.
        """
        resp = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_question.strip()}
            ]
        )
        return (resp.choices[0].message.content or "").strip()


    # -------- main entrypoint --------
    def parse(self, user_query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Parsing query: {user_query}")
            
            # DEBUG: Check what's in schema_info
            logger.info(f"Schema_info keys: {list(schema_info.keys())}")
            if 'semantic_context' in schema_info:
                logger.info(f"✅ Found semantic_context in schema_info")
                semantic_context = schema_info['semantic_context']
                logger.info(f"Semantic context type: {type(semantic_context)}")
                if isinstance(semantic_context, dict) and 'tables' in semantic_context:
                    logger.info(f"Semantic tables: {[t.get('name', 'unknown') for t in semantic_context.get('tables', [])]}")
            else:
                logger.info(f"❌ No semantic_context found in schema_info")

            # 1) Normalize schema to manifest
            try:
                schema_manifest = self.adapt_schema_for_llm(schema_info)
            except Exception:
                schema_manifest = schema_info  # already in desired shape

            # 2) Build generic context
            context = self._build_context_generic(user_query, schema_manifest)

            # 3) Call model - semantic context is already in the context object
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}
            ]
            kwargs = dict(model=self.model, messages=messages, temperature=0, top_p=1, max_tokens=900)
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

            resp = self.client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or "{}"

            # 4) Parse & normalize output
            ir = self._safe_json_from_text(raw)
            # Pass context with detected mentions to _normalize_ir
            ir["context"] = context
            ir = self._normalize_ir(ir, schema_manifest)
            ir = self._validate_and_fill_defaults(ir)
            logger.info(f"Successfully parsed intent: {ir}")
            return ir

        except Exception as e:
            logger.exception(f"Error parsing intent: {e}")
            return {
                "action": "select",
                "entity": "unknown",
                "params": {
                    "column": None, "function": None, "group_by": [],
                    "order_by": None, "desc": None, "limit": None,
                    "filters": [], "joins": [], "date_column": None,
                    "target_dialect": "generic"
                },
                "mentions": [], "unmapped_mentions": [],
                "needs_clarification": False,
                "disambiguation": {"columns": [], "entities": []},
                "used_schema": {"tables": [], "columns": []},
                "confidence": 0.1
            }





    
    

########################################################
