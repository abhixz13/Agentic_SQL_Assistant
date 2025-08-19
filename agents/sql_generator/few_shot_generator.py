"""
Few-Shot Example Generator for SQL queries using OpenAI GPT-4o-mini

This module generates complex SQL examples with thinking processes that can be
injected as few-shot examples in the SQL generator agent's prompts.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv
from agents.schema_loader.semantic_schema import SemanticSchemaManager

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class FewShotExampleGenerator:
    """
    Generates complex SQL examples with thinking processes using GPT-4o-mini.
    
    Features:
    - Schema-aware example generation
    - Complex query patterns (JOINs, subqueries, window functions)
    - Step-by-step thinking process documentation
    - JSON output for easy integration
    """
    
    def __init__(self, examples_dir: str = "data/few_shot_examples"):
        """
        Initialize the few-shot example generator.
        
        Args:
            examples_dir: Directory to store generated examples
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.examples_dir = Path(examples_dir)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.model = "gpt-4o-mini"  # Using 4o-mini for cost efficiency
        
        # Initialize semantic schema manager
        self.semantic_manager = SemanticSchemaManager("agents/schema_loader/config/semantic_schemas")
        
        logger.info(f"FewShotExampleGenerator initialized with model: {self.model}")
    
    def generate_examples(self, schema_info: Dict, query_intent: Optional[Dict] = None) -> Dict:
        """
        Generate few-shot examples based on database schema and query intent.
        
        Args:
            schema_info: Database schema information
            query_intent: Optional current query intent for context-aware examples
            
        Returns:
            Dict containing generated examples with thinking processes
        """
        try:
            # Try to get semantic schema context, fall back to basic schema
            schema_context = self._get_schema_context(schema_info)
            intent_context = self._create_intent_context(query_intent) if query_intent else ""
            
            system_prompt = """You are an expert SQL educator and query optimization specialist. 
            Your task is to generate complex, realistic SQL examples that demonstrate advanced query patterns 
            and thinking processes for a given database schema.

            Generate examples that show:
            1. Complex analytical queries (JOINs, subqueries, window functions)
            2. Step-by-step thinking process
            3. Query breakdown and planning
            4. Schema relationship understanding
            5. Performance considerations

            Return ONLY valid JSON in the specified format."""
            
            user_prompt = f"""
            Given this database schema:
            {schema_context}

            {intent_context}

            Generate 4-5 complex SQL examples that demonstrate advanced query patterns.
            For each example, include:
            1. Natural language question
            2. Thinking process (how to approach the problem)
            3. Query breakdown (steps to build the SQL)
            4. Final SQL query
            5. Explanation of the approach

            Return in this exact JSON format:
            {{
                "schema_hash": "<hash_of_schema>",
                "generated_at": "<timestamp>",
                "examples": [
                    {{
                        "id": "example_1",
                        "question": "<natural language question>",
                        "thinking_process": [
                            "Step 1: <thought>",
                            "Step 2: <thought>",
                            "..."
                        ],
                        "query_breakdown": [
                            "1. <breakdown step>",
                            "2. <breakdown step>",
                            "..."
                        ],
                        "sql": "<final SQL query>",
                        "explanation": "<why this approach was chosen>",
                        "complexity": "medium|high",
                        "patterns": ["<pattern1>", "<pattern2>"]
                    }}
                ]
            }}

            Focus on realistic business scenarios like:
            - Revenue analysis across multiple dimensions
            - Customer segmentation and behavior analysis
            - Time-series analysis with trends
            - Performance metrics and KPIs
            - Complex aggregations with filtering

            Ensure all table and column names exactly match the schema provided.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            # Parse and validate response
            content = response.choices[0].message.content.strip()
            examples_data = json.loads(self._extract_json(content))
            
            # Add metadata
            schema_hash = self._generate_schema_hash(schema_info)
            examples_data["schema_hash"] = schema_hash
            examples_data["generated_at"] = self._get_timestamp()
            
            # Save to file
            filename = f"examples_{schema_hash[:8]}.json"
            filepath = self.examples_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(examples_data, f, indent=2)
            
            logger.info(f"Generated {len(examples_data.get('examples', []))} examples, saved to {filepath}")
            return examples_data
            
        except Exception as e:
            logger.error(f"Error generating examples: {e}")
            return self._fallback_examples()
    
    def get_examples_for_schema(self, schema_info: Dict, query_intent: Optional[Dict] = None) -> Dict:
        """
        Get examples for a schema, generating if not cached.
        
        Args:
            schema_info: Database schema information
            query_intent: Optional current query intent for context-aware examples
            
        Returns:
            Dict containing examples or empty dict if failed
        """
        # Check if ANY examples already exist in the directory
        existing_files = list(self.examples_dir.glob("examples_*.json"))
        
        if existing_files:
            # Use the first available cached example file
            existing_file = existing_files[0]
            try:
                with open(existing_file, 'r') as f:
                    examples_data = json.load(f)
                logger.info(f"✅ Using cached examples from {existing_file.name}")
                return examples_data
            except Exception as e:
                logger.warning(f"Error loading cached examples from {existing_file}: {e}")
        
        # Only generate new examples if no cached examples exist
        logger.info("📝 No cached examples found, generating new few-shot examples...")
        return self.generate_examples(schema_info, query_intent)
    
    def format_examples_for_prompt(self, examples_data: Dict) -> str:
        """
        Format examples for injection into SQL generator prompt.
        
        Args:
            examples_data: Generated examples data
            
        Returns:
            Formatted string for prompt injection
        """
        if not examples_data or "examples" not in examples_data:
            return self._get_default_examples()
        
        formatted_examples = []
        
        for example in examples_data.get("examples", []):
            thinking = "\n".join([f"  {step}" for step in example.get("thinking_process", [])])
            breakdown = "\n".join([f"  {step}" for step in example.get("query_breakdown", [])])
            
            formatted = f"""
Example: {example.get('question', 'Unknown question')}

Thinking Process:
{thinking}

Query Breakdown:
{breakdown}

SQL:
{example.get('sql', 'SELECT * FROM table;')}

Explanation: {example.get('explanation', 'No explanation provided')}
"""
            formatted_examples.append(formatted)
        
        return "\n" + "="*50 + "\n".join(formatted_examples) + "\n" + "="*50
    
    def _get_schema_context(self, schema_info: Dict) -> str:
        """Get schema context, preferring combined semantic schema if available."""
        # Try to load semantic schema first
        semantic_schema = self.semantic_manager.load_schema("product_sales_semantic.json")
        
        if semantic_schema:
            logger.info("🧠 Using combined semantic + basic schema for few-shot example generation")
            return self.semantic_manager.get_combined_schema_context(semantic_schema, schema_info)
        else:
            logger.info("📊 Using basic schema for few-shot example generation")
            return self.semantic_manager._create_schema_summary(schema_info)
    

    
    def _create_intent_context(self, query_intent: Dict) -> str:
        """Create context from current query intent to guide example generation."""
        if not query_intent:
            return ""
        
        action = query_intent.get("action", "")
        entity = query_intent.get("entity", "")
        params = query_intent.get("params", {})
        
        context_parts = [
            f"CURRENT QUERY CONTEXT:",
            f"The user is asking for a '{action}' operation on '{entity}'."
        ]
        
        # Add parameter context
        if params:
            context_parts.append("Query parameters:")
            for key, value in params.items():
                if value:
                    context_parts.append(f"  - {key}: {value}")
        
        context_parts.extend([
            "",
            "Generate examples that include similar query patterns and demonstrate:",
            f"- How to handle '{action}' operations effectively",
            f"- Best practices for querying '{entity}' table",
            "- Related analytical patterns users might need",
            "- Progressive complexity from the current query type",
            ""
        ])
        
        return "\n".join(context_parts)
    
    def _generate_cache_key(self, schema_info: Dict, query_intent: Optional[Dict] = None) -> str:
        """Generate cache key including schema and intent context."""
        schema_str = json.dumps(schema_info, sort_keys=True)
        
        # Include intent action and entity for context-aware caching
        if query_intent:
            intent_context = f"{query_intent.get('action', '')}-{query_intent.get('entity', '')}"
        else:
            intent_context = "general"
        
        combined_str = f"{schema_str}-{intent_context}"
        return hashlib.md5(combined_str.encode()).hexdigest()[:8]
    
    def _generate_schema_hash(self, schema_info: Dict) -> str:
        """Generate hash for schema to use as cache key."""
        schema_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response."""
        import re
        # Remove code fences if present
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text
    
    def _fallback_examples(self) -> Dict:
        """Return fallback examples if generation fails."""
        return {
            "schema_hash": "fallback",
            "generated_at": self._get_timestamp(),
            "examples": [
                {
                    "id": "fallback_1",
                    "question": "Show revenue by region",
                    "thinking_process": [
                        "Step 1: Identify revenue column (total_price)",
                        "Step 2: Group by region dimension",
                        "Step 3: Sum revenue for each region"
                    ],
                    "query_breakdown": [
                        "1. SELECT region and SUM(total_price)",
                        "2. FROM product_sales table",
                        "3. GROUP BY region"
                    ],
                    "sql": "SELECT region, SUM(total_price) AS total_revenue FROM product_sales GROUP BY region;",
                    "explanation": "Simple aggregation grouped by categorical dimension",
                    "complexity": "medium",
                    "patterns": ["aggregation", "grouping"]
                }
            ]
        }
    
    def _get_default_examples(self) -> str:
        """Get default examples if none are available."""
        return """
Example: Revenue analysis by region
Thinking: Need to aggregate revenue (total_price) by region dimension
SQL: SELECT region, SUM(total_price) FROM product_sales GROUP BY region;

Example: Top customers by purchases  
Thinking: Aggregate total spending per customer, order by amount, limit results
SQL: SELECT customer_id, SUM(total_price) as total FROM product_sales GROUP BY customer_id ORDER BY total DESC LIMIT 5;
"""
