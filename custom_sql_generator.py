#!/usr/bin/env python3
"""
Custom SQL generator for product_sales dataset
"""

def generate_custom_sql(intent: dict) -> str:
    """Generate SQL based on intent for product_sales dataset"""
    
    action = intent.get("action", "select")
    entity = intent.get("entity", "product_sales")
    params = intent.get("params", {})
    
    if action == "aggregate":
        function = params.get("function", "sum")
        column = params.get("column", "total_price")
        group_by = params.get("group_by")
        
        if group_by:
            return f"SELECT {group_by}, {function.upper()}({column}) as result FROM {entity} GROUP BY {group_by} ORDER BY result DESC"
        else:
            return f"SELECT {function.upper()}({column}) as result FROM {entity}"
    
    elif action == "top_customers":
        limit = params.get("limit", 3)
        group_by = params.get("group_by", "region")
        return f"""
        SELECT customer_name, {group_by}, SUM(total_price) as total_sales
        FROM {entity}
        GROUP BY customer_name, {group_by}
        ORDER BY total_sales DESC
        LIMIT {limit}
        """
    
    elif action == "select":
        limit = params.get("limit", 10)
        return f"SELECT * FROM {entity} LIMIT {limit}"
    
    else:
        return f"SELECT * FROM {entity} LIMIT 10"

def parse_query_to_sql(natural_query: str) -> str:
    """Parse natural language query to SQL"""
    query_lower = natural_query.lower()
    
    if "revenue" in query_lower and "region" in query_lower:
        return "SELECT region, SUM(total_price) as revenue FROM product_sales GROUP BY region ORDER BY revenue DESC"
    
    elif "revenue" in query_lower and "category" in query_lower:
        return "SELECT category, SUM(total_price) as revenue FROM product_sales GROUP BY category ORDER BY revenue DESC"
    
    elif "orders" in query_lower and "category" in query_lower:
        return "SELECT category, COUNT(*) as order_count FROM product_sales GROUP BY category ORDER BY order_count DESC"
    
    elif "top" in query_lower and "customer" in query_lower and "region" in query_lower:
        return """
        SELECT customer_name, region, SUM(total_price) as total_sales
        FROM product_sales
        GROUP BY customer_name, region
        ORDER BY total_sales DESC
        LIMIT 3
        """
    
    elif "describe" in query_lower or "show" in query_lower:
        return "SELECT * FROM product_sales LIMIT 10"
    
    elif "payment" in query_lower:
        return "SELECT payment_method, COUNT(*) as count FROM product_sales GROUP BY payment_method ORDER BY count DESC"
    
    elif "sales channel" in query_lower or "channel" in query_lower:
        return "SELECT sales_channel, SUM(total_price) as revenue FROM product_sales GROUP BY sales_channel ORDER BY revenue DESC"
    
    else:
        return "SELECT * FROM product_sales LIMIT 20" 