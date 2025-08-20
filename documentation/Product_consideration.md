Key consideration

1. What is a better fit here - An network of AI agents or simple Python script? 

IMO, a network of AI agents will do a far better job at making a higher agency SQL assistant. 

A script can be sufficient in the syste," 
- Always follow a single-shot execution pipeline (1. Pass: NL ---> Generate SQL ----> Execute to fetch data ----> Generate chart)
- Operates in a deterministic, rule-based fashion (e.g. " always create bar chart for group-by queries")
- Supports single-turn use cases (user asks, model answers, done)
- Doesn't need intermediate reasoning


- An agent makes sense when we need the AI to build some level of reasoning and reflection, resulting in the following:
1. Generate SQL
2. Reflect and Revise. 
3. We want the LLM to think about the best way to interpret the intent and break the complex query. 

