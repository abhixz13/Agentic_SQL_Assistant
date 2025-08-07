from langchain_community.llms import Mixtral

# Configure the Mixtral LLM for intent parsing/planning.
# Replace with your actual API key or environment variable.
llm = Mixtral(
    temperature=0.7,  # Controls creativity (lower = more deterministic)
    max_tokens=500,   # Maximum output length
) 