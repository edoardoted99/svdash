import json
from typing import Dict, List, Any

SYSTEM_PROMPT = """You are a Knowledge Graph Transformation Agent.
You receive a JSON-LD Schema.org knowledge graph and a user instruction.
You must return ONLY valid JSON-LD (no markdown, no explanation, just the JSON).
The output must be a valid Schema.org graph.
Preserve all existing data unless the instruction says otherwise.
Use only valid Schema.org types and properties.
"""

def build_messages(user_prompt: str, kg_jsonld: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Builds the messages list for the LLM chat completion.
    
    Args:
        user_prompt: The instruction from the user on how to transform the KG.
        kg_jsonld: The current Knowledge Graph in JSON-LD format.
        
    Returns:
        A list of message dictionaries (role, content).
    """
    # Serialize KG to string
    try:
        kg_str = json.dumps(kg_jsonld, indent=2)
    except (TypeError, ValueError):
        kg_str = str(kg_jsonld)

    user_content = f"""Current Knowledge Graph (JSON-LD):
{kg_str}

Instruction:
{user_prompt}

Output JSON-LD:
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
