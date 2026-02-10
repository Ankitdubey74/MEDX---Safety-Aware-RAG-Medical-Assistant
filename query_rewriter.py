from langchain_community.llms import Ollama

# ⚠️ Very small model
llm = Ollama(
    model="phi",
    temperature=0,
    num_ctx=256,
    keep_alive="0"
)

def rewrite_query(query: str) -> str:
    prompt = f"""
You are a query rewriting agent for a medical search engine.

TASK:
Rewrite the user's query into a clear, complete medical question.

RULES:
- Keep meaning same
- Expand abbreviations
- Do NOT add new medical facts
- Output ONLY rewritten query
- Max 1 sentence

User Query:
{query}

Rewritten Query:
"""
    try:
        rewritten = llm.invoke(prompt).strip()

        # Safety fallback
        if len(rewritten.split()) < len(query.split()):
            return query

        return rewritten

    except Exception:
        return query
