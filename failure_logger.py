import json
import datetime

LOG_FILE = "logs/model_failures.jsonl"

def log_model_failure(
    query: str,
    confidence: float,
    mode: str,
    used_tools=None
):
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query": query,
        "confidence": round(confidence, 3),
        "mode": mode,
        "used_tools": used_tools or []
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
