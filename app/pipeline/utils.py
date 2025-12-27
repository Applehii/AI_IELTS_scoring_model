import json
import re


# ===============================
# JSON extraction
# ===============================
def extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*?\}", raw)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    json_str = match.group()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}\nRAW:\n{json_str}")

# ===============================
# RAG helpers
# ===============================
def extract_rubric(doc: str, max_chars: int = 800) -> str:
    """
    Extract only rubric-relevant part to avoid payload bloat
    """
    if "[BAND DESCRIPTORS]" in doc:
        doc = doc.split("[BAND DESCRIPTORS]")[1]

    return doc.strip()[:max_chars]


def extract_summary(doc: str) -> str:
    """
    Extract short sample summary only (not full essay)
    """
    if "[SUMMARY]" in doc:
        return doc.split("[SUMMARY]")[1].split("[")[0].strip()
    return ""


def trim_context(chunks: list[str], max_chars: int = 6000) -> list[str]:
    """
    Hard cap context length (production safety)
    """
    result = []
    total = 0
    for c in chunks:
        if total + len(c) > max_chars:
            break
        result.append(c)
        total += len(c)
    return result
