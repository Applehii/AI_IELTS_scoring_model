import json
import re


# ===============================
# JSON extraction
# ===============================
import json
import re


def _repair_common_llm_json_errors(text: str) -> str:
    # 1. Remove trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # 2. Fix unquoted multiline string values (VERY conservative)
    def fix_unquoted(match):
        key = match.group(1)
        value = match.group(2).strip()
        value = value.replace('"', '\\"')
        return f'"{key}": "{value}"'

    text = re.sub(
        r'"(\w+)"\s*:\s*\n\s*([A-Za-z][^\{\}\[\]]+?)(?=\n\s*")',
        fix_unquoted,
        text,
        flags=re.MULTILINE
    )

    return text

def _has_unclosed_string(text: str) -> bool:
    in_string = False
    escaped = False

    for ch in text:
        if ch == '"' and not escaped:
            in_string = not in_string
        escaped = (ch == '\\') and not escaped

    return in_string

def _strip_json_comments(text: str) -> str:
    """
    Remove // and /* */ comments from JSON-like text.
    Safe for LLM outputs that violate JSON spec.
    """
    # Remove // comments
    text = re.sub(r'//.*?(?=\n|$)', '', text)

    # Remove /* */ comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    return text




def extract_json(raw: str) -> dict:
    raw = raw.strip()

    # 1. Fast path
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2. Extract first JSON object by brace counting
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    brace_count = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            brace_count += 1
        elif raw[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                json_str = raw[start:i + 1]
                if _has_unclosed_string(json_str):
                    raise ValueError(
                        "LLM returned JSON with unclosed string literal.\n"
                        "This is unsafe to auto-repair.\n"
                        f"RAW JSON:\n{json_str}"
         )

                json_candidate = _strip_json_comments(json_str)

                # 3. Parse extracted JSON
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    repaired = _repair_common_llm_json_errors(json_candidate)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            "Invalid JSON from LLM after repair\n"
                            f"ERROR: {e}\n"
                            f"RAW JSON:\n{json_candidate}\n\n"
                            f"REPAIRED JSON:\n{repaired}"
                        )

    raise ValueError("Unclosed JSON object in LLM response")




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