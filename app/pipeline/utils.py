import json
import re
import base64
def extract_json(text: str):
    """
    Safely extract the first JSON object from a model response.
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM did not return valid JSON.")
    return json.loads(match.group(0))

def image_to_base64(path: str):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
