import requests
import json

class LLMClient:
    def __init__(self, model_name="llama3.1"):
        self.model = model_name
        self.url = "http://localhost:11434/api/chat"

    def ask(self, system_prompt: str, user_prompt: str):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }

        r = requests.post(self.url, json=payload)
        content = r.json()["message"]["content"]

        # Parse JSON from LLM output
        return json.loads(content)
