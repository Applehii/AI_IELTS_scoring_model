import requests
import json

class LLMClient:
    def __init__(self, model_name="llama3.1"):
        self.model = model_name
        self.url = "http://127.0.0.1:11434/api/chat"

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }

        try:
            r = requests.post(self.url, json=payload)
            r.raise_for_status()
            # TRẢ VỀ RAW STRING, KHÔNG PARSE JSON Ở ĐÂY
            return r.json()["message"]["content"]
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""