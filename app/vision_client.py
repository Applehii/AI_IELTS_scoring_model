import base64
import requests

class VisionClient:
    def __init__(self, model="qwen3-vl:8b"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def encode_image(self, image_path: str) -> str:
        """Convert image → base64 string"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def describe_chart(self, image_path: str):
        """Send base64-encoded image to Ollama Vision."""
        img_b64 = self.encode_image(image_path)

        payload = {
            "model": self.model,
            "prompt": "Describe this chart in JSON with keys: title, chartType, keyTrends, values.",
            "images": [img_b64],
            "stream": False
        }

        print("\n===== DEBUG: Sending to Ollama =====")
        print(f"POST {self.url}")
        print(f"Payload size: {len(img_b64)} base64 chars")

        res = requests.post(self.url, json=payload)

        print("\n===== DEBUG: Raw response =====")
        print(res.text)

        if res.status_code != 200:
            raise Exception(f"Ollama Vision error {res.status_code}: {res.text}")

        try:
            return res.json().get("response", "")
        except Exception:
            raise Exception("Failed to parse JSON response from Vision model")
