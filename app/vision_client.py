import requests
import base64

class VisionClient:
    def __init__(self, model="qwen2-vl:7b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def describe_chart(self, image_path):
        # encode ảnh base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.model,
            "prompt": "Describe this chart in structured JSON: main trends, comparisons, peaks, lows.",
            "images": [img_b64],
            "stream": False
        }

        res = requests.post(self.url, json=payload)
        txt = res.json()["response"]

        # Extract JSON
        import re, json
        match = re.search(r"\{[\s\S]*\}", txt)
        return json.loads(match.group(0)) if match else {"description": txt}
