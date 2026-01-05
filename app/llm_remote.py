from openai import OpenAI
from app.base_llm import BaseLLM

class NvidiaLLM(BaseLLM):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing NVIDIA API key")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

    def ask(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        completion = self.client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=kwargs.get("temperature", 0.6),
            top_p=kwargs.get("top_p", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
            stream=False
        )

        msg = completion.choices[0].message

        return msg.content
