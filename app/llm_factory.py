import os
from app.llm_remote import NvidiaLLM

class LLMFactory:
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        assert self.api_key, "Missing NVIDIA_API_KEY"

    def create(self) -> NvidiaLLM:
        return NvidiaLLM(api_key=self.api_key)
