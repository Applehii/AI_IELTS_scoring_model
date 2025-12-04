import json
from app.rag_manager import RAGManager
from app.llm_client import LLMClient
from app.vision_client import VisionClient
from app.pipeline.utils import extract_json


class WritingTask1Pipeline:
    def __init__(self):
        self.rag = RAGManager()
        self.llm = LLMClient("llama3.1")
        self.vision = VisionClient("qwen3-vl:8b")

    def score(self, question: str, answer: str, chart_path: str):

        # 1. Vision đọc ảnh
        chart_data = self.vision.describe_chart(chart_path)

        # 2. Lấy rubric (list)
        rubric = self.rag.retrieve(
            "IELTS Writing band descriptors",
            top_k=8,
            where={"type": "writing_rubric"}
        )["documents"]

        # 3. Lấy Task 1 guide (list)
        guides = self.rag.retrieve(
            question,
            top_k=6,
            where={"type": "writing_task1_guide"}
        )["documents"]

        # 4. Lấy sample Task 1 (list)
        samples = self.rag.retrieve(
            f"IELTS Writing sample {question}",
            top_k=6,
            where={"type": "writing_sample"}
        )["documents"]

        # 5. Gộp tất cả
        all_docs = rubric + guides + samples
        context = "\n\n---\n\n".join(all_docs)

        # 6. Load prompt
        with open("app/prompts/writing_task1.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        user_prompt = f"""
[CHART_DATA]
{json.dumps(chart_data, indent=2)}

[QUESTION]
{question}

[ANSWER]
{answer}

[CONTEXT]
{context}
"""

        raw = self.llm.ask(system_prompt, user_prompt)
        if isinstance(raw, dict):
           raw = raw.get("response", json.dumps(raw))
        result = extract_json(raw)
        
        

        return {
            "task": "1",
            "chartData": chart_data,
            "scores": result["scores"],
            "overall": result["overall"],
            "feedback": result["feedback"]
        }
