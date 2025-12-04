from app.rag_manager import RAGManager
from app.llm_client import LLMClient
import json
import re


# -----------------------------
# JSON extractor (safe)
# -----------------------------
def extract_json(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM did not return valid JSON.")
    return json.loads(match.group(0))


# -----------------------------
# Helper: call prompt files
# -----------------------------
def call_prompt(llm: LLMClient, prompt_path: str, question: str):
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    raw = llm.ask(system_prompt, f"Question:\n{question}")
    return extract_json(raw)


# ======================================================
#              MAIN PIPELINE CLASS
# ======================================================
class WritingPipeline:
    def __init__(self):
        self.rag = RAGManager()
        self.llm = LLMClient("llama3.1")

    # --------------------------
    # 1. DETECT TASK
    # --------------------------
    def detect_task(self, question: str):
        return call_prompt(
            self.llm,
            "app/prompts/classify_task.txt",
            question
        )

    # --------------------------
    # 2. DETECT SUBTYPE (TASK 2)
    # --------------------------
    def detect_task2_subtype(self, question: str):
        return call_prompt(
            self.llm,
            "app/prompts/classify_task2.txt",
            question
        )

    # --------------------------
    # 3. BUILD RAG CONTEXT
    # --------------------------
    def build_context(self, question: str, task: str, subtype: str | None):
        docs = []

        # Rubrics
        rubric_results = self.rag.retrieve(
            "IELTS Writing band descriptors",
            top_k=8,
            where={"type": "writing_rubric"}
        )
        docs.extend(rubric_results["documents"][0])

        # Sample essays (semantic similarity)
        sample_results = self.rag.retrieve(
            f"IELTS Writing sample {question}",
            top_k=6,
            where={"type": "writing_sample"}
        )
        docs.extend(sample_results["documents"][0])

        # Task 1 Guides
        if task == "1":
            guide_results = self.rag.retrieve(
                question,
                top_k=6,
                where={"type": "writing_task1_guide"}
            )
            docs.extend(guide_results["documents"][0])

        # Task 2 Guides
        if task == "2":
            general_guides = self.rag.retrieve(
                question,
                top_k=6,
                where={"type": "writing_task2_guide"}
            )
            docs.extend(general_guides["documents"][0])

            # Subtype guides
            if subtype:
                sub_guides = self.rag.retrieve(
                    subtype,
                    top_k=4,
                    where={"type": "writing_task2_guide"}
                )
                docs.extend(sub_guides["documents"][0])

        return "\n\n---\n\n".join(docs)

    # --------------------------
    # 4. SCORE WRITING (LLM)
    # --------------------------
    def score_writing(self, question: str, answer: str):
        # Detect task
        task_info = self.detect_task(question)
        task = task_info["task"]

        # Detect subtype if Task 2
        subtype = None
        if task == "2":
            subtype_info = self.detect_task2_subtype(question)
            subtype = subtype_info["type"]

        # RAG context
        context = self.build_context(question, task, subtype)

        # Load scoring prompt
        with open("app/prompts/writing.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        user_prompt = f"""
[DETECTED]
Task: {task}
Subtype: {subtype}

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
{answer}
"""

        # Call LLM
        raw = self.llm.ask(system_prompt, user_prompt)
        result = extract_json(raw)

        return {
            "task": task,
            "subtype": subtype,
            "scores": result["scores"],
            "overall": result["overall"],
            "feedback": result["feedback"]
        }
