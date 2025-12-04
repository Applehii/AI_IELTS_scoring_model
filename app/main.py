from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import json
import re

from app.whisper_transcriber import Transcriber
from app.llm_client import LLMClient
from app.vision_client import VisionClient

from app.pipeline.writing import WritingPipeline
from app.pipeline.writing_task1 import WritingTask1Pipeline


app = FastAPI()

# ===== Global Services =====
transcriber = Transcriber("medium")
llm = LLMClient("llama3.1")
vision = VisionClient("qwen3-vl:8b")

# ===== Pipelines =====
writing_pipeline = WritingPipeline()
writing_task1_pipeline = WritingTask1Pipeline()


# -----------------------------
# JSON extractor (safe)
# -----------------------------
def extract_json(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM did not return valid JSON")
    return json.loads(match.group(0))


# ============================================================
#                     SPEAKING SCORING
# ============================================================
@app.post("/speaking/score")
async def score_speaking(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 1. Speech → text
    transcript = transcriber.transcribe(tmp_path)

    # 2. Retrieve speaking rubric
    rag_results = writing_pipeline.rag.retrieve(
        f"IELTS Speaking {question}",
        top_k=8,
        where={"type": "speaking_rubric"}
    )

    # FIX: use all documents (list), NOT documents[0]
    context_docs = rag_results["documents"]
    context = "\n\n---\n\n".join(context_docs)

    # Load prompt
    with open("app/prompts/speaking.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    user_prompt = f"""
Context:
{context}

Question:
{question}

Transcript:
{transcript}
"""

    # LLM evaluate
    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)
    result["transcript"] = transcript
    return result


# ============================================================
#                     WRITING TASK 2 SCORING
# ============================================================
@app.post("/writing/score")
async def score_writing(
    question: str = Form(...),
    answer: str = Form(...)
):
    """Task auto-detect → Task 2 → scoring pipeline"""
    return writing_pipeline.score_writing(question, answer)


# ============================================================
#                     WRITING TASK 1 SCORING
# ============================================================
@app.post("/writing/score-task1")
async def score_writing_task1(
    question: str = Form(...),
    answer: str = Form(...),
    chart: UploadFile = File(...)
):
    """Task 1 scoring using Vision pipeline."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await chart.read())
        img_path = tmp.name

    return writing_task1_pipeline.score(
        question=question,
        answer=answer,
        chart_path=img_path
    )
