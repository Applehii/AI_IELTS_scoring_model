from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import json
import re

from app.whisper_transcriber import Transcriber
from app.llm_client import LLMClient
from app.vision_client import VisionClient
from app.pipeline.writing import WritingPipeline

app = FastAPI()

# ===== Global Services =====
transcriber = Transcriber("medium")
llm = LLMClient("llama3.1")
vision = VisionClient("qwen3-vl:8b")

# ===== Pipeline =====
writing_pipeline = WritingPipeline()


# ============================================================
# SPEAKING SCORING
# ============================================================
@app.post("/speaking/score")
async def score_speaking(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    transcript = transcriber.transcribe(tmp_path)

    rag_results = writing_pipeline.rag.retrieve(
        f"IELTS Speaking {question}",
        top_k=8,
        where={"type": "speaking_rubric"}
    )

    context = "\n\n---\n\n".join(rag_results["documents"])

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

    raw = llm.ask(system_prompt, user_prompt)
    result = json.loads(re.search(r"\{[\s\S]*\}", raw).group())
    result["transcript"] = transcript
    return result


# ============================================================
# WRITING SCORING (TASK 1 + TASK 2)
# ============================================================
@app.post("/writing/score")
async def score_writing(
    question: str = Form(...),
    answer: str = Form(...),
    chart: UploadFile | None = File(None)
):
    chart_path = None
    if chart:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(await chart.read())
            chart_path = tmp.name

    return writing_pipeline.score_writing(
        question=question,
        answer=answer,
        chart_path=chart_path
    )
