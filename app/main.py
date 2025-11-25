from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from app.whisper_transcriber import Transcriber
from app.rag_manager import RAGManager
from app.llm_client import LLMClient
from app.vision_client import VisionClient
import tempfile
import json
import re

app = FastAPI()

transcriber = Transcriber("medium")
rag = RAGManager()
llm = LLMClient("llama3.1")
vision = VisionClient("qwen3-vl:8b")


# ---------------------- UTILS ----------------------
def extract_json(text: str):
    """Trích JSON trong response model (phòng trường hợp nó trả chữ thừa)."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM did not return JSON")
    return json.loads(match.group(0))


def call_prompt(prompt_path: str, question: str):
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    raw = llm.ask(system_prompt, f"Question:\n{question}")
    return extract_json(raw)


# ---------------------- SPEAKING ----------------------
@app.post("/speaking/score")
async def score_speaking(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    # 1. Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 2. Audio → text
    transcript = transcriber.transcribe(tmp_path)

    # 3. RAG lấy rubric
    rag_results = rag.retrieve(
        f"IELTS Speaking {question}",
        top_k=8,
        where={"type": "speaking_rubric"}
    )

    context_docs = rag_results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(context_docs)

    # 4. Prompt
    with open("app/prompts/speaking.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    user_prompt = f"""
Context (IELTS Rubric + Samples):
{context}

Question:
{question}

Transcript:
{transcript}
"""

    # 5. Gọi LLM
    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    # 6. Thêm transcript cho Spring Boot
    result["transcript"] = transcript
    return result


# ---------------------- WRITING (COMMON) ----------------------

class WritingScoreResponse(BaseModel):
    scores: dict
    overall: float
    feedback: dict
    task: str
    subtype: str | None = None


# --- Detect Task (1/2) ---
def detect_task(question: str):
    return call_prompt("app/prompts/classify_task.txt", question)


# --- Detect Task 2 subtype ---
def detect_task2_type(question: str):
    return call_prompt("app/prompts/classify_task2.txt", question)


# --- Build RAG context ---
def build_writing_context(question: str, task: str, subtype: str | None):
    docs: list[str] = []

    # 1. Rubric chung
    results = rag.retrieve(
        "IELTS Writing band descriptors",
        top_k=8,
        where={"type": "writing_rubric"}
    )
    docs.extend(results.get("documents", [[]])[0])

    # 2. Sample theo nội dung câu hỏi
    samples = rag.retrieve(
        f"IELTS Writing sample {question}",
        top_k=6,
        where={"type": "writing_sample"}
    )
    docs.extend(samples.get("documents", [[]])[0])

    # 3. Guide theo Task
    if task == "1":
        guides = rag.retrieve(
            question,
            top_k=6,
            where={"type": "writing_task1_guide"}
        )
        docs.extend(guides.get("documents", [[]])[0])

    if task == "2":
        # Guide chung task 2
        guides = rag.retrieve(
            question,
            top_k=6,
            where={"type": "writing_task2_guide"}
        )
        docs.extend(guides.get("documents", [[]])[0])

        # Guide theo subtype
        if subtype:
            sub_guides = rag.retrieve(
                subtype,
                top_k=4,
                where={"type": "writing_task2_guide"}
            )
            docs.extend(sub_guides.get("documents", [[]])[0])

    return "\n\n---\n\n".join(docs)


# --- MAIN ENDPOINT: Writing (Task auto detect, chủ yếu cho Task 2) ---
@app.post("/writing/score", response_model=WritingScoreResponse)
async def score_writing(
    question: str = Form(...),
    answer: str = Form(...)
):
    # 1. Detect Task
    task_info = detect_task(question)
    task = task_info["task"]

    # 2. Detect subtype nếu là Task 2
    subtype = None
    if task == "2":
        subtype_info = detect_task2_type(question)
        subtype = subtype_info["type"]

    # 3. Build context từ RAG
    context = build_writing_context(question, task, subtype)

    # 4. Prompt chấm điểm
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

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    return WritingScoreResponse(
        scores=result["scores"],
        overall=result["overall"],
        feedback=result["feedback"],
        task=task,
        subtype=subtype
    )


# ---------------------- WRITING TASK 1 + VISION ----------------------

@app.post("/writing/score-task1")
async def score_writing_task1(
    question: str = Form(...),
    answer: str = Form(...),
    chart: UploadFile = File(...)
):
    """
    Chấm riêng Task 1: FE gửi luôn đề + bài viết + ảnh biểu đồ.
    Dùng Vision (qwen2-vl) để đọc chart, rồi llama3.1 để chấm.
    """

    # 1. Lưu ảnh chart tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await chart.read())
        img_path = tmp.name

    # 2. Vision model đọc chart → structured data
    chart_data = vision.describe_chart(img_path)

    # 3. Lấy context Task 1 từ RAG (rubric + guide Task 1)
    context = build_writing_context(question, task="1", subtype=None)

    # 4. Prompt chấm Task 1
    with open("app/prompts/writing_task1.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    user_prompt = f"""
[CHART_DATA_FROM_VISION]
{json.dumps(chart_data, indent=2)}

[QUESTION]
{question}

[ANSWER]
{answer}

[CONTEXT_FROM_RAG]
{context}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    # Option: đính thêm chart_data cho FE debug nếu muốn
    result["chartData"] = chart_data
    result["task"] = "1"

    return result
