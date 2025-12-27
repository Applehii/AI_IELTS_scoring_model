from app.llm_client import LLMClient
from app.rag_manager import RAGManager
from app.pipeline.utils import (
    extract_json,
    extract_rubric,
    extract_summary,
    trim_context,
)
from app.pipeline import phases


class WritingTask2Pipeline:
    def __init__(self):
        self.llm = LLMClient("llama3.1")
        self.rag = RAGManager()

    # --------------------------------------------------
    # BUILD RAG CONTEXT (TASK 2 ONLY)
    # --------------------------------------------------
    def build_context(self, question: str) -> str:
        chunks = []

        # 1️⃣ Rubric (VERY SHORT)
        rubric = self.rag.retrieve(
            "IELTS Writing Task 2 band descriptors",
            top_k=2,
            where={"type": "writing_rubric_task2"}
        )
        for d in rubric["documents"]:
            chunks.append("RUBRIC:\n" + extract_rubric(d))

        # 2️⃣ Task 2 guide
        guide = self.rag.retrieve(
            question,
            top_k=1,
            where={"type": "writing_task2_guide"}
        )
        for d in guide["documents"]:
            chunks.append("GUIDE:\n" + extract_rubric(d))

        # 3️⃣ Sample summaries ONLY
        samples = self.rag.retrieve(
            question,
            top_k=2,
            where={"type": "writing_sample"}
        )
        for d in samples["documents"]:
            summary = extract_summary(d)
            if summary:
                chunks.append("SAMPLE SUMMARY:\n" + summary)

        return "\n\n---\n\n".join(trim_context(chunks))

    # --------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------
    def score(
        self,
        question: str,
        answer: str,
        debug: bool = False
    ):
        # =====================
        # BUILD CONTEXT
        # =====================
        context = self.build_context(question)

        # =====================
        # PHASE 1 – PARSE ESSAY
        # =====================
        parsed_essay = phases.phase1_parse(self.llm, answer)

        # =====================
        # PHASE 2 – TASK RESPONSE
        # =====================
        tr = phases.phase2_tr(
            self.llm,
            question,
            parsed_essay,
            context
        )

        # =====================
        # PHASE 3 – COHERENCE & COHESION
        # =====================
        cc = phases.phase3_cc(self.llm, parsed_essay)

        # =====================
        # PHASE 4 – LEXICAL RESOURCE
        # =====================
        lr = phases.phase4_lr(self.llm, answer)

        # =====================
        # PHASE 5 – GRAMMAR
        # =====================
        gra = phases.phase5_gra(self.llm, parsed_essay)

        # =====================
        # PHASE 6 – OVERALL BAND
        # =====================
        overall = phases.phase6_band(tr, cc, lr, gra)

        # =====================
        # PHASE 7 – FEEDBACK
        # =====================
        feedback = phases.phase7_feedback_task2(
            self.llm,
            question,
            answer,
            {
                "Task Response": tr,
                "Coherence & Cohesion": cc,
                "Lexical Resource": lr,
                "Grammar Range & Accuracy": gra,
                "Overall": overall,
            },
        )

        result = {
            "task": "IELTS Writing Task 2",
            "overall": overall,
            "bands": {
                "TR": tr,
                "CC": cc,
                "LR": lr,
                "GRA": gra,
            },
            "feedback": feedback,
        }

        if debug:
            result["debug"] = {
                "context": context,
                "parsed_essay": parsed_essay,
            }

        return result
