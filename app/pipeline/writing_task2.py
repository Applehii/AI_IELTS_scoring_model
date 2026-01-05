from app.llm_client import LLMClient
from app.rag_manager import RAGManager
from app.pipeline import phases
from app.pipeline.rule_exec import (
    apply_all_rules,
    finalize_score_task2,
    apply_gra_ceiling,
)


class WritingTask2Pipeline:
    def __init__(self):
        self.llm = LLMClient("llama3.1")
        self.rag = RAGManager()
    # ==================================================
    # MAIN ENTRY
    # ==================================================
    def score(
        self,
        question: str,
        answer: str,
        debug: bool = False
    ):

        # =====================
        # PHASE 1 – PARSE ESSAY
        # =====================
        parsed_essay = phases.phase1_parse_task2(self.llm, question,answer)

        # =====================
        # PHASE 2 – TASK RESPONSE (DETECTION ONLY)
        # =====================
        tr_output = phases.phase2_tr(
            self.llm,
            question,
            parsed_essay,
            task_type=parsed_essay["task_type"]
        )

        tr_band = tr_output["band"]

        tr = {
            "base_band": tr_band,
            "final_band": tr_band,
            "band": tr_band,
            "violations": tr_output.get("violations", {})
        }

        # =====================
        # PHASE 3 – COHERENCE & COHESION
        # =====================
        cc = phases.phase3_cc(self.llm, parsed_essay)

        # =====================
        # PHASE 4 – LEXICAL RESOURCE
        # =====================
        lr = phases.phase4_lr(self.llm, parsed_essay)

        # =====================
        # PHASE 5 – GRAMMATICAL RANGE & ACCURACY
        # =====================
        gra = phases.phase5_gra(self.llm, parsed_essay)

        # =====================
        # PHASE 5.5 – RAW BANDS SNAPSHOT
        # =====================
        raw_bands = {
            "TR": tr["base_band"],
            "CC": cc["band"],
            "LR": lr["band"],
            "GRA": gra["band"],
        }

        # =====================
        # PHASE 6 – RULE ENGINE (HARD + SOFT)
        # =====================
        capped_bands, overall_cap, applied_hard, applied_soft = apply_all_rules(
            raw_bands,
            tr.get("violations", {})
        )

        # =====================
        # PHASE 6.5 – FINAL OVERALL SCORING
        # =====================
        final_band, note = finalize_score_task2(
            bands_after_rules=capped_bands,
            gra_violations=gra.get("violations", []),
            overall_cap=overall_cap
        )

        # =====================
        # PHASE 7 – ATTACH FINAL BANDS (UNIFIED)
        # =====================
        gra_ceiled = apply_gra_ceiling(
            capped_bands["GRA"],
            gra.get("violations", [])
        )

        for criterion, value in zip(
            [tr, cc, lr, gra],
            [
                capped_bands["TR"],
                capped_bands["CC"],
                capped_bands["LR"],
                gra_ceiled,
            ]
        ):
            criterion["final_band"] = value
            criterion["band"] = value

        tr["applied_soft"] = applied_soft

        overall = {
            "band": final_band,
            "note": note,
            "hard_caps": applied_hard
        }

        # =====================
        # PHASE 8 – FEEDBACK
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

        # =====================
        # FINAL RESULT
        # =====================
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
                "raw_bands": raw_bands,
                "bands_after_rules": capped_bands,
                "applied_hard": applied_hard,
                "applied_soft": applied_soft,
            }

        return result
