from concurrent.futures import ThreadPoolExecutor

from app.llm_client import LLMClient
from app.vision_client import VisionClient
from app.pipeline import phases
from app.pipeline.rule_exec import apply_hard_caps


class WritingTask1Pipeline:
    def __init__(self):
        self.llm = LLMClient("llama3.1")
        self.vision = VisionClient("qwen3-vl:8b")

    def score(
        self,
        question: str,
        answer: str,
        chart_path: str,
        debug: bool = False
    ):
        """
        IELTS Writing Task 1 scoring pipeline (multi-phase)
        """

        # =====================
        # PHASE 0 – CHART
        # =====================
        chart_data = phases.phase0_chart(
            self.vision,
            chart_path
        )

        # =====================
        # PHASE 1 – PARSE ESSAY
        # =====================
        parsed_essay = phases.phase1_parse(
            self.llm,
            answer
        )

        # =====================
        # PHASE 2 – TASK ACHIEVEMENT (SYNC)
        # =====================
        ta = phases.phase2_ta(
            self.llm,
            chart_data,
            parsed_essay
        )

        # =====================
        # PHASE 3–4–5 – PARALLEL
        # =====================
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_cc = executor.submit(
                phases.phase3_cc,
                self.llm,
                parsed_essay
            )
            future_lr = executor.submit(
                phases.phase4_lr,
                self.llm,
                answer
            )
            future_gra = executor.submit(
                phases.phase5_gra,
                self.llm,
                parsed_essay
            )

            cc = future_cc.result()
            lr = future_lr.result()
            gra = future_gra.result()
        
        bands = {
             "TA": ta["band"],
             "CC": cc["band"],
             "LR": lr["band"],
             "GRA": gra["band"],
        }
        violations = ta.get("violations", {})

        capped_bands, overall_cap, applied_rules = apply_hard_caps(
             bands,
             violations
        ) 
        ta["band"] = capped_bands["TA"]
        cc["band"] = capped_bands["CC"]
        lr["band"] = capped_bands["LR"]
        gra["band"] = capped_bands["GRA"]


        # =====================
        # PHASE 6 – OVERALL BAND
        # =====================
        overall = phases.phase6_band(
            ta, cc, lr, gra, overall_cap=overall_cap
        )
        overall["hard_cap_reasons"] = applied_rules

        # =====================
        # PHASE 7 – FEEDBACK
        # =====================
        feedback = phases.phase7_feedback(
            self.llm,
            chart_data,
            answer,
            {
                "Task Achievement": ta,
                "Coherence & Cohesion": cc,
                "Lexical Resource": lr,
                "Grammar Range & Accuracy": gra,
                "Overall": overall,
            },
        )

        result = {
            "task": "IELTS Writing Task 1",
            "overall": overall,
            "bands": {
                "TA": ta,
                "CC": cc,
                "LR": lr,
                "GRA": gra,
            },
            "feedback": feedback,
        }

        if debug:
            result["debug"] = {
                "chart": chart_data,
                "parsed_essay": parsed_essay,
            }

        return result
