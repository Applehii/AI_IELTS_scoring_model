from concurrent.futures import ThreadPoolExecutor

from app.llm_remote import NvidiaLLM
from app.llm_factory import LLMFactory
from app.vision_client import VisionClient
from app.pipeline import phases
from app.pipeline.rule_exec import (
    apply_all_rules,
    finalize_score,
    apply_gra_ceiling,
)
import os
from dotenv import load_dotenv
load_dotenv()



class WritingTask1Pipeline:
    def __init__(self):
        self.llm = NvidiaLLM(api_key=os.getenv("NVIDIA_API_KEY"))
        self.vision = VisionClient()

    def score(self, question: str, answer: str, chart_path: str, debug: bool = False):
        # =====================
        # PHASE 0 – CHART UNDERSTANDING
        # =====================
        chart_data = phases.phase0_chart(self.vision, chart_path)

        # =====================
        # PHASE 1 – PARSE ESSAY STRUCTURE
        # =====================
        parsed_essay = phases.phase1_parse(self.llm, answer)

        # =====================
        # PHASE 2 – TASK ACHIEVEMENT (DETECTION ONLY)
        # =====================
        ta_output = phases.phase2_ta(self.llm, chart_data, parsed_essay)
        ta_band = ta_output["band"]

        ta = {
            "base_band": ta_band,
            "final_band": ta_band,
            "band": ta_band,
            "violations": ta_output.get("violations", {}),
            "applied_soft": []
        }

        # =====================
        # PHASE 3–5 – PARALLEL SCORING (CC, LR, GRA)
        # =====================
        factory = LLMFactory()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
               "CC": executor.submit(
                     phases.phase3_cc,
                     factory.create(),
                     parsed_essay
                ),
                "LR": executor.submit(
                      phases.phase4_lr,
                      factory.create(),
                      parsed_essay
             ),
                "GRA": executor.submit(
                       phases.phase5_gra,
                       factory.create(),
                       parsed_essay
                ),
       }
            cc = futures["CC"].result()
            lr = futures["LR"].result()
            gra = futures["GRA"].result()

        # =====================
        # PHASE 5.5 – RAW BANDS SNAPSHOT
        # =====================
        raw_bands = {
            "TA": ta["base_band"],
            "CC": cc["band"],
            "LR": lr["band"],
            "GRA": gra["band"],
        }

        violations = ta.get("violations", {})

        # =====================
        # PHASE 6 – RULE ENGINE (HARD + SOFT)
        # =====================
        capped_bands, overall_cap, applied_hard, applied_soft = apply_all_rules(raw_bands, violations)

        # =====================
        # PHASE 6.5 – FINAL OVERALL SCORING
        # =====================
        final_band, note = finalize_score(
            bands_after_rules=capped_bands,
            gra_violations=gra.get("violations", {}),
            overall_cap=overall_cap
        )

        # =====================
        # PHASE 7 – ATTACH FINAL BANDS (UNIFIED)
        # =====================
        gra_ceiled = apply_gra_ceiling(capped_bands["GRA"], gra.get("violations", {}))

        # GRA ceiling flag
        gra["ceiling_applied"] = gra_ceiled != gra["band"]
        gra["final_band"] = gra_ceiled
        gra["band"] = gra_ceiled

        # Update final bands for others
        cc["final_band"] = capped_bands["CC"]
        cc["band"] = capped_bands["CC"]

        lr["final_band"] = capped_bands["LR"]
        lr["band"] = capped_bands["LR"]

        ta["final_band"] = capped_bands["TA"]
        ta["band"] = capped_bands["TA"]
        ta["applied_soft"] = applied_soft

        overall = {
            "band": final_band,
            "note": note,
            "hard_caps": applied_hard
        }

        # =====================
        # PHASE 8 – FEEDBACK (READ-ONLY)
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
            soft_traces=applied_soft,
            hard_traces=applied_hard
        )

        # =====================
        # FINAL RESULT
        # =====================
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
                "chart_data": chart_data,
                "parsed_essay": parsed_essay,
                "violations": violations,
                "raw_bands": raw_bands,
                "bands_after_rules": capped_bands,
                "applied_hard": applied_hard,
                "applied_soft": applied_soft,
            }

        return result
