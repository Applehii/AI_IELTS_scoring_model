from app.pipeline.utils import extract_json
from app.pipeline.prompt_loader import load_prompt


# =====================================================
# PHASE 0 – CHART UNDERSTANDING
# =====================================================
def phase0_chart(vision, chart_path: str):
    return vision.describe_chart(chart_path)


# =====================================================
# PHASE 1 – PARSE ESSAY STRUCTURE
# =====================================================
def phase1_parse(llm, essay: str):
    system_prompt = load_prompt(
        "phase1_parse.txt",
        rubric_name=None
    )

    raw = llm.ask(system_prompt, essay)
    result = extract_json(raw)

    if not isinstance(result, dict):
        raise ValueError("phase1_parse: invalid JSON")

    return result


def phase1_parse_task2(llm, question: str, essay: str):
    system_prompt = load_prompt(
        "phase1_parse_task2.txt",
        rubric_name=None
    )

    user_prompt = f"""
[QUESTION]
{question}

[ESSAY]
{essay}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    if not isinstance(result, dict):
        raise ValueError("phase1_parse: invalid JSON")

    return result


# =====================================================
# PHASE 2 – TASK ACHIEVEMENT
# =====================================================
def phase2_ta(llm, chart_data, parsed_essay):
    system_prompt = load_prompt(
        "phase2_ta.txt",
        rubric_name="TA"
    )

    user_prompt = f"""
[CHART]
{chart_data}

[OVERVIEW]
{parsed_essay.get("overview")}

[BODY_PARAGRAPHS]
{parsed_essay.get("body_paragraphs")}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    _ensure_band(result, "TA")
    return result


def phase2_tr(llm, question: str, parsed_essay: dict, task_type: str, debug: bool = False):
    system_prompt = load_prompt(
        "phase2_tr.txt",
        rubric_name="TR"
    )

    essay_text = "\n".join(parsed_essay["sentences"])

    user_prompt = f"""
[QUESTION]
{question}

TASK TYPE (FIXED – DO NOT REINTERPRET):
{task_type}

You MUST evaluate Task Response STRICTLY according to this task type.
Do NOT apply requirements from other task types.

[ESSAY]
{essay_text}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    _ensure_band(result, "TR")

    if debug:
        return {
            **result,
            "_debug": {
                "task_type": task_type,
                "essay_preview": essay_text[:300],
                "raw_response": raw
            }
        }

    return result

# =====================================================
# PHASE 3 – COHERENCE & COHESION
# =====================================================
def phase3_cc(llm, parsed_essay):
    system_prompt = load_prompt(
        "phase3_cc.txt",
        rubric_name="CC"
    )

    user_prompt = f"""
[INTRODUCTION]
{parsed_essay.get("introduction")}

[BODY_PARAGRAPHS]
{parsed_essay.get("body_paragraphs")}

[CONCLUSION]
{parsed_essay.get("conclusion")}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    _ensure_band(result, "CC")
    return result


def phase4_lr(llm, parsed_essay):
    system_prompt = load_prompt(
        "phase4_lr.txt",
        rubric_name="LR"
    )

    essay_text = "\n".join(parsed_essay["sentences"])

    user_prompt = f"""
[SENTENCES]
{essay_text}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    _ensure_band(result, "LR")
    return result



# =====================================================
# PHASE 5 – GRAMMATICAL RANGE & ACCURACY
# =====================================================
def phase5_gra(llm, parsed_essay):
    system_prompt = load_prompt(
        "phase5_gra.txt",
        rubric_name="GRA"
    )
    sentences_text = "\n".join(parsed_essay["sentences"])
    user_prompt = f"""
[SENTENCES]
{sentences_text}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    _ensure_band(result, "GRA") 

    # Gán cấu trúc consistent
    gra = {
        "base_band": result["band"],
        "final_band": result["band"], 
        "band": result["band"],
        "strengths": result.get("strengths", []),
        "weaknesses": result.get("weaknesses", []),
        "violations": result.get("violations", {}) 
    }

    return gra


# =====================================================
# PHASE 6 – FINAL BAND CALCULATION (LLM-BASED)
# =====================================================
def phase6_band(ta, cc, lr, gra, overall_cap=None):
    avg = (
        ta["band"]
        + cc["band"]
        + lr["band"]
        + gra["band"]
    ) / 4

    frac = avg % 1
    if frac < 0.25:
        final = int(avg)
        rule = "Rounded DOWN (.00/.25)"
    elif frac < 0.75:
        final = int(avg) + 0.5
        rule = "Rounded to HALF (.50)"
    else:
        final = int(avg) + 1
        rule = "Rounded UP (.75)"

    # ===== HARD CAP APPLY =====
    if overall_cap is not None and final > overall_cap:
        final = overall_cap
        rule = f"Hard capped at {overall_cap} due to task violation"

    return {
        "average": round(avg, 2),
        "final_band": final,
        "rounding_applied": rule,
    }




# =====================================================
# PHASE 7 – FEEDBACK GENERATION (TEXT)
# =====================================================
def phase7_feedback(
    llm,
    chart,
    essay,
    bands,
    soft_traces=None,
    hard_traces=None
):
    soft_traces = soft_traces or []
    hard_traces = hard_traces or []

    system_prompt = load_prompt("phase7_feedback.txt")

    user_prompt = f"""
[CHART]
{chart}

[ESSAY]
{essay}

[FINAL BANDS]
{bands}

[HARD CAPS APPLIED]
{hard_traces}

[SOFT PENALTIES APPLIED]
{soft_traces}

INSTRUCTIONS:
- Explain ONLY issues that appear in HARD CAPS or SOFT PENALTIES
- Do NOT invent new problems
- For each issue, explain:
  1. What is wrong
  2. Where it appears in the essay
  3. Why it blocks a higher band
  4. How to fix it
- If content is irrelevant, say explicitly that it is NOT RELATED to the chart
- Use IELTS examiner logic, not AI guessing
"""

    feedback_text = llm.ask(system_prompt, user_prompt)

    return {
        "type": "tutor_feedback",
        "content": feedback_text.strip()
    }
    
def phase7_feedback_task2(
    llm,
    question,
    essay,
    bands,
    soft_traces=None,
    hard_traces=None
):
    soft_traces = soft_traces or []
    hard_traces = hard_traces or []

    system_prompt = load_prompt("phase7_feedback_task2.txt")

    user_prompt = f"""
[QUESTION]
{question}

[ESSAY]
{essay}

[FINAL BANDS]
{bands}

[HARD CAPS APPLIED]
{hard_traces}

[SOFT PENALTIES APPLIED]
{soft_traces}

INSTRUCTIONS:
- Explain ONLY issues that appear in HARD CAPS or SOFT PENALTIES
- Do NOT invent new problems
- For each issue, explain:
  1. What is wrong
  2. Where it appears in the essay
  3. Why it blocks a higher band for IELTS Writing Task 2
  4. How to fix it
- If content is irrelevant, say explicitly that it is NOT RELATED to the QUESTION
- Use official IELTS examiner logic
- Do NOT comment on grammar or vocabulary unless they appear in penalties
"""

    feedback_text = llm.ask(system_prompt, user_prompt)

    return {
        "type": "tutor_feedback",
        "content": feedback_text.strip()
    }



# =====================================================
# INTERNAL UTIL
# =====================================================
def _ensure_band(result: dict, phase: str, fallback: float = 5.0):
    if not isinstance(result, dict):
        raise ValueError(f"{phase}: invalid JSON result")

    band = result.get("band", None)

    # Case 1: missing or null band → fallback
    if band is None:
        result["band"] = fallback
        result["_band_fallback"] = True
        return result

    # Case 2: numeric already
    if isinstance(band, (int, float)):
        result["band"] = float(band)
        return result

    # Case 3: string → try extract number
    if isinstance(band, str):
        try:
            num = float(band.strip())
            result["band"] = num
            result["_band_coerced"] = True
            return result
        except ValueError:
            pass

    # Case 4: total garbage → fallback
    result["band"] = fallback
    result["_band_fallback"] = True
    return result

