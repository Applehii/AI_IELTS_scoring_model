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


# =====================================================
# PHASE 3 – COHERENCE & COHESION
# =====================================================
def phase3_cc(llm, parsed_essay):
    system_prompt = load_prompt(
        "phase3_cc.txt",
        rubric_name="CC"
    )

    user_prompt = f"""
[OVERVIEW]
{parsed_essay.get("overview")}

[BODY_PARAGRAPHS]
{parsed_essay.get("body_paragraphs")}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    _ensure_band(result, "CC")
    return result


# =====================================================
# PHASE 4 – LEXICAL RESOURCE
# =====================================================
def phase4_lr(llm, essay: str):
    system_prompt = load_prompt(
        "phase4_lr.txt",
        rubric_name="LR"
    )

    raw = llm.ask(system_prompt, essay)
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

    user_prompt = f"""
[SENTENCES]
{parsed_essay.get("sentences")}
"""

    raw = llm.ask(system_prompt, user_prompt)
    result = extract_json(raw)

    _ensure_band(result, "GRA")
    return result


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
def phase7_feedback(llm, chart, essay, bands):
    system_prompt = load_prompt(
        "phase7_feedback.txt"
    )

    user_prompt = f"""
[CHART]
{chart}

[ESSAY]
{essay}

[EVALUATION]
{bands}
"""


    feedback_text = llm.ask(system_prompt, user_prompt)

    return {
        "type": "tutor_feedback",
        "content": feedback_text.strip()
    }


# =====================================================
# INTERNAL UTIL
# =====================================================
def _ensure_band(result: dict, phase: str):
    if not isinstance(result, dict):
        raise ValueError(f"{phase}: invalid JSON result")

    if "band" not in result:
        raise ValueError(f"{phase}: missing 'band' field")

    try:
        result["band"] = float(result["band"])
    except Exception:
        raise ValueError(f"{phase}: band is not numeric")

    return result
