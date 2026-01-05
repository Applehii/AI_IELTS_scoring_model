# =====================================================
# IELTS WRITING TASK 1 – SCORING ENGINE (FIXED)
# =====================================================

from .ruleset import HARD_CAP_RULES, SOFT_RULES

# =====================================================
# PHASE 2 – TA RULE DETECTION (NO BAND CHANGE)
# =====================================================

def apply_ta_rules(ta_output):

    violations = {}
    s = ta_output["signals"]

    if s["data_misinterpretation"]:
        violations["data_misinterpretation"] = {
            "active": True,
            "location": "body",
            "evidence": None,
            "reason": "Incorrect interpretation of data"
        }

    if s["overview_present"] and s["trend_mentioned"]:
        if (not s["grouping_present"]
            and not s["dominant_pattern_mentioned"]):
            violations["weak_overview"] = {
                "active": True,
                "location": "overview",
                "evidence": ta_output["evidence"].get("overview_sentence"),
                "reason": "Overview lacks synthesis of main patterns"
            }

    return violations


# =====================================================
# PHASE 3 – GRAMMAR
# =====================================================

GRA_CEILING_RULES = {
    "capitalization_errors": 7.0,
    "systematic_punctuation_errors": 7.0,
    "run_on_sentences": 7.0
}


def apply_gra_ceiling(gra_band, gra_violations):

    capped = gra_band
    for v in gra_violations or []:
        if v in GRA_CEILING_RULES:
            capped = min(capped, GRA_CEILING_RULES[v])

    return capped


# =====================================================
# GENERIC RULE ENGINE (HARD + SOFT, SINGLE SOURCE)
# =====================================================

def apply_all_rules(bands: dict, violations: dict):
    capped = bands.copy()
    overall_cap = None
    applied_hard = []
    applied_soft = []

    # ---------- HARD CAPS ----------
    for violation, data in violations.items():
        if not data.get("active"):
            continue

        rule = HARD_CAP_RULES.get(violation)
        if not rule:
            continue

        caps = {}
        for k, max_value in rule.items():
            if not k.endswith("_max"):
                continue
            crit = k.replace("_max", "")
            caps[crit] = max_value

            if crit != "overall" and crit in capped:
                capped[crit] = min(capped[crit], max_value)

            if crit == "overall":
                overall_cap = (
                    max_value if overall_cap is None
                    else min(overall_cap, max_value)
                )

        applied_hard.append({
            "violation": violation,
            "caps": caps,
            "location": data.get("location"),
            "evidence": data.get("evidence"),
            "reason": data.get("reason"),
        })

    # ---------- SOFT PENALTIES ----------
    for violation, data in violations.items():
        if not data.get("active"):
            continue

        rule = SOFT_RULES.get(violation)
        if not rule:
            continue

        crit = rule.get("criterion")
        penalty = rule.get("penalty")

        if crit not in capped or penalty is None:
            continue

        old = capped[crit]

        if crit in ["TA", "TR"]:
            new_val = max(6.0, old - penalty)
        else:
            new_val = max(0.0, old - penalty)

        if new_val < old:
            capped[crit] = new_val
            applied_soft.append({
                "criterion": crit,
                "violation": violation,
                "location": data.get("location"),
                "evidence": data.get("evidence"),
                "reason": rule.get("explain"),
                "penalty": penalty
            })

    return capped, overall_cap, applied_hard, applied_soft


# =====================================================
# TA → OVERALL CEILING
# =====================================================

def apply_ta_overall_ceiling(final_band, ta_band):

    if ta_band < 6.0:
        return min(final_band, 6.5), "TA < 6.0 ceiling"

    if ta_band <= 6.5:
        return min(final_band, 7.0), "TA ceiling at 7.0"

    return final_band, "No TA ceiling"


def apply_tr_overall_ceiling(final_band, tr_band):
    if tr_band < 6.0:
        return min(final_band, 6.5), "TR < 6.0 ceiling"
    if tr_band <= 6.5:
        return min(final_band, 7.0), "TR ceiling at 7.0"
    return final_band, "No TR ceiling"



# =====================================================
# IELTS ROUNDING
# =====================================================

def ielts_rounding(score):

    d = score - int(score)

    if d < 0.25:
        return float(int(score))
    elif d < 0.75:
        return int(score) + 0.5
    else:
        return int(score) + 1.0


# =====================================================
# PHASE 4 – FINAL SCORING (SINGLE SOURCE OF TRUTH)
# =====================================================

def finalize_score(
    bands_after_rules,
    gra_violations,
    overall_cap
):

    ta = bands_after_rules["TA"]
    cc = bands_after_rules["CC"]
    lr = bands_after_rules["LR"]
    gra = apply_gra_ceiling(bands_after_rules["GRA"], gra_violations)

    raw = (ta + cc + lr + gra) / 4

    capped, note = apply_ta_overall_ceiling(raw, ta)

    if overall_cap is not None:
        capped = min(capped, overall_cap)
        note += " + HARD overall cap"

    final = ielts_rounding(capped)

    return final, note

def finalize_score_task2(
    bands_after_rules,
    gra_violations,
    overall_cap
):
    """
    bands_after_rules: {"TR": float, "CC": float, "LR": float, "GRA": float}
    gra_violations: list of str
    overall_cap: float | None
    """
    tr = bands_after_rules["TR"]
    cc = bands_after_rules["CC"]
    lr = bands_after_rules["LR"]
    gra = apply_gra_ceiling(bands_after_rules["GRA"], gra_violations)

    raw = (tr + cc + lr + gra) / 4

    capped, note = apply_tr_overall_ceiling(raw, tr)

    if overall_cap is not None:
        capped = min(capped, overall_cap)
        note += " + HARD overall cap"

    final = ielts_rounding(capped)

    return final, note

