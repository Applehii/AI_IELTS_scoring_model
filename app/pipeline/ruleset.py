# =====================================================
# HARD CAPS – FUNDAMENTAL TASK FAILURES ONLY
# =====================================================

HARD_CAP_RULES = {

    # Using irrelevant or off-task data
    "irrelevant_data": {
        "TA_max": 6.0,
        "overall_max": 6.5
    },

    # No overview at all
    "no_overview": {
        "TA_max": 6.0,
        "overall_max": 6.5
    },

    # Misinterpreting trends or values
    "data_misinterpretation": {
        "TA_max": 5.5,
        "overall_max": 6.0
    },
    
    #Task2
    
    # No position/opinion stated at all
    "no_position": {
        "TR_max": 5.0,
        "overall_max": 5.5
    },

    # Position is completely contradictory
    "contradictory_position": {
        "TR_max": 5.5,
        "overall_max": 6.0
    },

    # Off-topic or unrelated content
    "irrelevant_content": {
        "TR_max": 6.0,
        "overall_max": 6.5
    },

    # Addresses wrong task type (e.g., agrees/disagrees instead of discuss)
    "mixed_task": {
        "TR_max": 5.5,
        "overall_max": 6.0
    },
}


# =====================================================
# SOFT RULES – NON-FATAL QUALITY ISSUES
# =====================================================

SOFT_RULES = {

    # Generic but valid overview
    "weak_overview": {
        "criterion": "TA",
        "penalty": 0.25,
        "explain": "Overview is present but generic"
    },

    # Missing a dominant extreme
    "missing_key_extreme": {
        "criterion": "TA",
        "penalty": 0.25,
        "explain": "A dominant extreme is not highlighted"
    },

    # Mostly descriptive comparisons
    "limited_comparison": {
        "criterion": "TA",
        "penalty": 0.25,
        "explain": "Limited synthesis across categories"
    },

    # Listing numbers mechanically
    "mechanical_listing": {
        "criterion": "TA",
        "penalty": 0.25,
        "explain": "Data listed mechanically with limited synthesis"
    },

    # Excessive detail
    "over_detailing": {
        "criterion": "CC",
        "penalty": 0.25,
        "explain": "Too much detail reduces clarity and focus"
    },

    # Weak logical grouping
    "weak_grouping": {
        "criterion": "CC",
        "penalty": 0.25,
        "explain": "Paragraphs not grouped logically by trend"
    },

    # Grammar range limiter
    "repetitive_sentence_openings": {
        "criterion": "GRA",
        "penalty": 0.25,
        "explain": "Repetitive sentence openings reduce grammatical range"
    },
    
     # Position is weak, vague, or inconsistent
    "weak_position": {
        "criterion": "TR",
        "penalty": 0.25,
        "explain": "Position is unclear or partially inconsistent"
    },

    # Ideas underdeveloped or examples weak
    "underdeveloped_ideas": {
        "criterion": "TR",
        "penalty": 0.25,
        "explain": "Main ideas lack depth or examples"
    },

    # Limited cohesion / weak logical progression
    "limited_cohesion": {
        "criterion": "TR",
        "penalty": 0.25,
        "explain": "Essay lacks clear structure or progression"
    }
}


GRA_CEILING_RULES = {

    "capitalization_errors": 7.0,
    "systematic_punctuation_errors": 7.0,
    "run_on_sentences": 7.0
}