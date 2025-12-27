from .ruleset import HARD_CAP_RULES

def apply_hard_caps(bands: dict, violations: dict):
    capped = bands.copy()
    overall_cap = None
    applied_rules = []

    for violation, active in violations.items():
        if not active:
            continue

        rule = HARD_CAP_RULES.get(violation, {})

        for k, max_value in rule.items():

            if k.endswith("_max") and k != "overall_max":
                crit = k.replace("_max", "")

                if crit in capped:
                    if not isinstance(max_value, (int, float)):
                        raise ValueError(
                            f"Invalid hard cap value for {violation}.{k}: {max_value}"
                        )

                    capped[crit] = min(capped[crit], max_value)
                    applied_rules.append(
                        f"{crit} capped at {max_value} due to {violation}"
                    )

            if k == "overall_max":
                if not isinstance(max_value, (int, float)):
                    raise ValueError(
                        f"Invalid overall_max for {violation}: {max_value}"
                    )

                overall_cap = (
                    max_value if overall_cap is None else min(overall_cap, max_value)
                )

    return capped, overall_cap, applied_rules
