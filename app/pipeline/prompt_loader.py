from pathlib import Path
from app.pipeline.rubric_cache import get_rubric

PROMPT_DIR = Path("app/pipeline/prompts")


def load_prompt(
    filename: str,
    rubric_name: str | None = None
) -> str:
    """
    filename: phase2_ta.txt, phase3_cc.txt, ...
    rubric_name: TA | CC | LR | GRA | None
    """
    path = PROMPT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    prompt = path.read_text(encoding="utf-8")

    if rubric_name:
        rubric = get_rubric(rubric_name)
        prompt = prompt.replace(
            f"{{{rubric_name}_RUBRIC}}",
            rubric
        )

    return prompt
