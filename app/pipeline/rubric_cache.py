from functools import lru_cache
from pathlib import Path

RUBRIC_DIR = Path("data/writing_rubric")


@lru_cache(maxsize=16)
def get_rubric(name: str) -> str:
    """
    name: TA | CC | LR | GRA
    """
    path = RUBRIC_DIR / f"{name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Rubric not found: {path}")

    return path.read_text(encoding="utf-8")
