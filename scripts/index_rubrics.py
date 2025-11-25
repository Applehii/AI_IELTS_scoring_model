import os
from pathlib import Path

from app.rag_manager import RAGManager


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_text_files(base_path: Path):
    for root, dirs, files in os.walk(base_path):
        for fname in files:
            if fname.lower().endswith((".txt", ".md")):
                full_path = Path(root) / fname
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                yield full_path, content


def parse_writing_rubric_meta(file_path: Path):
    name = file_path.stem
    parts = name.split("_")
    band = None
    criterion = None

    for p in parts:
        if p.lower().startswith("band"):
            band = p.lower().replace("band", "")
        else:
            criterion = p.upper()

    return {
        "type": "writing_rubric",
        "band": band,
        "criterion": criterion,
    }


def parse_writing_sample_meta(file_path: Path):
    name = file_path.stem
    parts = name.split("_")
    task = None
    band = None

    for p in parts:
        if p.lower().startswith("task"):
            task = p.lower().replace("task", "")
        if p.lower().startswith("band"):
            band = p.lower().replace("band", "")

    return {
        "type": "writing_sample",
        "task": task,
        "band": band,
    }


def parse_speaking_rubric_meta(file_path: Path):
    name = file_path.stem
    parts = name.split("_")
    band = None
    criterion = None
    for p in parts:
        if p.lower().startswith("band"):
            band = p.lower().replace("band", "")
        else:
            criterion = p.upper()
    return {
        "type": "speaking_rubric",
        "band": band,
        "criterion": criterion,
    }


def parse_speaking_sample_meta(file_path: Path):
    name = file_path.stem
    parts = name.split("_")
    part = None
    band = None
    for p in parts:
        if p.lower().startswith("part"):
            part = p.lower().replace("part", "")
        if p.lower().startswith("band"):
            band = p.lower().replace("band", "")
    return {
        "type": "speaking_sample",
        "part": part,
        "band": band,
    }


def main():
    rag = RAGManager()

    # Writing rubric
    writing_rubric_dir = DATA_DIR / "writing_rubric"
    print(f"Indexing writing rubrics from {writing_rubric_dir}")
    for path, text in load_text_files(writing_rubric_dir):
        meta = parse_writing_rubric_meta(path)
        doc_id = f"writing_rubric::{path.stem}"
        rag.add_document(doc_id, text, meta)

    # Writing samples
    writing_samples_dir = DATA_DIR / "writing_samples"
    print(f"Indexing writing samples from {writing_samples_dir}")
    for path, text in load_text_files(writing_samples_dir):
        meta = parse_writing_sample_meta(path)
        doc_id = f"writing_sample::{path.stem}"
        rag.add_document(doc_id, text, meta)

    # FIXED: Speaking rubric
    speaking_rubric_dir = DATA_DIR / "speaking_rubric"
    print(f"Indexing speaking rubrics from {speaking_rubric_dir}")
    for path, text in load_text_files(speaking_rubric_dir):
        meta = parse_speaking_rubric_meta(path)
        doc_id = f"speaking_rubric::{path.stem}"
        rag.add_document(doc_id, text, meta)

    # Speaking samples
    speaking_samples_dir = DATA_DIR / "speaking_samples"
    print(f"Indexing speaking samples from {speaking_samples_dir}")
    for path, text in load_text_files(speaking_samples_dir):
        meta = parse_speaking_sample_meta(path)
        doc_id = f"speaking_sample::{path.stem}"
        rag.add_document(doc_id, text, meta)

# ========== Writing guides (Task 1 & Task 2) ==========
    # ========== Writing Task 1 Guides ==========
    writing_task1_dir = DATA_DIR / "writing_task1_guides"
    print(f"Indexing Task 1 guides from {writing_task1_dir}")
    for path, text in load_text_files(writing_task1_dir):
        doc_id = f"writing_task1_guide::{path.stem}"
        meta = {
            "type": "writing_task1_guide",
            "task": "1"
        }
        rag.add_document(doc_id, text, meta)

    # ========== Writing Task 2 Guides ==========
    writing_task2_dir = DATA_DIR / "writing_task2_guides"
    print(f"Indexing Task 2 guides from {writing_task2_dir}")
    for path, text in load_text_files(writing_task2_dir):
        doc_id = f"writing_task2_guide::{path.stem}"
        meta = {
            "type": "writing_task2_guide",
            "task": "2"
        }
        rag.add_document(doc_id, text, meta)

    print("✅ Done indexing rubrics & samples into ChromaDB.")


if __name__ == "__main__":
    main()
