import os
import sys
import json
from pathlib import Path

# ===============================
# Add project root to sys.path
# ===============================
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.rag_manager import RAGManager


BASE_DIR = ROOT
DATA_DIR = BASE_DIR / "data" / "samples"   # new standardized dataset


def parse_sections(content: str):
    """Parse [QUESTION], [SUMMARY], [SAMPLE_ANSWER], [OVERVIEW], [RATIONALE]."""
    blocks = {}
    current = None

    for line in content.split("\n"):
        line = line.strip()

        if line.startswith("[") and line.endswith("]"):
            header = line.strip("[]").strip().upper()
            blocks[header] = []
            current = header
        else:
            if current:
                blocks[current].append(line)

    for k in blocks:
        blocks[k] = "\n".join(blocks[k]).strip()

    return blocks


def load_all_sample_files():
    """Yield (path, raw_text, metadata_json, parsed_sections)."""

    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith(".txt"):
                continue

            full_path = Path(root) / fname
            raw = full_path.read_text(encoding="utf-8")

            # Split JSON header
            lines = raw.lstrip().split("\n", 1)

            try:
                meta = json.loads(lines[0].strip())
            except Exception as e:
                print(f"❌ ERROR: Invalid JSON header in {fname}: {e}")
                continue

            if len(lines) < 2:
                print(f"❌ ERROR: Missing content after JSON header in {fname}")
                continue

            content = lines[1]
            sections = parse_sections(content)

            # Ensure required sections exist
            if "SUMMARY" not in sections:
                print(f"⚠️ WARNING: Missing SUMMARY in {fname}")
                sections["SUMMARY"] = ""

            yield full_path, raw, meta, sections


def main():
    rag = RAGManager()
    count = 0

    for file_path, full_text, meta, sections in load_all_sample_files():
        sample_id = meta.get("sample_id", file_path.stem)
        doc_id = f"{meta['task']}::{sample_id}"

        summary = sections.get("SUMMARY", "")
        document_text = full_text
        
        clean_meta = {}
        for k, v in meta.items():
           if isinstance(v, list):
            clean_meta[k] = json.dumps(v)  # convert list to JSON string
           else:
            clean_meta[k] = v
  
        rag.add_document(
            doc_id=doc_id,
            text=document_text,
            metadata=clean_meta,
            embedding_text=summary
        )

        print(f"📌 Indexed: {doc_id}")
        count += 1

    print(f"\n🎉 DONE! Indexed {count} sample files into ChromaDB.\n")


if __name__ == "__main__":
    main()
