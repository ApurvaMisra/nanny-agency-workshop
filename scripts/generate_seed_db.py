"""Generate the seed database from synthetic PDFs.

Run once: `python scripts/generate_seed_db.py`. Commit the resulting JSON.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from nanny_workshop.models import NannyProfile, ParentIntake
from nanny_workshop.openai_client import CachedOpenAI
from nanny_workshop.pdf_loader import extract_text
from nanny_workshop.prompts import (
    NANNY_RESUME_EXTRACTION_PROMPT,
    PARENT_INTAKE_EXTRACTION_PROMPT,
)

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT = ROOT / "data" / "seed_db.json"


def extract_nanny(client: CachedOpenAI, pdf_path: Path, idx: int) -> dict:
    text = extract_text(pdf_path)
    raw = client.complete(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": NANNY_RESUME_EXTRACTION_PROMPT.replace("{resume_text}", text),
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(raw)
    data["id"] = f"n_{idx:02d}"
    return NannyProfile.model_validate(data).model_dump()


def extract_parent(client: CachedOpenAI, pdf_path: Path, idx: int) -> dict:
    text = extract_text(pdf_path)
    raw = client.complete(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": PARENT_INTAKE_EXTRACTION_PROMPT.replace("{intake_text}", text),
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(raw)
    data["id"] = f"p_{idx:02d}"
    return ParentIntake.model_validate(data).model_dump()


def main() -> None:
    cache_dir = ROOT / ".cache" / "seed_db"
    client = CachedOpenAI(cache_dir=cache_dir)

    nannies = [
        extract_nanny(client, p, i)
        for i, p in enumerate(sorted(PDF_DIR.glob("nanny_resume_*.pdf")), start=1)
    ]
    parents = [
        extract_parent(client, p, i)
        for i, p in enumerate(sorted(PDF_DIR.glob("parent_intake_*.pdf")), start=1)
    ]

    OUT.write_text(json.dumps({"nannies": nannies, "parents": parents}, indent=2))
    print(f"wrote {OUT} — {len(nannies)} nannies, {len(parents)} parents")


if __name__ == "__main__":
    main()
