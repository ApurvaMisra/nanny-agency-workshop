"""Generate synthetic nanny resume and parent intake PDFs.

Run once: `python scripts/generate_pdfs.py`. Commit the resulting PDFs.

Each PDF is rendered from LLM-drafted text so the workshop has diverse, realistic
content while avoiding any real PII.
"""

import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from nanny_workshop.openai_client import CachedOpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

NANNY_PERSONAS = [
    "bilingual (English/Spanish), 8 years experience, infant-CPR certified, weekend availability, pet-friendly",
    "early-childhood-education degree, 3 years experience, specializes in toddlers (1-3 years), Mon-Fri only, no pets due to allergy",
    "twelve years experience, CPR + First Aid, fluent in Mandarin, available every day except Sundays, comfortable with multiples (twins/triplets)",
    "former NICU nurse, 6 years private nanny experience, infant/special-needs focus, weekday mornings only, CPR + medication-administration certified",
    "art-and-music background, 2 years experience, ages 4-10, after-school hours, pet-friendly (dogs and cats), runs creative workshops",
]

PARENT_PERSONAS = [
    "first-time parents, 6-month-old infant, both work hybrid (in-office Tue/Thu), need CPR-certified nanny, no pets, anxious about leaving baby",
    "experienced family with twin 3-year-olds, both parents medical residents, irregular schedule, must love dogs (golden retriever at home)",
    "single parent with a 5-year-old, school pickup at 3pm + 4 hours after-school care Mon/Wed/Fri, child has mild peanut allergy",
    "family with a 7-year-old and a 2-year-old, weekend brunch coverage Sat 9-1, child #2 needs a nap, prefers Spanish-speaking nanny for language exposure",
    "first child due in 3 months, looking ahead for night-nurse / newborn-care specialist, 3 nights/week, NICU or postpartum experience strongly preferred",
]


def draft_text(client: CachedOpenAI, kind: str, persona: str, idx: int) -> str:
    if kind == "resume":
        prompt = (
            f"Write a realistic 1-page nanny resume in plain text (no markdown) for this persona:\n"
            f"{persona}\n\n"
            f"Include: full name (invented), summary, certifications with dates, "
            f"work experience (2-3 entries with dates and family descriptions only - no PII), "
            f"languages, availability, references-on-request.\n"
            f"Use the placeholder identifier RESUME-{idx:02d} in the header.\n"
            f"Keep under 400 words."
        )
    else:
        prompt = (
            f"Write a realistic nanny-agency intake form filled out by a parent, in plain text "
            f"(no markdown), for this persona:\n"
            f"{persona}\n\n"
            f"Include: family identifier INTAKE-{idx:02d}, family last name (invented), "
            f"number of children with ages, schedule needed, must-have requirements, "
            f"nice-to-haves, neighborhood (general - e.g., 'Eastside'), free-text notes.\n"
            f"Keep under 350 words."
        )
    return client.complete(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )


def render_pdf(text: str, output_path: Path) -> None:
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    y = height - 72
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, width=90) or [""]:
            if y < 72:
                c.showPage()
                y = height - 72
            c.drawString(72, y, line)
            y -= 14
    c.save()


def main() -> None:
    cache_dir = ROOT / ".cache" / "pdf_gen"
    client = CachedOpenAI(cache_dir=cache_dir)

    for i, persona in enumerate(NANNY_PERSONAS, start=1):
        text = draft_text(client, "resume", persona, i)
        out = PDF_DIR / f"nanny_resume_{i:02d}.pdf"
        render_pdf(text, out)
        print(f"wrote {out}")

    for i, persona in enumerate(PARENT_PERSONAS, start=1):
        text = draft_text(client, "intake", persona, i)
        out = PDF_DIR / f"parent_intake_{i:02d}.pdf"
        render_pdf(text, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
