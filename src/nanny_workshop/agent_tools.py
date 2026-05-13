"""Python tool implementations dispatched by the BAML agent.

Tools are pure functions that take string-keyed args and return either str
(get_policy, draft_email) or list[dict] (search_nannies) or bool (check_availability).
"""

import json
import os
import re
from pathlib import Path
from functools import lru_cache

from .openai_client import CachedOpenAI
from .chroma_client import NannyChroma

ROOT = Path(__file__).resolve().parent.parent.parent
SEED_PATH = ROOT / "data" / "seed_db.json"
POLICIES_PATH = ROOT / "data" / "policies" / "agency_policies.md"
NANNY_DB = ROOT / "nanny_db"


@lru_cache(maxsize=1)
def _seed() -> dict:
    return json.loads(SEED_PATH.read_text())


@lru_cache(maxsize=1)
def _policies() -> str:
    return POLICIES_PATH.read_text()


@lru_cache(maxsize=1)
def _client() -> CachedOpenAI:
    return CachedOpenAI(cache_dir=ROOT / ".cache" / "agent_tools")


@lru_cache(maxsize=1)
def _db() -> NannyChroma:
    return NannyChroma(persist_dir=NANNY_DB, collection_name="profiles")


def search_nannies(query: str, top_k: int = 3) -> list[dict]:
    """Search the Chroma collection for nannies matching the query."""
    vec = _client().embed(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        text=query,
    )
    out = _db().query(query_embedding=vec, n_results=top_k, where={"role": "nanny"})
    results = []
    seed_by_id = {n["id"]: n for n in _seed()["nannies"]}
    for nanny_id, meta in zip(out["ids"][0], out["metadatas"][0]):
        record = seed_by_id.get(nanny_id, {})
        results.append({
            "id": nanny_id,
            "name": meta.get("name", record.get("name", "?")),
            "certifications": record.get("certifications", []),
            "languages": record.get("languages", []),
            "availability_days": record.get("availability_days", []),
        })
    return results


def get_policy(topic: str) -> str:
    """Return the policy section matching `topic`, or '' if not found.

    Match is fuzzy: tokens from `topic` are searched case-insensitively in section headings.
    """
    text = _policies()
    sections = re.split(r"\n## ", "\n" + text)
    topic_tokens = [t for t in re.findall(r"\w+", topic.lower()) if len(t) > 2]
    for sec in sections:
        heading_line = sec.split("\n", 1)[0].lower()
        if any(t in heading_line for t in topic_tokens):
            return ("## " + sec).strip() if not sec.startswith("# ") else sec.strip()
    return ""


def check_availability(nanny_id: str, day: str) -> bool:
    """Return True if the nanny is available on the given 3-letter day code."""
    nanny = next((n for n in _seed()["nannies"] if n["id"] == nanny_id), None)
    if nanny is None:
        return False
    return day.lower() in [d.lower() for d in nanny["availability_days"]]


def draft_email(parent_id: str, nanny_id: str, day: str, hours: str, notes: str = "") -> str:
    """Generate a booking confirmation email."""
    parents = {p["id"]: p for p in _seed()["parents"]}
    nannies = {n["id"]: n for n in _seed()["nannies"]}
    parent = parents.get(parent_id, {"family_name": parent_id})
    nanny = nannies.get(nanny_id, {"name": nanny_id})

    return _client().complete(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Nanny Agency's customer-care assistant. "
                    "Tone: warm, professional, reassuring. 4-6 sentences. "
                    "Sign as 'The Nanny Agency Team'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Draft a booking confirmation email.\n"
                    f"Family: {parent['family_name']}\n"
                    f"Nanny: {nanny['name']}\n"
                    f"Day: {day}\n"
                    f"Hours: {hours}\n"
                    f"Special notes: {notes}\n"
                    f"Include the family name in the greeting and the nanny name + day in the body."
                ),
            },
        ],
        temperature=0.0,
    )
