import json
from pathlib import Path

import pytest

from nanny_workshop.agent_tools import (
    search_nannies,
    get_policy,
    check_availability,
    draft_email,
)


@pytest.fixture(scope="module")
def seed():
    return json.loads((Path(__file__).resolve().parent.parent / "data" / "seed_db.json").read_text())


def test_search_nannies_returns_at_most_top_k(seed):
    results = search_nannies(query="bilingual CPR weekend", top_k=3)
    assert len(results) <= 3
    assert all("name" in r and "id" in r for r in results)


def test_get_policy_finds_section(seed):
    text = get_policy(topic="booking confirmation")
    assert "confirmed" in text.lower()
    assert len(text) > 50


def test_get_policy_unknown_topic_returns_empty():
    text = get_policy(topic="quantum mechanics")
    assert text == ""


def test_check_availability_with_known_nanny(seed):
    nanny_id = seed["nannies"][0]["id"]
    available_days = seed["nannies"][0]["availability_days"]
    assert check_availability(nanny_id=nanny_id, day=available_days[0]) is True
    all_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    unavail = list(all_days - set(available_days))
    if unavail:
        assert check_availability(nanny_id=nanny_id, day=unavail[0]) is False


def test_check_availability_unknown_nanny_returns_false():
    assert check_availability(nanny_id="n_does_not_exist", day="mon") is False


def test_check_availability_accepts_full_day_names_case_insensitively(seed):
    # Agents naturally pass "Saturday", not the stored 3-letter "sat" code.
    nanny = next(n for n in seed["nannies"] if n["availability_days"])
    code_to_name = {
        "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
        "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
    }
    avail_code = nanny["availability_days"][0]
    full_name = code_to_name[avail_code]
    assert check_availability(nanny_id=nanny["id"], day=full_name) is True
    assert check_availability(nanny_id=nanny["id"], day=full_name.upper()) is True
    all_codes = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    unavail = list(all_codes - set(nanny["availability_days"]))
    if unavail:
        assert check_availability(nanny_id=nanny["id"], day=code_to_name[unavail[0]]) is False


def test_draft_email_includes_names(seed):
    parent = seed["parents"][0]
    nanny = seed["nannies"][0]
    email = draft_email(
        parent_id=parent["id"],
        nanny_id=nanny["id"],
        day="thu",
        hours="6",
    )
    assert parent["family_name"] in email
    assert nanny["name"] in email
    assert "thu" in email.lower()
