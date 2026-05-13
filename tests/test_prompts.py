from nanny_workshop.prompts import (
    NANNY_RESUME_EXTRACTION_PROMPT,
    PARENT_INTAKE_EXTRACTION_PROMPT,
    BIRTHDAY_WISH_PROMPT,
    BOOKING_CONFIRM_EMAIL_PROMPT,
)


def test_extraction_prompts_include_json_keyword():
    assert "JSON" in NANNY_RESUME_EXTRACTION_PROMPT
    assert "JSON" in PARENT_INTAKE_EXTRACTION_PROMPT


def test_birthday_prompt_mentions_personalization():
    assert "{child_name}" in BIRTHDAY_WISH_PROMPT
    assert "{age}" in BIRTHDAY_WISH_PROMPT


def test_booking_email_prompt_mentions_tone():
    assert "tone" in BOOKING_CONFIRM_EMAIL_PROMPT.lower()
