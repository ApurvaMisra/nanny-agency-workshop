"""Shared prompt strings used across workshop notebooks."""

NANNY_RESUME_EXTRACTION_PROMPT = """You are extracting structured data from a nanny's resume.

Return a JSON object matching this schema exactly:
{
  "id": str,
  "name": str,
  "years_experience": int,
  "certifications": [str],
  "languages": [str],
  "availability_days": [str],  // 3-letter lowercase: mon, tue, wed, thu, fri, sat, sun
  "pet_friendly": bool,
  "special_skills": [str],
  "bio": str
}

Use the resume text only. Do not invent data. If a field is missing, use a sensible default
(empty list, empty string, or false).

Resume text:
---
{resume_text}
---
"""

PARENT_INTAKE_EXTRACTION_PROMPT = """You are extracting structured data from a parent's intake form for a nanny agency.

Return a JSON object matching this schema exactly:
{
  "id": str,
  "family_name": str,
  "children": [{"age_years": float, "needs_nap": bool, "notes": str}],
  "schedule_days": [str],  // 3-letter lowercase: mon, tue, wed, thu, fri, sat, sun
  "schedule_hours_per_day": int,
  "must_haves": [str],
  "nice_to_haves": [str],
  "neighborhood": str,
  "notes": str
}

Use the intake text only. Do not invent data.

Intake text:
---
{intake_text}
---
"""

BIRTHDAY_WISH_PROMPT = """Write a warm 2-sentence birthday wish for a child.

Child name: {child_name}
Age turning: {age}
Interests: {interests}

Keep it age-appropriate and personal. Mention at least one interest."""

BOOKING_CONFIRM_EMAIL_PROMPT = """Draft a booking confirmation email to a parent.

Tone: warm, professional, reassuring. Match the family's communication style.

Parent name: {parent_name}
Nanny name: {nanny_name}
Day: {day}
Hours: {hours}
Special notes: {notes}

The email should be 4-6 sentences. Sign as "The Nanny Agency Team"."""
