import pytest
from pydantic import ValidationError
from nanny_workshop.models import NannyProfile, ParentIntake, ChildProfile


def test_nanny_profile_minimum_fields():
    n = NannyProfile(
        id="n_01",
        name="Alex Rivera",
        years_experience=5,
        certifications=["CPR", "First Aid"],
        languages=["English", "Spanish"],
        availability_days=["mon", "tue", "wed"],
        pet_friendly=True,
        special_skills=["infant care"],
        bio="Bilingual nanny with 5 years experience.",
    )
    assert n.id == "n_01"
    assert "CPR" in n.certifications


def test_nanny_profile_rejects_bad_day():
    with pytest.raises(ValidationError):
        NannyProfile(
            id="n_02",
            name="Sam",
            years_experience=1,
            certifications=[],
            languages=["English"],
            availability_days=["funday"],  # not a valid day
            pet_friendly=False,
            special_skills=[],
            bio="x",
        )


def test_parent_intake_with_children():
    p = ParentIntake(
        id="p_01",
        family_name="Chen",
        children=[ChildProfile(age_years=2, needs_nap=True, notes="loves trains")],
        schedule_days=["thu"],
        schedule_hours_per_day=6,
        must_haves=["CPR"],
        nice_to_haves=["Spanish"],
        neighborhood="Eastside",
        notes="First time leaving baby.",
    )
    assert p.children[0].age_years == 2
    assert "CPR" in p.must_haves
