"""Pydantic models shared across all workshop notebooks."""

from typing import Literal
from pydantic import BaseModel, Field

DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class NannyProfile(BaseModel):
    """A nanny candidate extracted from a resume PDF."""

    id: str
    name: str
    years_experience: int = Field(ge=0, le=60)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    availability_days: list[DayOfWeek] = Field(default_factory=list)
    pet_friendly: bool = False
    special_skills: list[str] = Field(default_factory=list)
    bio: str = ""


class ChildProfile(BaseModel):
    """One child in a parent intake form."""

    age_years: float = Field(ge=0, le=18)
    needs_nap: bool = False
    notes: str = ""


class ParentIntake(BaseModel):
    """A parent intake form extracted from a PDF."""

    id: str
    family_name: str
    children: list[ChildProfile] = Field(default_factory=list)
    schedule_days: list[DayOfWeek] = Field(default_factory=list)
    schedule_hours_per_day: int = Field(default=4, ge=1, le=24)
    must_haves: list[str] = Field(default_factory=list)
    nice_to_haves: list[str] = Field(default_factory=list)
    neighborhood: str = ""
    notes: str = ""


class BookingRequest(BaseModel):
    """A parent-confirmed booking proposal produced by the agent."""

    parent_id: str
    nanny_id: str
    day: DayOfWeek
    hours: int = Field(ge=1, le=24)
    notes: str = ""
