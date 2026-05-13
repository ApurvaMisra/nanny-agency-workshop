// Shared BAML types. Mirror src/nanny_workshop/models.py so Python and BAML agree.

class NannyProfile {
  id string
  name string
  years_experience int
  certifications string[]
  languages string[]
  availability_days string[]
  pet_friendly bool
  special_skills string[]
  bio string
}

class ChildProfile {
  age_years float
  needs_nap bool
  notes string
}

class ParentIntake {
  id string
  family_name string
  children ChildProfile[]
  schedule_days string[]
  schedule_hours_per_day int
  must_haves string[]
  nice_to_haves string[]
  neighborhood string
  notes string
}

class BookingRequest {
  parent_id string
  nanny_id string
  day string
  hours int
  notes string
}
