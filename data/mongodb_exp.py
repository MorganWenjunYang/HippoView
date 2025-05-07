# JSON document schema

# {
#   "nct_id": "NCT01234567",
#   "brief_title": "A Study of Something",
#   "study_type": "Interventional",
#   "overall_status": "Completed",
#   "why_stopped": null,
#   "start_date": "2020-01-01T00:00:00",
#   "completion_date": "2021-01-01T00:00:00",
#   "phase": "Phase 2",
#   "enrollment": 100,
#   "brief_summary": "This is a summary.",
#   "allocation": "Randomized",
#   "intervention_model": "Parallel Assignment",
#   "primary_purpose": "Treatment",
#   "time_perspective": null,
#   "masking": "Double",
#   "observational_model": null,
#   "criteria": "Inclusion: ... Exclusion: ...",
#   "gender": "All",
#   "minimum_age": "18 Years",
#   "maximum_age": "65 Years",
#   "healthy_volunteers": "No",
#   "sampling_method": "Probability Sample",
#   "condition": ["Diabetes Mellitus", "Hypertension"],
#   "intervention": ["Metformin", "Placebo"],
#   "keyword": ["Blood Sugar", "Insulin"],
#   "outcomes": [
#     {
#       "outcome_type": "Primary",
#       "title": "Change in HbA1c",
#       "description": "Measured at 6 months",
#       "time_frame": "6 months",
#       "population": "Adults",
#       "units": "%"
#     }
#   ],
#   "sponsors": [
#     {
#       "agency_class": "Industry",
#       "lead_or_collaborator": "Lead",
#       "name": "PharmaCorp"
#     }
#   ]
# }

from utils import fetch_document_by_nct_id, fetch_documents_by_sponsor_name

if __name__ == "__main__":
    document=fetch_document_by_nct_id("NCT01724996")
    print(document)

    documents=fetch_documents_by_sponsor_name("AstraZeneca")
    print(documents[1])

    documents=fetch_documents_by_sponsor_name("Roche")
    print(documents[1])

