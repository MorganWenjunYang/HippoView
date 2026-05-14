import re
from typing import Dict, Optional
from urllib.parse import urlencode

class CTGovQueryBuilder:
    """
    A class to translate human queries into ClinicalTrials.gov API URLs.
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    
    def __init__(self):
        self.params = {
            "format": "json",
            "pageSize": 100,
            "countTotal": "true"
        }
    
    def add_condition(self, condition: str) -> 'CTGovQueryBuilder':
        """Add a condition to search for"""
        self.params["query.cond"] = condition
        return self
        
    def add_intervention(self, intervention: str) -> 'CTGovQueryBuilder':
        """Add an intervention to search for"""
        self.params["query.intr"] = intervention
        return self
        
    def add_sponsor(self, sponsor: str) -> 'CTGovQueryBuilder':
        """Add a sponsor to search for"""
        self.params["query.spons"] = sponsor
        return self
        
    def set_status(self, status: str) -> 'CTGovQueryBuilder':
        """Set the recruitment status filter"""
        valid_statuses = [
            "NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION",
            "ACTIVE_NOT_RECRUITING", "SUSPENDED", "TERMINATED", "COMPLETED",
            "WITHDRAWN", "UNKNOWN_STATUS"
        ]
        if status.upper() in valid_statuses:
            self.params["filter.overallStatus"] = status.upper()
        return self
        
    def set_study_type(self, study_type: str) -> 'CTGovQueryBuilder':
        """Set the study type filter"""
        self.params["filter.advanced"] = f"AREA[StudyType]{study_type.upper()}"
        return self
        
    def set_phase(self, phase: str) -> 'CTGovQueryBuilder':
        """Set the study phase filter"""
        valid_phases = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NOT_APPLICABLE"]
        if phase.upper() in valid_phases:
            self.params["filter.phase"] = phase.upper()
        return self
        
    def build_url(self) -> str:
        """Build the final URL with all parameters"""
        return f"{self.BASE_URL}?{urlencode(self.params, doseq=True)}"

def translate_human_query(query: str) -> str:
    """
    Translate a human query into a ClinicalTrials.gov API URL.
    
    Example queries:
    - "Find breast cancer trials"
    - "Show recruiting trials for COVID-19"
    - "Find phase 3 diabetes studies"
    - "Show interventional studies for Alzheimer's by Pfizer"
    """
    builder = CTGovQueryBuilder()
    
    # Extract condition
    condition_match = re.search(r"(?:for|about|on|treating)\s+([^,\.]+?)(?:\s+(?:by|in|that|with|trials|studies)|$)", query.lower())
    if condition_match:
        builder.add_condition(condition_match.group(1).strip())
    
    # Check recruitment status
    if "recruiting" in query.lower():
        builder.set_status("RECRUITING")
    elif "completed" in query.lower():
        builder.set_status("COMPLETED")
        
    # Check study type
    if "interventional" in query.lower():
        builder.set_study_type("INTERVENTIONAL")
    elif "observational" in query.lower():
        builder.set_study_type("OBSERVATIONAL")
        
    # Check phase
    phase_match = re.search(r"phase\s+(\d)", query.lower())
    if phase_match:
        phase_num = phase_match.group(1)
        builder.set_phase(f"PHASE{phase_num}")
        
    # Check sponsor
    sponsor_match = re.search(r"by\s+([^,\.]+?)(?:\s+(?:in|that|with|trials|studies)|$)", query.lower())
    if sponsor_match:
        builder.add_sponsor(sponsor_match.group(1).strip())
    
    return builder.build_url()

# Example usage
if __name__ == "__main__":
    example_queries = [
        "Find breast cancer trials",
        "Show recruiting trials for COVID-19",
        "Find phase 3 diabetes studies",
        "Show interventional studies for Alzheimer's by Pfizer"
    ]
    
    for query in example_queries:
        print(f"\nQuery: {query}")
        print(f"URL: {translate_human_query(query)}")
