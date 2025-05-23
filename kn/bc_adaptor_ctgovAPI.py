# adapt from https://github.com/biocypher/igan/blob/main/igan/adapters/clinicaltrials_adapter.py
# biolink-model: https://raw.githubusercontent.com/biolink/biolink-model/v3.2.1/biolink-model.owl.ttl
# clinicaltrials.gov API: https://clinicaltrials.gov/api/oas/v2

import random
import string
from enum import Enum, auto
from itertools import chain
from typing import Optional
from biocypher._logger import logger
import sys, os
import concurrent.futures
from functools import partial
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger.debug(f"Loading module {__name__}.")

import requests

QUERY_PARAMS = {
    "format": "json",
    "query.cond": "COPD",
    "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
    "fields": ",".join([
        # Identification fields
        "NCTId",
        "BriefTitle", 
        "OfficialTitle",
        # "IdentificationModule",
        
        # # # Status fields
        "StatusModule",# "ProtocolSection.StatusModule.OverallStatus",
        # "StartDate",# "ProtocolSection.StatusModule.StartDate",
        # "CompletionDate",# "ProtocolSection.StatusModule.CompletionDate",
        
        # # # Design fields
        # "StudyType",# "ProtocolSection.DesignModule.StudyType",
        # "Phases",# "ProtocolSection.DesignModule.Phases",
        # "EnrollmentInfo.Count",# "ProtocolSection.DesignModule.EnrollmentInfo.Count",
        # "DesignInfo.Allocation",# "ProtocolSection.DesignModule.DesignInfo.Allocation",
        # "DesignInfo.Masking",# "ProtocolSection.DesignModule.DesignInfo.Masking",
        # "DesignInfo.WhoMasked",# "ProtocolSection.DesignModule.DesignInfo.WhoMasked",
        # "DesignInfo.InterventionModel",# "ProtocolSection.DesignModule.DesignInfo.InterventionModel",
        # "DesignInfo.PrimaryPurpose",# "ProtocolSection.DesignModule.DesignInfo.PrimaryPurpose",
        "DesignModule",


        # # # Description fields
        # "BriefSummary",# "ProtocolSection.DescriptionModule.BriefSummary",
        # "DetailedDescription",# "ProtocolSection.DescriptionModule.DetailedDescription",
        "DescriptionModule",
        
        # # # Condition fields
        # "Conditions",# "ProtocolSection.ConditionsModule.Conditions",
        # "Keywords",# "ProtocolSection.ConditionsModule.Keywords",
        "ConditionsModule",
        
        # # # Intervention fields
        # "Interventions",# "ProtocolSection.ArmsInterventionsModule.Interventions",
        # "InterventionType",# "ProtocolSection.ArmsInterventionsModule.Interventions.InterventionType",
        # "InterventionName",# "ProtocolSection.ArmsInterventionsModule.Interventions.InterventionName",
        # "Description",# "ProtocolSection.ArmsInterventionsModule.Interventions.Description",
        "ArmsInterventionsModule",

        # # # Outcome fields
        # # "ProtocolSection.OutcomesModule.PrimaryOutcomes",
        # # "ProtocolSection.OutcomesModule.PrimaryOutcomes.Measure",
        # # "ProtocolSection.OutcomesModule.PrimaryOutcomes.TimeFrame",
        # # "ProtocolSection.OutcomesModule.PrimaryOutcomes.Description",
        # # "ProtocolSection.OutcomesModule.SecondaryOutcomes",
        # # "ProtocolSection.OutcomesModule.SecondaryOutcomes.Measure",
        # # "ProtocolSection.OutcomesModule.SecondaryOutcomes.TimeFrame",
        # # "ProtocolSection.OutcomesModule.SecondaryOutcomes.Description",
        "OutcomesModule",


        # # # Sponsor fields
        # # "ProtocolSection.SponsorCollaboratorsModule.LeadSponsor",
        # # "ProtocolSection.SponsorCollaboratorsModule.LeadSponsor.Name",
        # # "ProtocolSection.SponsorCollaboratorsModule.LeadSponsor.Class",
        # # "ProtocolSection.SponsorCollaboratorsModule.Collaborators",
        "SponsorCollaboratorsModule",
        
        # # Eligibility fields (for future reference)
        # "ProtocolSection.EligibilityModule.EligibilityCriteria",
        # "ProtocolSection.EligibilityModule.HealthyVolunteers",
        # "ProtocolSection.EligibilityModule.Sex",
        # "ProtocolSection.EligibilityModule.MinimumAge",
        # "ProtocolSection.EligibilityModule.MaximumAge",
        "EligibilityModule",
        
        # # Location fields (if needed)
        # "ProtocolSection.ContactsLocationsModule.Locations"
        "ContactsLocationsModule",
    ]),
    "pageSize": 100,
    "countTotal": "true"
}


class ClinicalTrialsAdapterNodeType(Enum):
    """
    Define types of nodes the adapter can provide.
    """
    STUDY = auto()
    CONDITION = auto()
    INTERVENTION = auto()
    OUTCOME = auto()
    SPONSOR = auto()


class ClinicalTrialsAdapterStudyField(Enum):
    """
    Define possible fields the adapter can provide for studies.
    """
    ID = "identificationModule/nctId"
    STUDY_TYPE = "designModule/studyType"
    STATUS = "statusModule/overallStatus"
    PHASE = "designModule/phases"
    START_DATE = "statusModule/startDate"
    COMPLETION_DATE = "statusModule/completionDate"
    ENROLLMENT = "designModule/enrollmentInfo/count"
    BRIEF_SUMMARY = "descriptionModule/briefSummary"
    ALLOCATION = "designModule/designInfo/allocation"
    MASKING = "designModule/designInfo/masking"


class ClinicalTrialsAdapterDiseaseField(Enum):
    """
    Define possible fields the adapter can provide for diseases.
    """

    ID = "id"
    NAME = "name"
    DESCRIPTION = "description"


class ClinicalTrialsAdapterEdgeType(Enum):
    """
    Define types of edges the adapter can provide.
    """
    STUDY_HAS_CONDITION = auto()
    STUDY_HAS_INTERVENTION = auto()
    STUDY_HAS_OUTCOME = auto()
    SPONSOR_HAS_STUDY = auto()
    CONDITION_HAS_INTERVENTION = auto()
    CONDITION_HAS_OUTCOME = auto()
    INTERVENTION_HAS_OUTCOME = auto()


class ClinicalTrialsAdapterProteinProteinEdgeField(Enum):
    """
    Define possible fields the adapter can provide for protein-protein edges.
    """

    INTERACTION_TYPE = "interaction_type"
    INTERACTION_SOURCE = "interaction_source"


class ClinicalTrialsAdapterProteinDiseaseEdgeField(Enum):
    """
    Define possible fields the adapter can provide for protein-disease edges.
    """

    ASSOCIATION_TYPE = "association_type"
    ASSOCIATION_SOURCE = "association_source"


class ClinicalTrialsAdapter:
    """
    ClinicalTrials BioCypher adapter. Generates nodes and edges for creating a
    knowledge graph.

    Args:
        node_types: List of node types to include in the result.
        node_fields: List of node fields to include in the result.
        edge_types: List of edge types to include in the result.
        edge_fields: List of edge fields to include in the result.
        max_workers: Maximum number of parallel workers for API requests
        chunk_size: Number of studies to process in each worker
    """

    def __init__(
        self,
        node_types: Optional[list] = None,
        node_fields: Optional[list] = None,
        edge_types: Optional[list] = None,
        edge_fields: Optional[list] = None,
        max_workers: int = 4,
        chunk_size: int = 25
    ):
        self._set_types_and_fields(
            node_types, node_fields, edge_types, edge_fields
        )

        self.base_url = "https://clinicaltrials.gov/api/v2"
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        
        # Cache studies to avoid repeated API calls
        self._studies_cache = {}
        self._chunk_tokens = []
        self._prefetch_chunk_tokens()

    def _prefetch_chunk_tokens(self):
        """
        Fetches the first page and divides the results into chunks for parallel processing
        """
        try:
            # Make initial request to get total count and first page
            params = QUERY_PARAMS.copy()
            url = f"{self.base_url}/studies"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            if not response.content:
                logger.error("Empty response received from ClinicalTrials.gov API")
                return
                
            result = response.json()
            
            # Start with the first page token
            if result.get("nextPageToken"):
                self._chunk_tokens.append(None)  # First page has no token
                
                current_token = result.get("nextPageToken")
                while current_token:
                    self._chunk_tokens.append(current_token)
                    
                    # Fetch next page token
                    params["pageToken"] = current_token
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    result = response.json()
                    current_token = result.get("nextPageToken")
                    
                    # Limit the number of prefetched tokens to avoid overloading
                    if len(self._chunk_tokens) >= 10:
                        break
            else:
                # Only one page of results
                self._chunk_tokens.append(None)
                
            logger.info(f"Prefetched {len(self._chunk_tokens)} page tokens for parallel processing")
            
        except Exception as e:
            logger.error(f"Error prefetching chunk tokens: {str(e)}")
            # Add at least one token to process the first page
            self._chunk_tokens.append(None)

    def _fetch_studies_chunk(self, page_token=None):
        """
        Fetches a chunk of studies from the API.
        
        Args:
            page_token: Token for pagination
            
        Returns:
            List of studies in the chunk
        """
        # Check if we already have this chunk cached
        if page_token in self._studies_cache:
            return self._studies_cache[page_token]
            
        try:
            params = QUERY_PARAMS.copy()
            if page_token:
                params["pageToken"] = page_token
                
            url = f"{self.base_url}/studies"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            if not response.content:
                return []
                
            result = response.json()
            studies = result.get("studies", [])
            
            # Cache the result
            self._studies_cache[page_token] = studies
            
            return studies
            
        except Exception as e:
            logger.error(f"Error fetching studies chunk with token {page_token}: {str(e)}")
            return []

    def _ensure_string(self, value, default="N/A"):
        """Convert any value to a properly formatted string"""
        if value is None:
            return default
        if isinstance(value, str):
            return replace_quote(value)
        return replace_quote(str(value))
        
    def _ensure_int(self, value, default=0):
        """Convert any value to an integer"""
        if value is None or value == "N/A":
            return default
        try:
            if isinstance(value, str) and not value.strip().isdigit():
                return default
            return int(value)
        except (ValueError, TypeError):
            return default
            
    def _ensure_date(self, value, default=None):
        """Convert any value to a date string"""
        if value is None or value == "N/A":
            return default
        # Handle date structure objects
        if isinstance(value, dict) and value.get("date"):
            value = value.get("date")
        # Clean string dates
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return value
        
    def _format_list(self, value_list, default="N/A"):
        """Format a list of values as a comma-separated string"""
        if not value_list:
            return default
        cleaned = [self._ensure_string(v) for v in value_list if v]
        if not cleaned:
            return default
        return ", ".join(cleaned)
        
    def _process_study_nodes(self, study):
        """
        Process a single study and yield all related nodes.
        
        Args:
            study: Study data from the API
            
        Yields:
            Node tuples for all entity types
        """
        if not study.get("protocolSection"):
            return
            
        try:
            protocol = study["protocolSection"]
            _id = protocol.get("identificationModule", {}).get("nctId")
            
            if not _id:
                return
                
            # Study node
            if ClinicalTrialsAdapterNodeType.STUDY in self.node_types:
                _props = self._get_study_props_from_fields(study)
                yield (_id, "study", _props)

            # Sponsor nodes
            if ClinicalTrialsAdapterNodeType.SPONSOR in self.node_types:
                lead_sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor")
                if lead_sponsor and lead_sponsor.get("name"):
                    sponsor_name = self._ensure_string(lead_sponsor.get("name"))
                    agency_class = self._ensure_string(lead_sponsor.get("class"))
                    yield (
                        sponsor_name,
                        "sponsor",
                        {"agency_class": agency_class}
                    )

            # Condition nodes
            if ClinicalTrialsAdapterNodeType.CONDITION in self.node_types:
                conditions = protocol.get("conditionsModule", {}).get("conditions", [])
                for condition in conditions:
                    if condition:
                        condition_name = self._ensure_string(condition)
                        if condition_name:
                            yield (
                                condition_name,
                                "disease",
                                {"name": condition_name}
                            )

            # Intervention nodes
            if ClinicalTrialsAdapterNodeType.INTERVENTION in self.node_types:
                interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
                for intervention in interventions:
                    if intervention and intervention.get("name"):
                        intervention_name = self._ensure_string(intervention.get("name"))
                        intervention_type = self._ensure_string(intervention.get("type"))
                        intervention_desc = self._ensure_string(intervention.get("description"))
                        
                        yield (
                            intervention_name,
                            "intervention",
                            {
                                "name": intervention_name,
                                "type": intervention_type,
                                "description": intervention_desc
                            }
                        )

            # Outcome nodes
            if ClinicalTrialsAdapterNodeType.OUTCOME in self.node_types:
                outcomes_module = protocol.get("outcomesModule", {})
                
                # Process outcomes with proper type handling
                def process_outcome(outcome, outcome_type="primary"):
                    if outcome and outcome.get("measure"):
                        measure = self._ensure_string(outcome.get("measure"))
                        description = self._ensure_string(outcome.get("description"))
                        time_frame = self._ensure_string(outcome.get("timeFrame"))
                        
                        return (
                            measure,
                            "outcome",
                            {
                                "title": measure,
                                "description": description,
                                "time_frame": time_frame,
                                "population": "N/A",
                                "units": "N/A"
                            }
                        )
                    return None
                
                # Primary outcomes
                for outcome in outcomes_module.get("primaryOutcomes", []):
                    outcome_node = process_outcome(outcome, "primary")
                    if outcome_node:
                        yield outcome_node
                
                # Secondary outcomes
                for outcome in outcomes_module.get("secondaryOutcomes", []):
                    outcome_node = process_outcome(outcome, "secondary")
                    if outcome_node:
                        yield outcome_node
        except Exception as e:
            logger.error(f"Error processing study nodes for study {study.get('nctId', 'unknown')}: {str(e)}")
            logger.error(f"Exception details: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())

    def _process_study_edges(self, study):
        """
        Process a single study and yield all related edges.
        
        Args:
            study: Study data from the API
            
        Yields:
            Edge tuples for all relationship types
        """
        if not study.get("protocolSection"):
            return
            
        try:
            protocol = study["protocolSection"]
            _id = protocol.get("identificationModule", {}).get("nctId")
            
            if not _id:
                return
                
            # Sponsor to study edges
            if ClinicalTrialsAdapterEdgeType.SPONSOR_HAS_STUDY in self.edge_types:
                lead_sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor")
                if lead_sponsor and lead_sponsor.get("name"):
                    sponsor_name = self._ensure_string(lead_sponsor.get("name"))
                    # if sponsor_name == "N/A" or sponsor_name=='' or _id == '':
                        # logger.info(f"Sponsor name: {sponsor_name}, Study ID: {_id}")
                    yield (None, sponsor_name, _id, "SPONSOR_HAS_STUDY", {"lead_or_collaborator": "lead"})

            # Study to condition edges
            conditions = protocol.get("conditionsModule", {}).get("conditions", [])
            interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
            
            # Process conditions with proper type handling
            for condition in conditions:
                if condition:
                    condition_name = self._ensure_string(condition)
                    
                    # Study has condition
                    if ClinicalTrialsAdapterEdgeType.STUDY_HAS_CONDITION in self.edge_types:
                        yield (None, _id, condition_name, "STUDY_HAS_CONDITION", {})
                    
                    # Condition has intervention
                    if ClinicalTrialsAdapterEdgeType.CONDITION_HAS_INTERVENTION in self.edge_types:
                        for intervention in interventions:
                            if intervention and intervention.get("name"):
                                intervention_name = self._ensure_string(intervention.get("name"))
                                yield (None, condition_name, intervention_name, "condition_has_intervention", {})

            # Study to intervention edges
            if ClinicalTrialsAdapterEdgeType.STUDY_HAS_INTERVENTION in self.edge_types:
                for intervention in interventions:
                    if intervention and intervention.get("name"):
                        intervention_name = self._ensure_string(intervention.get("name"))
                        yield (None, _id, intervention_name, "study_has_intervention", {})

            # Study to outcome edges
            if ClinicalTrialsAdapterEdgeType.STUDY_HAS_OUTCOME in self.edge_types:
                outcomes_module = protocol.get("outcomesModule", {})
                
                # Primary outcomes
                for outcome in outcomes_module.get("primaryOutcomes", []):
                    if outcome and outcome.get("measure"):
                        measure = self._ensure_string(outcome.get("measure"))
                        yield (None, _id, measure, "study_has_outcome", {"type": "primary"})
                
                # Secondary outcomes
                for outcome in outcomes_module.get("secondaryOutcomes", []):
                    if outcome and outcome.get("measure"):
                        measure = self._ensure_string(outcome.get("measure"))
                        yield (None, _id, measure, "study_has_outcome", {"type": "secondary"})
        except Exception as e:
            logger.error(f"Error processing study edges for study {study.get('nctId', 'unknown')}: {str(e)}")
            logger.error(f"Exception details: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())

    def get_nodes(self):
        """
        Returns a generator of node tuples for node types specified in the
        adapter constructor. Uses parallel processing for better performance.
        """
        logger.info("Generating nodes using parallel processing.")
        from biocypher._create import BioCypherNode
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self._fetch_studies_chunk, token): token 
                for token in self._chunk_tokens
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_token = future_to_chunk[future]
                try:
                    studies = future.result()
                    
                    # Process each study's nodes
                    for study in studies:
                        for node_tuple in self._process_study_nodes(study):
                            # Just yield the tuple directly, don't convert
                            yield node_tuple
                        
                except Exception as e:
                    logger.error(f"Error processing chunk with token {chunk_token}: {str(e)}")

  
    def get_edges(self):
        from biocypher._create import BioCypherEdge
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._fetch_studies_chunk, token) for token in self._chunk_tokens]
            
            for future in concurrent.futures.as_completed(futures):
                studies = future.result()
                for study in studies:
                    for edge_tuple in self._process_study_edges(study):
                        # Explicitly map tuple elements to BioCypherEdge parameters:
                        edge_id, source_id, target_id, edge_type, props = edge_tuple
                        edge = BioCypherEdge(
                            source_id=source_id,  # This must not be None
                            target_id=target_id,  # This must not be None
                            relationship_label=edge_type,
                            relationship_id=edge_id,  # This can be None
                            properties=props
                        )
                        yield edge

    def _set_types_and_fields(
        self, node_types, node_fields, edge_types, edge_fields
    ):
        if node_types:
            self.node_types = node_types
        else:
            self.node_types = [type for type in ClinicalTrialsAdapterNodeType]

        if node_fields:
            self.node_fields = node_fields
        else:
            self.node_fields = [
                field
                for field in chain(
                    ClinicalTrialsAdapterStudyField,
                    ClinicalTrialsAdapterDiseaseField,
                )
            ]

        if edge_types:
            self.edge_types = edge_types
        else:
            self.edge_types = [type for type in ClinicalTrialsAdapterEdgeType]

        if edge_fields:
            self.edge_fields = edge_fields
        else:
            self.edge_fields = [field for field in chain()]

    def _get_study_props_from_fields(self, study):
        """
        Returns a dictionary of properties for a study node, given the selected
        fields.

        Args:
            study: The study (raw API result) to extract properties from.

        Returns:
            A dictionary of properties.
        """
        # Define field type categories
        integer_fields = [ClinicalTrialsAdapterStudyField.ENROLLMENT]
        date_fields = [
            ClinicalTrialsAdapterStudyField.START_DATE,
            ClinicalTrialsAdapterStudyField.COMPLETION_DATE
        ]
        list_fields = [ClinicalTrialsAdapterStudyField.PHASE]
        # All other fields are treated as strings

        props = {}

        for field in self.node_fields:
            if field not in ClinicalTrialsAdapterStudyField:
                continue

            if field == ClinicalTrialsAdapterStudyField.ID:
                continue

            # Extract raw value from the study data
            path = field.value.split("/")
            value = study.get("protocolSection")
            if value:
                for step in path:
                    if value:
                        value = value.get(step)

            # Apply type-specific conversions
            field_name = field.name.lower()

            # 1. Handle integer fields
            if field in integer_fields:
                value = self._ensure_int(value)

            # 2. Handle date fields
            elif field in date_fields:
                value = self._ensure_date(value)

            # 3. Handle list fields
            elif field in list_fields:
                if isinstance(value, list):
                    value = self._format_list(value)
                else:
                    value = self._ensure_string(value)
            
            # 4. Handle any remaining lists (not in specific list_fields)
            elif isinstance(value, list):
                value = self._format_list(value)
            
            # 5. Handle string values (default case)
            else:
                value = self._ensure_string(value)

            # Update the properties dictionary
            props[field_name] = value if value is not None else "N/A"

        return props


def replace_quote(string):
    return string.replace('"', "'")


def replace_newline(string):
    return string.replace("\n", " | ")


if __name__ == "__main__":
    from data.utils import connect_to_mongo
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Process clinical trials data with parallel computation')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads (default: 4)')
    parser.add_argument('--batch-size', type=int, default=25, help='Batch size for processing (default: 25)')
    parser.add_argument('--condition', type=str, help='Filter by condition (e.g., "COPD")')
    args = parser.parse_args()
    
    # Update query params if condition is specified
    if args.condition:
        QUERY_PARAMS["query.cond"] = args.condition
    
    try:
        # Initialize MongoDB connection
        db = connect_to_mongo()
        if not db:
            logger.error("Failed to connect to MongoDB")
            sys.exit(1)
        
        start_time = time.time()
        
        # Initialize adapter with parallel processing configuration
        adapter = ClinicalTrialsAdapter(
            max_workers=args.workers,
            chunk_size=args.batch_size
        )
        
        logger.info(f"Starting parallel processing with {args.workers} workers")
        
        # Process nodes (use a memory-efficient approach)
        node_count = 0
        node_types = {}
        
        for node in adapter.get_nodes():
            node_count += 1
            node_type = node.get_type()
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_count % 1000 == 0:
                logger.info(f"Processed {node_count} nodes so far")
        
        # Process edges (use a memory-efficient approach)
        edge_count = 0
        edge_types = {}
        
        for edge in adapter.get_edges():
            edge_count += 1
            edge_type = edge.get_type()
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            if edge_count % 1000 == 0:
                logger.info(f"Processed {edge_count} edges so far")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print summary
        logger.info(f"Processed {node_count} nodes and {edge_count} edges in {duration:.2f} seconds")
        logger.info(f"Node types: {node_types}")
        logger.info(f"Edge types: {edge_types}")
        logger.info(f"Processing speed: {(node_count + edge_count) / duration:.2f} entities/second")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
