import random
import string
from enum import Enum, auto
from itertools import chain
from typing import Optional
from biocypher._logger import logger
import os, sys
import numpy as np
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import connect_to_mongo

logger.debug(f"Loading module {__name__}.")

# Define API query parameters
QUERY_PARAMS = {
    "format": "json",
    "query.cond": "COPD",
    # Add other parameters as needed
}

class ClinicalTrialsAdapterNodeType(Enum):
    """
    Define types of nodes the adapter can provide.
    """
    STUDY = auto()
    CONDITION = auto()
    INTERVENTION = auto()
    SPONSOR = auto()
    OUTCOME = auto()


class ClinicalTrialsAdapterStudyField(Enum):
    """
    Define possible fields the adapter can provide for studies.
    """
    NCT_ID = "nct_id"
    STUDY_TYPE = "study_type"
    STATUS = "status"
    PHASE = "phase"
    START_DATE = "start_date"
    COMPLETION_DATE = "completion_date"
    ENROLLMENT = "enrollment"
    BRIEF_SUMMARY = "brief_summary"
    ALLOCATION = "allocation"
    MASKING = "masking"
    

class ClinicalTrialsAdapterConditionField(Enum):
    """
    Define possible fields the adapter can provide for conditions.
    """
    NAME = "name"


class ClinicalTrialsAdapterInterventionField(Enum):
    """
    Define possible fields the adapter can provide for interventions.
    """
    NAME = "name"


class ClinicalTrialsAdapterOutcomeField(Enum):
    """
    Define possible fields the adapter can provide for outcomes.
    """
    TYPE = "type"
    TITLE = "title"
    DESCRIPTION = "description"
    TIME_FRAME = "time_frame"
    POPULATION = "population"
    UNITS = "units"
    

class ClinicalTrialsAdapterSponsorField(Enum):
    """
    Define possible fields the adapter can provide for sponsors.
    """
    NAME = "name"
    AGENCY_CLASS = "agency_class"
    LEAD_OR_COLLABORATOR = "lead_or_collaborator"


class ClinicalTrialsAdapterEdgeType(Enum):
    """
    Enum for the edge types defined in the schema.
    """
    STUDY_HAS_CONDITION = auto()
    STUDY_HAS_INTERVENTION = auto()
    STUDY_HAS_OUTCOME = auto()
    SPONSOR_HAS_STUDY = auto()
    CONDITION_HAS_INTERVENTION = auto()
    CONDITION_HAS_OUTCOME = auto()
    INTERVENTION_HAS_OUTCOME = auto()


class ClinicalTrialsAdapterSponsorHasStudyEdgeField(Enum):
    """
    Define possible fields the adapter can provide for sponsor-study edges.
    """
    LEAD_OR_COLLABORATOR = "lead_or_collaborator"


class ClinicalTrialAdapterStudyHasOutcomeEdgeField(Enum):
    """
    Define possible fields the adapter can provide for study-outcome edges.
    """
    TYPE = "type"


class ClinicalTrialsAdapter:
    """
    Unified ClinicalTrials BioCypher adapter. Generates nodes and edges for creating a
    knowledge graph with support for MongoDB or ClinicalTrials.gov API as data sources.

    Args:
        use_mongodb: If True, use MongoDB as data source, otherwise use ClinicalTrials.gov API
        node_types: List of node types to include in the result.
        node_fields: List of node fields to include in the result.
        edge_types: List of edge types to include in the result.
        edge_fields: List of edge fields to include in the result.
    """

    def __init__(
        self,
        use_mongodb: bool = True,
        node_types: Optional[list] = None,
        node_fields: Optional[list] = None,
        edge_types: Optional[list] = None,
        edge_fields: Optional[list] = None,
    ):
        self._set_types_and_fields(
            node_types, node_fields, edge_types, edge_fields
        )

        self.use_mongodb = use_mongodb
        
        # Initialize data source
        if self.use_mongodb:
            self._init_mongodb()
        else:
            self._init_api()

        self._preprocess()

    def _init_mongodb(self):
        """Initialize MongoDB connection and fetch study IDs"""
        self._db = connect_to_mongo()["clinical_trials"]
        self._collection = self._db["trialgpt_trials"]
        self._study_ids = list(self._collection.distinct("nct_id"))

    def _init_api(self):
        """Initialize API connection and fetch studies"""
        self.base_url = "https://clinicaltrials.gov/api/v2"
        self._studies = self._get_studies(QUERY_PARAMS)

    def _get_studies(self, query_params):
        """
        Get all studies fitting the parameters from the API.

        Args:
            query_params: Dictionary of query parameters to pass to the API.

        Returns:
            A list of studies (dictionaries).
        """
        url = f"{self.base_url}/studies"
        response = requests.get(url, params=query_params)
        result = response.json()
        # append pages until empty
        while result.get("nextPageToken"):
            query_params["pageToken"] = result.get("nextPageToken")
            response = requests.get(url, params=query_params)
            result.get("studies").extend(response.json().get("studies"))
            result["nextPageToken"] = response.json().get("nextPageToken")

        return result.get("studies")

    def _preprocess(self):
        """
        Preprocess raw data into node and edge types.
        Dispatches to the appropriate preprocessing method based on data source.
        """
        self._studies = {}
        self._sponsors = {}
        self._outcomes = {}
        self._interventions = {}
        self._conditions = {}

        # Initialize edge collections
        self._study_has_condition_edges = []
        self._study_has_intervention_edges = []
        self._study_has_outcome_edges = []
        self._sponsor_has_study_edges = []
        self._condition_has_intervention_edges = []
        self._condition_has_outcome_edges = []
        self._intervention_has_outcome_edges = []

        if self.use_mongodb:
            for study_id in self._study_ids:
                self._preprocess_mongodb_study(study_id)
        else:
            for study in self._studies:
                self._preprocess_api_study(study)

        self._create_additional_edges()

    def _preprocess_mongodb_study(self, study_id):
        """
        Preprocess a study from MongoDB.
        
        Args:
            study_id: The study ID to retrieve and process from MongoDB.
        """
        study = self._collection.find_one({"nct_id": study_id})

        if not study:
            return

        try:
            _id = study.get("nct_id")
            study_type = study.get("study_type")
            status = study.get("status")
            phase = study.get("phase")
            start_date = study.get("start_date")
            completion_date = study.get("completion_date")
            enrollment = study.get("enrollment")
            brief_summary = study.get("brief_summary")
        except AttributeError:
            _id = None

        if not _id:
            return

        # study
        if ClinicalTrialsAdapterNodeType.STUDY in self.node_types:
            self._studies[_id] = {
                "nct_id": _id,
                "study_type": study_type,
                "status": status,
                "phase": phase,
                "start_date": start_date,
                "completion_date": completion_date,
                "enrollment": enrollment,
                "brief_summary": brief_summary,
            }

        # sponsor
        if ClinicalTrialsAdapterNodeType.SPONSOR in self.node_types:
            try:
                sponsors = study.get("sponsors")
                if isinstance(sponsors, float) and np.isnan(sponsors):
                    sponsors = []
            except AttributeError:
                sponsors = []

            for sponsor in sponsors:
                name = sponsor.get("name")
                agency_class = sponsor.get("agency_class")
                lead_or_collaborator = sponsor.get("lead_or_collaborator")

                # add sponsor node to dictionary
                if name and name not in self._sponsors:
                    self._sponsors[name] = {
                        "name": name,
                        "agency_class": agency_class,
                    }
                
                # sponsor to study edges
                self._sponsor_has_study_edges.append(
                    (
                        None,
                        name,  # source: sponsor
                        _id,   # target: study
                        "sponsor_has_study",
                        {"lead_or_collaborator": lead_or_collaborator},
                    )
                )

        # outcomes
        if ClinicalTrialsAdapterNodeType.OUTCOME in self.node_types:
            try:
                outcomes = study.get("outcomes")
                if isinstance(outcomes, float) and np.isnan(outcomes):
                    outcomes = []
            except AttributeError:
                outcomes = []

            for outcome in outcomes:
                type = outcome.get("outcome_type")
                title = outcome.get("title")
                description = outcome.get("description")
                time_frame = outcome.get("time_frame")
                population = outcome.get("population")
                units = outcome.get("units")
                
                # add outcome node to dictionary
                if title and title not in self._outcomes:
                    self._outcomes[title] = {
                        "title": title,
                        "description": description,
                        "time_frame": time_frame,
                        "population": population,
                        "units": units,
                    }
                
                # study to outcome edges
                self._study_has_outcome_edges.append(
                    (
                        None,
                        _id,
                        title,  
                        "study_has_outcome",
                        {'type': type},
                    )
                )

        # interventions
        if ClinicalTrialsAdapterNodeType.INTERVENTION in self.node_types:
            try:
                interventions = study.get("intervention")
                if isinstance(interventions, float) and np.isnan(interventions):
                    interventions = []
            except AttributeError:
                interventions = []

            for intervention in interventions:
                if intervention not in self._interventions.keys():
                    self._interventions[intervention] = {
                        "name": intervention,
                    }

                self._study_has_intervention_edges.append(
                    (
                        None,
                        _id,
                        intervention,
                        "study_has_intervention",
                        {},
                    )
                )
                
        # conditions
        if ClinicalTrialsAdapterNodeType.CONDITION in self.node_types:
            try:
                conditions = study.get("condition")
                if isinstance(conditions, float) and np.isnan(conditions):
                    conditions = []
            except AttributeError:
                conditions = []

            for condition in conditions:
                if condition not in self._conditions.keys():
                    self._conditions[condition] = {
                        "name": condition,
                    }

                self._study_has_condition_edges.append(
                    (
                        None,
                        _id,
                        condition,
                        "study_has_condition",
                        {},
                    )
                )

    def _preprocess_api_study(self, study):
        """
        Preprocess a study from the ClinicalTrials.gov API.
        
        Args:
            study: The raw study data from the API.
        """
        if not study.get("protocolSection"):
            return

        try:
            _id = (
                study.get("protocolSection")
                .get("identificationModule")
                .get("nctId")
            )
        except AttributeError:
            _id = None

        if not _id:
            return

        study["nctId"] = _id
        protocol = study.get("protocolSection")

        # Extract study fields
        if ClinicalTrialsAdapterNodeType.STUDY in self.node_types:
            try:
                study_type = protocol.get("designModule").get("studyType")
            except AttributeError:
                study_type = None

            try:
                status = protocol.get("statusModule").get("overallStatus")
            except AttributeError:
                status = None

            try:
                phase = protocol.get("designModule").get("phases")
                if isinstance(phase, list):
                    phase = ", ".join(phase)
            except AttributeError:
                phase = None

            try:
                start_date = protocol.get("statusModule").get("startDateStruct").get("date")
            except AttributeError:
                start_date = None

            try:
                completion_date = protocol.get("statusModule").get("completionDateStruct").get("date")
            except AttributeError:
                completion_date = None

            try:
                enrollment = protocol.get("designModule").get("enrollmentInfo").get("count")
            except AttributeError:
                enrollment = None

            try:
                brief_summary = protocol.get("descriptionModule").get("briefSummary")
                brief_summary = replace_quote(brief_summary)
            except AttributeError:
                brief_summary = None

            self._studies[_id] = {
                "nct_id": _id,
                "study_type": study_type,
                "status": status,
                "phase": phase,
                "start_date": start_date,
                "completion_date": completion_date,
                "enrollment": enrollment,
                "brief_summary": brief_summary,
            }

        # sponsor
        if ClinicalTrialsAdapterNodeType.SPONSOR in self.node_types:
            try:
                lead = protocol.get("sponsorCollaboratorsModule").get("leadSponsor")
                # other sponsors
                # other_sponsors = protocol.get("sponsorCollaboratorsModule").get("collaborators")
            except AttributeError:
                lead = None

            if lead:
                name = lead.get("name")
                agency_class = lead.get("class")

                if name not in self._sponsors.keys():
                    self._sponsors.update(
                        {
                            name: {
                                "name": name,
                                "agency_class": agency_class,
                            }
                        }
                    )

                # sponsor to study edges
                self._sponsor_has_study_edges.append(
                    (
                        None,
                        name,  # source: sponsor
                        _id,   # target: study
                        "sponsor_has_study",
                        {"lead_or_collaborator": "lead"},
                    )
                )

        # outcomes
        if ClinicalTrialsAdapterNodeType.OUTCOME in self.node_types:
            try:
                primary = protocol.get("outcomesModule").get("primaryOutcomes")
            except AttributeError:
                primary = None

            try:
                secondary = protocol.get("outcomesModule").get("secondaryOutcomes")
            except AttributeError:
                secondary = None

            if primary:
                for outcome in primary:
                    self._add_api_outcome(outcome, _id, "primary")

            if secondary:
                for outcome in secondary:
                    self._add_api_outcome(outcome, _id, "secondary")

        # interventions
        if ClinicalTrialsAdapterNodeType.INTERVENTION in self.node_types:
            try:
                interventions = protocol.get("armsInterventionsModule").get("interventions")
            except AttributeError:
                interventions = None

            if interventions:
                for intervention in interventions:
                    try:
                        name = intervention.get("name")
                    except AttributeError:
                        name = None

                    try:
                        intervention_type = intervention.get("type")
                    except AttributeError:
                        intervention_type = None

                    if name:
                        if name not in self._interventions.keys():
                            self._interventions.update(
                                {
                                    name: {
                                        "name": name,
                                        "type": intervention_type or "N/A",
                                    },
                                }
                            )

                        # study to intervention edges
                        self._study_has_intervention_edges.append(
                            (
                                None,
                                _id,
                                name,
                                "study_has_intervention",
                                {},
                            )
                        )

        # conditions
        if ClinicalTrialsAdapterNodeType.CONDITION in self.node_types:
            try:
                conditions = protocol.get("conditionsModule").get("conditions")
            except AttributeError:
                conditions = None

            if conditions:
                for condition in conditions:
                    if condition not in self._conditions.keys():
                        self._conditions.update(
                            {condition: {"name": condition}}
                        )

                    # study to condition edges
                    self._study_has_condition_edges.append(
                        (
                            None,
                            _id,
                            condition,
                            "study_has_condition",
                            {},
                        )
                    )

    def _add_api_outcome(self, outcome, study_id, outcome_type):
        """
        Process an outcome from the API data and add it to the outcomes dictionary.
        
        Args:
            outcome: The outcome data from the API.
            study_id: The ID of the study this outcome belongs to.
            outcome_type: The type of outcome (primary or secondary).
        """
        try:
            measure = outcome.get("measure")
            measure = replace_quote(measure)
        except AttributeError:
            measure = None

        try:
            time_frame = outcome.get("timeFrame")
        except AttributeError:
            time_frame = None

        try:
            description = outcome.get("description")
            description = replace_quote(description)
        except AttributeError:
            description = None

        if measure:
            if measure not in self._outcomes:
                self._outcomes.update(
                    {
                        measure: {
                            "title": measure,
                            "time_frame": time_frame or "N/A",
                            "description": description or "N/A",
                        },
                    }
                )
            
            # Add the study to outcome edge
            self._study_has_outcome_edges.append(
                (
                    None,
                    study_id,
                    measure,
                    "study_has_outcome",
                    {"type": outcome_type},
                )
            )

    def get_nodes(self):
        """
        Returns a generator of node tuples for node types specified in the
        adapter constructor.
        """
        logger.info("Generating nodes.")

        if ClinicalTrialsAdapterNodeType.STUDY in self.node_types:
            for study_id, study_data in self._studies.items():
                yield (study_id, "study", study_data)

        if ClinicalTrialsAdapterNodeType.SPONSOR in self.node_types:
            for name, props in self._sponsors.items():
                yield (name, "sponsor", props)

        if ClinicalTrialsAdapterNodeType.OUTCOME in self.node_types:
            for title, props in self._outcomes.items():
                yield (title, "outcome", props)

        if ClinicalTrialsAdapterNodeType.INTERVENTION in self.node_types:
            for name, props in self._interventions.items():
                yield (name, "intervention", props)

        if ClinicalTrialsAdapterNodeType.CONDITION in self.node_types:
            for name, props in self._conditions.items():
                yield (name, "condition", props)

    def get_edges(self):
        """
        Returns a generator of edge tuples for edge types specified in the
        adapter constructor.
        """
        logger.info("Generating edges.")

        if ClinicalTrialsAdapterEdgeType.STUDY_HAS_CONDITION in self.edge_types:
            yield from self._study_has_condition_edges

        if ClinicalTrialsAdapterEdgeType.STUDY_HAS_INTERVENTION in self.edge_types:
            yield from self._study_has_intervention_edges

        if ClinicalTrialsAdapterEdgeType.STUDY_HAS_OUTCOME in self.edge_types:
            yield from self._study_has_outcome_edges

        if ClinicalTrialsAdapterEdgeType.SPONSOR_HAS_STUDY in self.edge_types:
            yield from self._sponsor_has_study_edges
            
        if ClinicalTrialsAdapterEdgeType.CONDITION_HAS_INTERVENTION in self.edge_types:
            yield from self._condition_has_intervention_edges
            
        if ClinicalTrialsAdapterEdgeType.CONDITION_HAS_OUTCOME in self.edge_types:
            yield from self._condition_has_outcome_edges
            
        if ClinicalTrialsAdapterEdgeType.INTERVENTION_HAS_OUTCOME in self.edge_types:
            yield from self._intervention_has_outcome_edges

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
                    ClinicalTrialsAdapterConditionField,
                    ClinicalTrialsAdapterInterventionField,
                    ClinicalTrialsAdapterOutcomeField,
                    ClinicalTrialsAdapterSponsorField,
                )
            ]

        if edge_types:
            self.edge_types = edge_types
        else:
            self.edge_types = [type for type in ClinicalTrialsAdapterEdgeType]

        if edge_fields:
            self.edge_fields = edge_fields
        else:
            self.edge_fields = [field for field in chain(
                ClinicalTrialsAdapterSponsorHasStudyEdgeField,
                ClinicalTrialAdapterStudyHasOutcomeEdgeField,
            )]

    def _create_additional_edges(self):
        """
        Create additional edges between entities based on existing relationships
        """
        # For each study, create connections between its conditions and interventions
        for edge in self._study_has_condition_edges:
            study_id = edge[1]   # study id (source)
            condition = edge[2]  # condition name (target)
            
            # Find all interventions for this study
            for int_edge in self._study_has_intervention_edges:
                if int_edge[1] == study_id:  # if same study
                    intervention = int_edge[2]  # intervention name (target)
                    
                    # Create condition_has_intervention edge
                    self._condition_has_intervention_edges.append(
                        (
                            None,
                            condition,     # source: condition
                            intervention,  # target: intervention
                            "condition_has_intervention",
                            {},
                        )
                    )
            
            # Find all outcomes for this study
            for out_edge in self._study_has_outcome_edges:
                if out_edge[1] == study_id:  # if same study (study is source)
                    outcome = out_edge[2]    # outcome name (target)
                    
                    # Create condition_has_outcome edge
                    self._condition_has_outcome_edges.append(
                        (
                            None,
                            condition,  # source: condition
                            outcome,    # target: outcome
                            "condition_has_outcome",
                            {},
                        )
                    )
        
        # For each study, create connections between its interventions and outcomes
        for edge in self._study_has_intervention_edges:
            study_id = edge[1]      # study id (source)
            intervention = edge[2]  # intervention name (target)
            
            # Find all outcomes for this study
            for out_edge in self._study_has_outcome_edges:
                if out_edge[1] == study_id:  # if same study (study is source)
                    outcome = out_edge[2]    # outcome name (target)
                    
                    # Create intervention_has_outcome edge
                    self._intervention_has_outcome_edges.append(
                        (
                            None,
                            intervention,  # source: intervention
                            outcome,       # target: outcome
                            "intervention_has_outcome",
                            {},
                        )
                    )


def replace_quote(string):
    if not isinstance(string, str):
        return string
    return string.replace('"', "'")


def replace_newline(string):
    if not isinstance(string, str):
        return string
    return string.replace("\n", " | ")


if __name__ == "__main__":
    # Example use with MongoDB
    # mongo_adapter = ClinicalTrialsAdapter(use_mongodb=True)
    # print(f"MongoDB adapter: {len(mongo_adapter._studies)} studies")
    
    # Example use with API
    api_adapter = ClinicalTrialsAdapter(use_mongodb=False)
    print(f"API adapter: {len(api_adapter._studies)} studies") 