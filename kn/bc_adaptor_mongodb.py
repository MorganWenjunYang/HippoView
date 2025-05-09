# adapt from https://github.com/biocypher/igan/blob/main/igan/adapters/clinicaltrials_adapter.py

import random
import string
from enum import Enum, auto
from itertools import chain
from typing import Optional
from biocypher._logger import logger
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import connect_to_mongo

logger.debug(f"Loading module {__name__}.")

import requests

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

    ID = "identificationModule/nctId"
    BRIEF_TITLE = "identificationModule/briefTitle"
    OFFICIAL_TITLE = "identificationModule/officialTitle"
    STATUS = "statusModule/overallStatus"
    BRIEF_SUMMARY = "descriptionModule/briefSummary"
    TYPE = "designModule/studyType"
    ALLOCATION = "designModule/designInfo/allocation"
    PHASES = "designModule/phases"
    MODEL = "designModule/designInfo/interventionModel"
    PRIMARY_PURPOSE = "designModule/designInfo/primaryPurpose"
    NUMBER_OF_PATIENTS = "designModule/enrollmentInfo/count"
    ELIGIBILITY_CRITERIA = "eligibilityModule/eligibilityCriteria"
    HEALTHY_VOLUNTEERS = "eligibilityModule/healthyVolunteers"
    SEX = "eligibilityModule/sex"
    MINIMUM_AGE = "eligibilityModule/minimumAge"
    MAXIMUM_AGE = "eligibilityModule/maximumAge"
    STANDARDISED_AGES = "eligibilityModule/stdAges"

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
    Define possible fields the adapter can provide for diseases.
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

    CONDITION_HAS_STUDY = auto()
    INTERVENTION_HAS_STUDY = auto()
    STUDY_HAS_OUTCOME = auto()
    SPONSOR_HAS_STUDY = auto()
    CONDITION_HAS_INTERVENTION = auto()
    CONDITION_HAS_OUTCOME = auto()
    INTERVENTION_HAS_OUTCOME = auto()


# class ClinicalTrialsAdapterProteinProteinEdgeField(Enum):
#     """
#     Define possible fields the adapter can provide for protein-protein edges.
#     """

#     INTERACTION_TYPE = "interaction_type"
#     INTERACTION_SOURCE = "interaction_source"


# class ClinicalTrialsAdapterProteinDiseaseEdgeField(Enum):
#     """
#     Define possible fields the adapter can provide for protein-disease edges.
#     """

#     ASSOCIATION_TYPE = "association_type"
#     ASSOCIATION_SOURCE = "association_source"


class ClinicalTrialsAdapter:
    """
    ClinicalTrials BioCypher adapter. Generates nodes and edges for creating a
    knowledge graph.

    Args:
        node_types: List of node types to include in the result.
        node_fields: List of node fields to include in the result.
        edge_types: List of edge types to include in the result.
        edge_fields: List of edge fields to include in the result.
    """

    def __init__(
        self,
        node_types: Optional[list] = None,
        node_fields: Optional[list] = None,
        edge_types: Optional[list] = None,
        edge_fields: Optional[list] = None,
    ):
        self._set_types_and_fields(
            node_types, node_fields, edge_types, edge_fields
        )

        # self.base_url = "https://clinicaltrials.gov/api/v2"

        # self._studies = self._get_studies(QUERY_PARAMS)

        self._db = connect_to_mongo()["clinical_trials"]
        self._collection = self._db["trialgpt_trials"]

        self._studies = list(self._collection.distinct("nct_id"))

        self._preprocess()

    def _preprocess(self):
        """
        Preprocess raw API results into node and edge types.
        """
        self._studies = {}
        self._sponsors = {}
        self._outcomes = {}
        self._interventions = {}
        self._conditions = {}

        # Updated edge collections based on schema
        self._condition_has_study_edges = []
        self._intervention_has_study_edges = []
        self._study_has_outcome_edges = []
        self._sponsor_has_study_edges = []
        self._condition_has_intervention_edges = []
        self._condition_has_outcome_edges = []
        self._intervention_has_outcome_edges = []

        for study in self._studies:
            self._preprocess_study(study)

        self._create_additional_edges()

    def _preprocess_study(self, study: str):
        study = self._collection.find_one({"nct_id": study})

        if not study:
            return

        try:
            _id = (
                study.get("nct_id")
            )
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

        study["nct_id"] = _id
        study["study_type"] = study_type
        study["status"] = status
        study["phase"] = phase
        study["start_date"] = start_date
        study["completion_date"] = completion_date
        study["enrollment"] = enrollment
        study["brief_summary"] = brief_summary  

        # sponsor
        if ClinicalTrialsAdapterNodeType.SPONSOR in self.node_types:
            try:
                lead = study.get("sponsors").get(
                    "leadSponsor"
                )
            except AttributeError:
                lead = None

            if lead:
                name = lead.get("name")

                if name not in self._sponsors.keys():
                    self._sponsors.update(
                        {
                            name: {
                                "class": lead.get("class"),
                            }
                        }
                    )

                # sponsor to study edges (correct direction per schema)
                self._sponsor_has_study_edges.append(
                    (
                        None,
                        name,  # source: sponsor
                        _id,   # target: study
                        "sponsor_has_study",
                        {},
                    )
                )

        # outcomes
        if ClinicalTrialsAdapterNodeType.OUTCOME in self.node_types:
            try:
                primary = protocol.get("outcomesModule").get("primaryOutcomes")
            except AttributeError:
                primary = None

            try:
                secondary = protocol.get("outcomesModule").get(
                    "secondaryOutcomes"
                )
            except AttributeError:
                secondary = None

            if primary:
                for outcome in primary:
                    self._add_outcome(outcome, True, _id)

            if secondary:
                for outcome in secondary:
                    self._add_outcome(outcome, False, _id)

        # interventions
        if ClinicalTrialsAdapterNodeType.INTERVENTION in self.node_types:
            try:
                interventions = protocol.get("armsInterventionsModule").get(
                    "interventions"
                )
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

                    try:
                        description = intervention.get("description")
                        description = replace_quote(description)
                        description = replace_newline(description)
                    except AttributeError:
                        description = None

                    try:
                        mapped_names = intervention.get(
                            "interventionMappedName"
                        )
                    except AttributeError:
                        mapped_names = None

                    if name:
                        if name not in self._interventions.keys():
                            self._interventions.update(
                                {
                                    name: {
                                        "type": intervention_type or "N/A",
                                        "description": description or "N/A",
                                        "mapped_names": mapped_names or "N/A",
                                    },
                                }
                            )

                        # intervention to study edges
                        self._intervention_has_study_edges.append(
                            (
                                None,
                                name,  # source: intervention
                                _id,   # target: study
                                "intervention_has_study",
                                {"description": description or "N/A"},
                            )
                        )

        # conditions
        if ClinicalTrialsAdapterNodeType.CONDITION in self.node_types:
            try:
                conditions = protocol.get("conditionsModule").get("conditions")
            except AttributeError:
                conditions = None

            try:
                keywords = protocol.get("conditionsModule").get("keywords")
            except AttributeError:
                keywords = []

            if conditions:
                for condition in conditions:
                    if condition not in self._diseases.keys():
                        self._diseases.update(
                            {condition: {"keywords": keywords}}
                        )
                    else:
                        if keywords:
                            if self._diseases[condition]["keywords"]:
                                self._diseases[condition]["keywords"].extend(
                                    keywords
                                )
                            else:
                                self._diseases[condition]["keywords"] = keywords

                    # condition to study edges
                    self._condition_has_study_edges.append(
                        (
                            None,
                            condition,  # source: condition 
                            _id,        # target: study
                            "condition_has_study",
                            {},
                        )
                    )

    def _add_outcome(self, outcome: dict, primary: bool, study_id: str = None):
        """
        Add an outcome to the outcomes dictionary and create corresponding edges if study_id is provided.
        
        Args:
            outcome: The outcome dictionary from the API
            primary: Whether this is a primary outcome
            study_id: The study ID to create edges for
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
                            "primary": primary,
                            "time_frame": time_frame or "N/A",
                            "description": description or "N/A",
                        },
                    }
                )
            
            # Create study_has_outcome edge if study_id is provided
            if study_id:
                self._study_has_outcome_edges.append(
                    (
                        None,
                        study_id,  # source: study
                        measure,   # target: outcome
                        "study_has_outcome",
                        {"primary": primary},
                    )
                )

    def get_nodes(self):
        """
        Returns a generator of node tuples for node types specified in the
        adapter constructor.
        """

        logger.info("Generating nodes.")

        if ClinicalTrialsAdapterNodeType.STUDY in self.node_types:
            for study in self._studies:
                if not study.get("nctId"):
                    continue

                _props = self._get_study_props_from_fields(study)

                yield (study.get("nctId"), "clinical trial", _props)

        if ClinicalTrialsAdapterNodeType.SPONSOR in self.node_types:
            for name, props in self._sponsors.items():
                yield (name, "Sponsor", props)

        if ClinicalTrialsAdapterNodeType.OUTCOME in self.node_types:
            for measure, props in self._outcomes.items():
                yield (measure, "Outcome", props)

        if ClinicalTrialsAdapterNodeType.INTERVENTION in self.node_types:
            for name, props in self._interventions.items():
                yield (name, "Intervention", props)

        if ClinicalTrialsAdapterNodeType.CONDITION in self.node_types:
            for name, props in self._diseases.items():
                yield (name, "Condition", props)

    def _get_study_props_from_fields(self, study):
        """
        Returns a dictionary of properties for a study node, given the selected
        fields.

        Args:
            study: The study (raw API result) to extract properties from.

        Returns:
            A dictionary of properties.
        """

        props = {}

        for field in self.node_fields:
            if field not in ClinicalTrialsAdapterStudyField:
                continue

            if field == ClinicalTrialsAdapterStudyField.ID:
                continue

            path = field.value.split("/")
            value = study.get("protocolSection")

            if value:
                for step in path:
                    if value:
                        value = value.get(step)

            if isinstance(value, list):
                value = [replace_quote(v) for v in value]
            elif isinstance(value, str):
                value = replace_quote(value)

            props.update({field.name.lower(): value or "N/A"})

        return props

    def get_edges(self):
        """
        Returns a generator of edge tuples for edge types specified in the
        adapter constructor.
        """

        logger.info("Generating edges.")

        if ClinicalTrialsAdapterEdgeType.CONDITION_HAS_STUDY in self.edge_types:
            yield from self._condition_has_study_edges

        if ClinicalTrialsAdapterEdgeType.INTERVENTION_HAS_STUDY in self.edge_types:
            yield from self._intervention_has_study_edges

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
            self.edge_fields = [field for field in chain()]

    def _create_additional_edges(self):
        """
        Create additional edges between entities based on existing relationships
        """
        # For each study, create connections between its conditions and interventions
        for edge in self._condition_has_study_edges:
            condition = edge[2]  # condition name (source)
            study_id = edge[3]   # study id (target)
            
            # Find all interventions for this study
            for int_edge in self._intervention_has_study_edges:
                if int_edge[3] == study_id:  # if same study
                    intervention = int_edge[2]  # intervention name (source)
                    
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
        for edge in self._intervention_has_study_edges:
            intervention = edge[2]  # intervention name (source)
            study_id = edge[3]      # study id (target)
            
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
    return string.replace('"', "'")


def replace_newline(string):
    return string.replace("\n", " | ")


if __name__ == "__main__":
    from data.utils import connect_to_mongo
    adapter = ClinicalTrialsAdapter()
    print(adapter._studies)
    print(len(adapter._studies), "studies in mongodb")
