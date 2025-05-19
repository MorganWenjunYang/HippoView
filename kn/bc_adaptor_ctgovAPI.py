# adapt from https://github.com/biocypher/igan/blob/main/igan/adapters/clinicaltrials_adapter.py

import random
import string
from enum import Enum, auto
from itertools import chain
from typing import Optional
from biocypher._logger import logger

logger.debug(f"Loading module {__name__}.")

import requests

QUERY_PARAMS = {
    "format": "json",
    "fields": [
        "NCTId",
        "BriefTitle",
        "OfficialTitle",
        "OverallStatus",
        "BriefSummary",
        "StudyType",
        "Phase",
        "StartDate",
        "CompletionDate",
        "EnrollmentCount",
        "Condition",
        "Intervention",
        "OutcomeMeasures",
        "Sponsor"
    ],
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

        self.base_url = "https://clinicaltrials.gov/api/v2"

        self._studies = self._get_studies(QUERY_PARAMS)

        self._preprocess()

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
        Preprocess raw API results into node and edge types.
        """
        self._conditions = {}
        self._interventions = {}
        self._outcomes = {}
        self._sponsors = {}

        self._study_has_condition_edges = []
        self._study_has_intervention_edges = []
        self._study_has_outcome_edges = []
        self._sponsor_has_study_edges = []
        self._condition_has_intervention_edges = []
        self._condition_has_outcome_edges = []
        self._intervention_has_outcome_edges = []

        for study in self._studies:
            self._preprocess_study(study)

    def _preprocess_study(self, study: dict):
        if not study.get("protocolSection"):
            return

        try:
            _id = study.get("protocolSection").get("identificationModule").get("nctId")
        except AttributeError:
            _id = None

        if not _id:
            return

        study["nctId"] = _id
        protocol = study.get("protocolSection")

        # sponsor
        if ClinicalTrialsAdapterNodeType.SPONSOR in self.node_types:
            try:
                lead = protocol.get("sponsorCollaboratorsModule").get("leadSponsor")
            except AttributeError:
                lead = None

            if lead:
                name = lead.get("name")
                if name not in self._sponsors.keys():
                    self._sponsors.update({
                        name: {
                            "agency_class": lead.get("class") or "N/A",
                        }
                    })

                # sponsor has study edges
                self._sponsor_has_study_edges.append((
                    None,
                    name,
                    _id,
                    "sponsor_has_study",
                    {"lead_or_collaborator": "lead"},
                ))

        # outcomes
        if ClinicalTrialsAdapterNodeType.OUTCOME in self.node_types:
            try:
                primary = protocol.get("outcomesModule").get("primaryOutcomes")
                secondary = protocol.get("outcomesModule").get("secondaryOutcomes")
            except AttributeError:
                primary = None
                secondary = None

            if primary:
                for outcome in primary:
                    self._add_outcome(outcome, "primary", _id)

            if secondary:
                for outcome in secondary:
                    self._add_outcome(outcome, "secondary", _id)

        # interventions
        if ClinicalTrialsAdapterNodeType.INTERVENTION in self.node_types:
            try:
                interventions = protocol.get("armsInterventionsModule").get("interventions")
            except AttributeError:
                interventions = None

            if interventions:
                for intervention in interventions:
                    name = intervention.get("name")
                    if name:
                        if name not in self._interventions.keys():
                            self._interventions.update({
                                name: {
                                    "type": intervention.get("type") or "N/A",
                                    "description": replace_quote(intervention.get("description") or "N/A"),
                                }
                            })

                        # study has intervention edges
                        self._study_has_intervention_edges.append((
                            None,
                            _id,
                            name,
                            "study_has_intervention",
                            {},
                        ))

        # conditions
        if ClinicalTrialsAdapterNodeType.CONDITION in self.node_types:
            try:
                conditions = protocol.get("conditionsModule").get("conditions")
            except AttributeError:
                conditions = None

            if conditions:
                for condition in conditions:
                    if condition not in self._conditions.keys():
                        self._conditions.update({
                            condition: {
                                "name": condition,
                            }
                        })

                    # study has condition edges
                    self._study_has_condition_edges.append((
                        None,
                        _id,
                        condition,
                        "study_has_condition",
                        {},
                    ))

                    # Create condition-intervention edges
                    if interventions:
                        for intervention in interventions:
                            name = intervention.get("name")
                            if name:
                                self._condition_has_intervention_edges.append((
                                    None,
                                    condition,
                                    name,
                                    "condition_has_intervention",
                                    {},
                                ))

    def _add_outcome(self, outcome: dict, outcome_type: str, study_id: str):
        try:
            measure = outcome.get("measure")
            measure = replace_quote(measure)
        except AttributeError:
            measure = None

        if measure:
            if measure not in self._outcomes:
                self._outcomes.update({
                    measure: {
                        "title": measure,
                        "description": replace_quote(outcome.get("description") or "N/A"),
                        "time_frame": outcome.get("timeFrame") or "N/A",
                        "population": "N/A",  # Not available in source data
                        "units": "N/A",  # Not available in source data
                    },
                })

            # study has outcome edges
            self._study_has_outcome_edges.append((
                None,
                study_id,
                measure,
                "study_has_outcome",
                {"type": outcome_type},
            ))

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
                yield (name, "sponsor", props)

        if ClinicalTrialsAdapterNodeType.OUTCOME in self.node_types:
            for measure, props in self._outcomes.items():
                yield (measure, "outcome", props)

        if ClinicalTrialsAdapterNodeType.INTERVENTION in self.node_types:
            for name, props in self._interventions.items():
                yield (name, "intervention", props)

        if ClinicalTrialsAdapterNodeType.CONDITION in self.node_types:
            for name, props in self._conditions.items():
                yield (name, "condition", props)

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

        if ClinicalTrialsAdapterEdgeType.STUDY_HAS_INTERVENTION in self.edge_types:
            yield from self._study_has_intervention_edges

        if ClinicalTrialsAdapterEdgeType.STUDY_HAS_CONDITION in self.edge_types:
            yield from self._study_has_condition_edges

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


def replace_quote(string):
    return string.replace('"', "'")


def replace_newline(string):
    return string.replace("\n", " | ")