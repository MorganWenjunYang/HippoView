# adapt from https://github.com/biocypher/igan/blob/main/igan/adapters/clinicaltrials_adapter.py

import random
import string
from enum import Enum, auto
from itertools import chain
from typing import Optional
from biocypher._logger import logger
import os, sys
import numpy as np
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

class ClinicalTrialsMGDBAdapter:
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

        self._study_ids = list(self._collection.distinct("nct_id"))

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
        self._study_has_condition_edges = []
        self._study_has_intervention_edges = []
        self._study_has_outcome_edges = []
        self._sponsor_has_study_edges = []
        self._condition_has_intervention_edges = []
        self._condition_has_outcome_edges = []
        self._intervention_has_outcome_edges = []

        for study_id in self._study_ids:
            self._preprocess_study(study_id)

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

        # study["nct_id"] = _id
        # study["study_type"] = study_type
        # study["status"] = status
        # study["phase"] = phase
        # study["start_date"] = start_date
        # study["completion_date"] = completion_date
        # study["enrollment"] = enrollment
        # study["brief_summary"] = brief_summary  

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
                        # "lead_or_collaborator": lead_or_collaborator should be in edge
                    }
                
                # sponsor to study edges (correct direction per schema)
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
                type = outcome.get("outcome_type") #should be in edge
                title = outcome.get("title")
                description = outcome.get("description")
                time_frame = outcome.get("time_frame")
                population = outcome.get("population")
                units = outcome.get("units")
                
                # add sponsor node to dictionary
                if title and title not in self._outcomes:
                    self._outcomes[title] = {
                        "title": title,
                        "description": description,
                        "time_frame": time_frame,
                        "population": population,
                        "units": units,
                    }
                
                # sponsor to study edges (correct direction per schema)
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
                print(interventions)
                print(study)
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
                
                # self._intervention_has_outcome_edges.append(
                #     (
                #         None,
                #         intervention,
                #         outcome,
                #         "intervention_has_outcome",
                #         {},
                #     )
                # )
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

                # self._condition_has_intervention_edges.append(
                #     (
                #         None,
                #         condition,
                #         intervention,
                #         "condition_has_intervention",
                #         {},
                #     )
                # )
                # self._condition_has_outcome_edges.append(
                #     (
                #         None,
                #         condition,
                #         outcome,
                #         "condition_has_outcome",
                #         {},
                #     )
                # )


    def get_nodes(self):
        """
        Returns a generator of node tuples for node types specified in the
        adapter constructor.
        """

        logger.info("Generating nodes.")

        if ClinicalTrialsAdapterNodeType.STUDY in self.node_types:
            for study_id, study_data in self._studies.items():
                _props = study_data
                yield (study_id, "study", _props)

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
            condition = edge[3]  # condition name (target)
            study_id = edge[2]   # study id (source)
            
            # Find all interventions for this study
            for int_edge in self._study_has_intervention_edges:
                if int_edge[2] == study_id:  # if same study
                    intervention = int_edge[3]  # intervention name (target)
                    
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
            intervention = edge[3]  # intervention name (target)
            study_id = edge[2]      # study id (source)
            
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
    adapter = ClinicalTrialsMGDBAdapter()
    print(adapter._studies)
    print(len(adapter._studies), "studies in mongodb")
