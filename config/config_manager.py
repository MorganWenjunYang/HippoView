"""
Configuration Manager for HippoView Clinical Trial Filtering

This module provides centralized configuration management for controlling
what clinical trials are included in the knowledge graph and vector database.
"""

import yaml
import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class APIFilters:
    """Configuration for API-level filtering"""
    condition: Optional[str] = None
    advanced_filters: Dict[str, Any] = field(default_factory=dict)
    location_filters: Dict[str, List[str]] = field(default_factory=dict)
    date_filters: Dict[str, Optional[str]] = field(default_factory=dict)
    sponsor_filters: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class PostFilters:
    """Configuration for post-processing filters"""
    requirements: Dict[str, Any] = field(default_factory=dict)
    content_filters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SamplingConfig:
    """Configuration for sampling strategies"""
    max_trials: Optional[int] = None
    sampling_strategy: str = "random"
    trials_per_condition: Optional[int] = None
    random_seed: int = 42

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    primary: str = "api"
    fallback: Optional[str] = None
    api_config: Dict[str, Any] = field(default_factory=dict)
    mongodb_config: Dict[str, Any] = field(default_factory=dict)

class TrialFilterConfig:
    """
    Central configuration manager for clinical trial filtering.
    
    This class loads and manages configuration from YAML files and provides
    methods to build query parameters for different data sources.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file.
                        If None, uses default config.
        """
        self.config_path = config_path or "kn/trial_filter_config.yaml"
        self.config = self._load_config()
        
        # Parse configuration sections
        self.api_filters = self._parse_api_filters()
        self.post_filters = self._parse_post_filters()
        self.sampling = self._parse_sampling_config()
        self.data_source = self._parse_data_source_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config file {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'api_filters': {
                'condition': None,
                'advanced_filters': {'study_type': 'INTERVENTIONAL'},
                'location_filters': {},
                'date_filters': {},
                'sponsor_filters': {}
            },
            'post_filters': {
                'requirements': {'require_brief_summary': True},
                'content_filters': {}
            },
            'sampling': {
                'max_trials': None,
                'sampling_strategy': 'random',
                'random_seed': 42
            },
            'data_source': {
                'primary': 'api',
                'api_config': {
                    'base_url': 'https://clinicaltrials.gov/api/v2',
                    'max_workers': 4,
                    'chunk_size': 100
                }
            }
        }
    
    def _parse_api_filters(self) -> APIFilters:
        """Parse API filters from configuration."""
        api_config = self.config.get('api_filters', {})
        return APIFilters(
            condition=api_config.get('condition'),
            advanced_filters=api_config.get('advanced_filters', {}),
            location_filters=api_config.get('location_filters', {}),
            date_filters=api_config.get('date_filters', {}),
            sponsor_filters=api_config.get('sponsor_filters', {})
        )
    
    def _parse_post_filters(self) -> PostFilters:
        """Parse post-processing filters from configuration."""
        post_config = self.config.get('post_filters', {})
        return PostFilters(
            requirements=post_config.get('requirements', {}),
            content_filters=post_config.get('content_filters', {})
        )
    
    def _parse_sampling_config(self) -> SamplingConfig:
        """Parse sampling configuration."""
        sampling_config = self.config.get('sampling', {})
        return SamplingConfig(
            max_trials=sampling_config.get('max_trials'),
            sampling_strategy=sampling_config.get('sampling_strategy', 'random'),
            trials_per_condition=sampling_config.get('trials_per_condition'),
            random_seed=sampling_config.get('random_seed', 42)
        )
    
    def _parse_data_source_config(self) -> DataSourceConfig:
        """Parse data source configuration."""
        ds_config = self.config.get('data_source', {})
        return DataSourceConfig(
            primary=ds_config.get('primary', 'api'),
            fallback=ds_config.get('fallback'),
            api_config=ds_config.get('api_config', {}),
            mongodb_config=ds_config.get('mongodb_config', {})
        )
    
    def build_api_query_params(self) -> Dict[str, Any]:
        """
        Build query parameters for ClinicalTrials.gov API based on configuration.
        
        Returns:
            Dictionary of query parameters for the API
        """
        params = {
            "format": "json",
            "pageSize": self.data_source.api_config.get('chunk_size', 100),
            "countTotal": "true"
        }
        
        # Add condition filter
        if self.api_filters.condition:
            params["query.cond"] = self.api_filters.condition
        
        # Build advanced filter string
        advanced_parts = []
        
        # Study type
        study_type = self.api_filters.advanced_filters.get('study_type')
        if study_type:
            advanced_parts.append(f"AREA[StudyType]{study_type}")
        
        # Recruitment status
        status = self.api_filters.advanced_filters.get('recruitment_status')
        if status:
            advanced_parts.append(f"AREA[OverallStatus]{status}")
        
        # Age group
        age_group = self.api_filters.advanced_filters.get('age_group')
        if age_group:
            advanced_parts.append(f"AREA[StdAge]{age_group}")
        
        # Gender
        gender = self.api_filters.advanced_filters.get('gender')
        if gender:
            advanced_parts.append(f"AREA[Gender]{gender}")
        
        # Study phase
        phase = self.api_filters.advanced_filters.get('study_phase')
        if phase:
            advanced_parts.append(f"AREA[Phase]{phase}")
        
        # Location filters
        countries = self.api_filters.location_filters.get('countries', [])
        for country in countries:
            advanced_parts.append(f"AREA[LocationCountry]{country}")
        
        states = self.api_filters.location_filters.get('states', [])
        for state in states:
            advanced_parts.append(f"AREA[LocationState]{state}")
        
        cities = self.api_filters.location_filters.get('cities', [])
        for city in cities:
            advanced_parts.append(f"AREA[LocationCity]{city}")
        
        # Date filters
        start_from = self.api_filters.date_filters.get('start_date_from')
        if start_from:
            advanced_parts.append(f"AREA[StartDate]RANGE[{start_from}, MAX]")
        
        start_to = self.api_filters.date_filters.get('start_date_to')
        if start_to:
            advanced_parts.append(f"AREA[StartDate]RANGE[MIN, {start_to}]")
        
        # Sponsor type filters - use OR logic for multiple sponsor types
        sponsor_types = self.api_filters.sponsor_filters.get('lead_sponsor_types', [])
        if sponsor_types:
            if len(sponsor_types) == 1:
                advanced_parts.append(f"AREA[LeadSponsorClass]{sponsor_types[0]}")
            else:
                # Multiple sponsor types should be OR'd together
                sponsor_filters = [f"AREA[LeadSponsorClass]{sponsor_type}" for sponsor_type in sponsor_types]
                advanced_parts.append(f"({' OR '.join(sponsor_filters)})")
        
        # Combine advanced filters
        if advanced_parts:
            params["filter.advanced"] = " AND ".join(advanced_parts)
        
        # Add field specification
        fields = [
            "NCTId", "BriefTitle", "OfficialTitle",
            "StatusModule", "DesignModule", "DescriptionModule",
            "ConditionsModule", "ArmsInterventionsModule", "OutcomesModule",
            "SponsorCollaboratorsModule", "EligibilityModule", "ContactsLocationsModule"
        ]
        params["fields"] = ",".join(fields)
        
        return params
    
    def build_mongodb_query(self) -> Dict[str, Any]:
        """
        Build MongoDB query filters based on configuration.
        
        Returns:
            Dictionary of MongoDB query filters
        """
        query = {}
        
        # Add condition filter
        if self.api_filters.condition:
            query["condition"] = {"$regex": self.api_filters.condition, "$options": "i"}
        
        # Add enrollment filters
        min_enrollment = self.post_filters.requirements.get('min_enrollment')
        max_enrollment = self.post_filters.requirements.get('max_enrollment')
        
        if min_enrollment is not None or max_enrollment is not None:
            enrollment_filter = {}
            if min_enrollment is not None:
                enrollment_filter["$gte"] = min_enrollment
            if max_enrollment is not None:
                enrollment_filter["$lte"] = max_enrollment
            query["enrollment"] = enrollment_filter
        
        # Add study type filter
        study_type = self.api_filters.advanced_filters.get('study_type')
        if study_type:
            query["study_type"] = study_type
        
        # Add brief summary requirement
        if self.post_filters.requirements.get('require_brief_summary'):
            query["brief_summary"] = {"$ne": None, "$ne": ""}
        
        # Add custom MongoDB filters from config
        additional_filters = self.data_source.mongodb_config.get('additional_filters', {})
        query.update(additional_filters)
        
        return query
    
    def should_include_trial(self, trial_data: Dict[str, Any]) -> bool:
        """
        Check if a trial should be included based on post-processing filters.
        
        Args:
            trial_data: Trial data dictionary
            
        Returns:
            True if trial should be included, False otherwise
        """
        # Check enrollment requirements
        enrollment = trial_data.get('enrollment')
        min_enrollment = self.post_filters.requirements.get('min_enrollment')
        max_enrollment = self.post_filters.requirements.get('max_enrollment')
        
        if min_enrollment is not None and (enrollment is None or enrollment < min_enrollment):
            return False
        if max_enrollment is not None and (enrollment is None or enrollment > max_enrollment):
            return False
        
        # Check brief summary requirement
        if self.post_filters.requirements.get('require_brief_summary'):
            brief_summary = trial_data.get('brief_summary') or trial_data.get('description')
            if not brief_summary or brief_summary.strip() == "":
                return False
        
        # Check outcomes requirement
        if self.post_filters.requirements.get('require_outcomes'):
            outcomes = trial_data.get('outcomes') or []
            if not outcomes:
                return False
        
        # Check interventions requirement  
        if self.post_filters.requirements.get('require_interventions'):
            interventions = trial_data.get('interventions') or trial_data.get('intervention')
            if not interventions:
                return False
        
        # Check required keywords
        required_keywords = self.post_filters.content_filters.get('required_keywords', [])
        if required_keywords:
            text_fields = [
                trial_data.get('brief_title', ''),
                trial_data.get('brief_summary', ''),
                str(trial_data.get('conditions', [])),
                str(trial_data.get('interventions', []))
            ]
            combined_text = ' '.join(text_fields).lower()
            
            for keyword in required_keywords:
                if keyword.lower() not in combined_text:
                    return False
        
        # Check excluded keywords
        excluded_keywords = self.post_filters.content_filters.get('excluded_keywords', [])
        if excluded_keywords:
            text_fields = [
                trial_data.get('brief_title', ''),
                trial_data.get('brief_summary', ''),
                str(trial_data.get('conditions', [])),
                str(trial_data.get('interventions', []))
            ]
            combined_text = ' '.join(text_fields).lower()
            
            for keyword in excluded_keywords:
                if keyword.lower() in combined_text:
                    return False
        
        # Check sponsor filters
        exclude_sponsors = self.api_filters.sponsor_filters.get('exclude_sponsors', [])
        include_sponsors = self.api_filters.sponsor_filters.get('include_sponsors', [])
        
        trial_sponsors = trial_data.get('sponsors', [])
        if isinstance(trial_sponsors, list):
            sponsor_names = [s.get('name', '') if isinstance(s, dict) else str(s) for s in trial_sponsors]
        else:
            sponsor_names = [str(trial_sponsors)]
        
        # Check exclude list
        for sponsor_name in sponsor_names:
            if any(exclude in sponsor_name for exclude in exclude_sponsors):
                return False
        
        # Check include list (if specified, only these sponsors are allowed)
        if include_sponsors:
            included = False
            for sponsor_name in sponsor_names:
                if any(include in sponsor_name for include in include_sponsors):
                    included = True
                    break
            if not included:
                return False
        
        return True
    
    # Vector DB configuration removed - vector DB will be populated from MongoDB
    
    def get_knowledge_graph_config(self) -> Dict[str, Any]:
        """Get knowledge graph configuration."""
        return self.config.get('knowledge_graph', {})
    
    def get_caching_config(self) -> Dict[str, Any]:
        """Get caching configuration."""
        return self.config.get('caching', {})
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        
        # Re-parse configuration sections
        self.api_filters = self._parse_api_filters()
        self.post_filters = self._parse_post_filters()
        self.sampling = self._parse_sampling_config()
        self.data_source = self._parse_data_source_config()
        
        logger.info("Configuration updated successfully")
    
    def save_config(self, output_path: str = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save the configuration. If None, uses current config_path.
        """
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {e}")


def load_trial_config(config_path: str = None) -> TrialFilterConfig:
    """
    Convenience function to load trial configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TrialFilterConfig instance
    """
    return TrialFilterConfig(config_path)


if __name__ == "__main__":
    # Example usage
    config = load_trial_config()
    
    print("API Query Parameters:")
    print(config.build_api_query_params())
    
    print("\nMongoDB Query:")
    print(config.build_mongodb_query())
    
    print("\nKnowledge Graph Config:")
    print(config.get_knowledge_graph_config()) 