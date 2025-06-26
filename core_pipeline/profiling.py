"""
Client and Publication Profiling Module.
This module aggregates features from connectors into comprehensive profiles.
"""

import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pydantic import ValidationError

from data_connectors.base_connector import BaseConnector
from data_connectors.schemas import (
    ClientProfile, PublicationData, PublicationType,
    ChatMessage, CRMEntry, RFQData, ReadershipData, ClientKYC,
    BusinessRules, PipelineConfig
)


class Profiling:
    """
    Aggregates features from connectors into comprehensive client and publication profiles.
    
    This class is responsible for:
    - Managing registered data connectors
    - Building comprehensive client profiles from multiple data sources
    - Creating publication profiles with LLM-generated features
    - Applying business rules and lookback periods
    - Handling errors gracefully across connectors
    """

    def __init__(self, 
                 business_rules_path: str = "config/business_rules.yaml",
                 pipeline_config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize the profiling system.
        
        Args:
            business_rules_path: Path to business rules configuration
            pipeline_config_path: Path to pipeline configuration
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configurations
        self.business_rules = self._load_business_rules(business_rules_path)
        self.pipeline_config = self._load_pipeline_config(pipeline_config_path)
        
        # Registered connectors
        self.connectors: Dict[str, BaseConnector] = {}
        
        # LLM client for profile processing
        self.llm_client = None
        
        self.logger.info("Profiling system initialized")

    def _load_business_rules(self, config_path: str) -> BusinessRules:
        """Load and validate business rules configuration."""
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            return BusinessRules(**config_data)
        except Exception as e:
            self.logger.error(f"Failed to load business rules from {config_path}: {e}")
            # Return default business rules
            return BusinessRules(
                lookback_periods_days={
                    "chat": 90,
                    "rfq": 30,
                    "readership": 180,
                    "crm": 365
                },
                publication_expiry_days={
                    "default": 90,
                    "morning_comment": 1,
                    "weekly_report": 7,
                    "deep_dive": 365
                }
            )

    def _load_pipeline_config(self, config_path: str) -> PipelineConfig:
        """Load and validate pipeline configuration."""
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            return PipelineConfig(**config_data)
        except Exception as e:
            self.logger.error(f"Failed to load pipeline config from {config_path}: {e}")
            # Return default pipeline config
            return PipelineConfig(
                scoring_weights={
                    "chat": 0.5,
                    "rfq": 0.2,
                    "crm": 0.2,
                    "readership": 0.1
                }
            )

    def register_connector(self, name: str, connector_instance: BaseConnector) -> None:
        """
        Register a data connector.
        
        Args:
            name: Connector name (should match business rules keys)
            connector_instance: Initialized connector instance
        """
        if not isinstance(connector_instance, BaseConnector):
            raise ValueError(f"Connector must inherit from BaseConnector, got {type(connector_instance)}")
        
        self.connectors[name] = connector_instance
        self.logger.info(f"Registered connector: {name}")

    def set_llm_client(self, llm_client) -> None:
        """Set the LLM client for profile processing."""
        self.llm_client = llm_client
        self.logger.info("LLM client set for profiling")

    def build_client_profile(self, 
                           client_id: str, 
                           override_lookbacks: Optional[Dict[str, int]] = None) -> ClientProfile:
        """
        Build a comprehensive profile for a given client.
        
        Args:
            client_id: Client identifier
            override_lookbacks: Optional override for lookback periods
            
        Returns:
            Comprehensive client profile
        """
        self.logger.info(f"Building client profile for: {client_id}")
        
        # Initialize profile
        profile_data = {
            "client_id": client_id,
            "chat_features": [],
            "rfq_features": [],
            "crm_features": [],
            "readership_features": [],
            "last_updated": datetime.now()
        }
        
        # Use override lookbacks or default from business rules
        lookback_periods = override_lookbacks or self.business_rules.lookback_periods_days
        
        # Process each registered connector
        for connector_name, connector in self.connectors.items():
            try:
                self.logger.info(f"Processing connector: {connector_name}")
                
                # Set connector lookback period if applicable
                lookback_days = lookback_periods.get(connector_name)
                if lookback_days and hasattr(connector.config, 'lookback_days'):
                    original_lookback = connector.config.lookback_days
                    connector.config.lookback_days = lookback_days
                
                # Process connector data
                connector_data = connector.process_with_error_handling(
                    client_id=client_id,
                    llm_client=self.llm_client
                )
                
                # Restore original lookback
                if lookback_days and hasattr(connector.config, 'lookback_days'):
                    connector.config.lookback_days = original_lookback
                
                # Map connector data to profile fields
                self._map_connector_data_to_profile(profile_data, connector_name, connector_data)
                
                self.logger.info(f"Successfully processed {len(connector_data)} records from {connector_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing connector {connector_name}: {e}")
                # Continue with other connectors
                continue
        
        # Validate and create client profile
        try:
            client_profile = ClientProfile(**profile_data)
            
            # Generate profile embedding if LLM client available
            if self.llm_client:
                profile_embedding = self._generate_profile_embedding(client_profile)
                client_profile.embedding = profile_embedding
            
            self.logger.info(f"Client profile built successfully for {client_id}")
            return client_profile
            
        except ValidationError as e:
            self.logger.error(f"Client profile validation failed: {e}")
            # Return minimal valid profile
            return ClientProfile(client_id=client_id)

    def _map_connector_data_to_profile(self, 
                                     profile_data: Dict[str, Any], 
                                     connector_name: str, 
                                     connector_data: List[Dict[str, Any]]) -> None:
        """
        Map connector data to appropriate profile fields.
        
        Args:
            profile_data: Profile data dictionary to update
            connector_name: Name of the connector
            connector_data: Data from the connector
        """
        if not connector_data:
            return
        
        # Map based on connector name to profile field
        field_mapping = {
            "bloomberg_chat": "chat_features",
            "refinitiv_chat": "chat_features", 
            "crm": "crm_features",
            "rfq": "rfq_features",
            "readership": "readership_features"
        }
        
        profile_field = field_mapping.get(connector_name, f"{connector_name}_features")
        
        # Convert to appropriate schema objects
        try:
            if "chat" in connector_name:
                # Convert to ChatMessage objects
                chat_messages = [ChatMessage(**data) for data in connector_data]
                profile_data["chat_features"].extend(chat_messages)
                
            elif connector_name == "crm":
                # Convert to CRMEntry objects
                crm_entries = [CRMEntry(**data) for data in connector_data]
                profile_data["crm_features"].extend(crm_entries)
                
            elif connector_name == "rfq":
                # Convert to RFQData objects
                rfq_entries = [RFQData(**data) for data in connector_data]
                profile_data["rfq_features"].extend(rfq_entries)
                
            elif connector_name == "readership":
                # Convert to ReadershipData objects
                readership_entries = [ReadershipData(**data) for data in connector_data]
                profile_data["readership_features"].extend(readership_entries)
                
            else:
                # Add as generic data if no specific mapping
                if profile_field not in profile_data:
                    profile_data[profile_field] = []
                profile_data[profile_field].extend(connector_data)
                
        except ValidationError as e:
            self.logger.warning(f"Validation error mapping {connector_name} data: {e}")
            # Add as raw data if validation fails
            if profile_field not in profile_data:
                profile_data[profile_field] = []
            profile_data[profile_field].extend(connector_data)

    def _generate_profile_embedding(self, client_profile: ClientProfile) -> Optional[List[float]]:
        """
        Generate embedding vector for client profile.
        
        Args:
            client_profile: Client profile to embed
            
        Returns:
            Embedding vector or None if generation fails
        """
        try:
            # Create profile text for embedding
            profile_text_parts = []
            
            # Add chat summaries
            for chat in client_profile.chat_features:
                if hasattr(chat, 'summary') and chat.summary:
                    profile_text_parts.append(chat.summary)
            
            # Add CRM summaries
            for crm in client_profile.crm_features:
                if hasattr(crm, 'summary') and crm.summary:
                    profile_text_parts.append(crm.summary)
            
            # Add RFQ product information
            for rfq in client_profile.rfq_features:
                if hasattr(rfq, 'product'):
                    profile_text_parts.append(f"Interested in {rfq.product}")
            
            # Combine and generate embedding
            if profile_text_parts:
                profile_text = " ".join(profile_text_parts[:10])  # Limit to avoid token limits
                return self.llm_client.generate_embedding(profile_text)
            
        except Exception as e:
            self.logger.error(f"Error generating profile embedding: {e}")
        
        return None

    def build_publication_profile(self, 
                                publication_id: str, 
                                publication_data: Dict[str, Any]) -> PublicationData:
        """
        Build a profile for a given publication.
        
        Args:
            publication_id: Publication identifier
            publication_data: Raw publication data
            
        Returns:
            Publication profile with LLM-generated features
        """
        self.logger.info(f"Building publication profile for: {publication_id}")
        
        try:
            # Ensure publication_data has required fields
            if "id" not in publication_data:
                publication_data["id"] = publication_id
            
            # Parse publication type from metadata
            metadata = publication_data.get("metadata", {})
            pub_type_str = metadata.get("type", "default")
            
            try:
                publication_type = PublicationType(pub_type_str)
            except ValueError:
                publication_type = PublicationType.DEFAULT
            
            publication_data["publication_type"] = publication_type
            
            # Create initial publication object
            publication = PublicationData(**publication_data)
            
            # Generate LLM features if LLM client available
            if self.llm_client and publication.full_text:
                publication = self._enrich_publication_with_llm(publication)
            
            self.logger.info(f"Publication profile built for {publication_id}")
            return publication
            
        except ValidationError as e:
            self.logger.error(f"Publication validation failed for {publication_id}: {e}")
            # Return minimal valid publication
            return PublicationData(
                id=publication_id,
                title=publication_data.get("title", "Unknown"),
                date=datetime.now(),
                full_text=publication_data.get("full_text", "")
            )

    def _enrich_publication_with_llm(self, publication: PublicationData) -> PublicationData:
        """
        Enrich publication with LLM-generated features.
        
        Args:
            publication: Publication to enrich
            
        Returns:
            Enriched publication
        """
        try:
            # Generate summary
            if not publication.summary:
                summary_prompt = f"Summarize this financial research publication in 2-3 sentences:\n\n{publication.full_text[:1000]}"
                summary_response = self.llm_client.generate_completion(summary_prompt, max_tokens=150)
                
                if hasattr(summary_response, 'content'):
                    publication.summary = summary_response.content
                else:
                    publication.summary = str(summary_response)
            
            # Generate keywords
            if not publication.keywords:
                keywords_prompt = f"Extract 5-10 key financial terms and topics from this publication:\n\n{publication.full_text[:1000]}"
                keywords_response = self.llm_client.generate_completion(keywords_prompt, max_tokens=100)
                
                if hasattr(keywords_response, 'content'):
                    keywords_text = keywords_response.content
                else:
                    keywords_text = str(keywords_response)
                
                # Parse keywords
                keywords = [kw.strip() for kw in keywords_text.split(",")]
                publication.keywords = [kw for kw in keywords if kw][:10]
            
            # Generate target audience
            if not publication.target_audience:
                audience_prompt = f"Who is the target audience for this financial publication? (e.g., equity traders, FX specialists, institutional investors):\n\n{publication.title}\n{publication.summary or publication.full_text[:500]}"
                audience_response = self.llm_client.generate_completion(audience_prompt, max_tokens=50)
                
                if hasattr(audience_response, 'content'):
                    publication.target_audience = audience_response.content
                else:
                    publication.target_audience = str(audience_response)
            
            # Generate embedding
            if not publication.embedding:
                embedding_text = f"{publication.title} {publication.summary or publication.full_text[:500]}"
                publication.embedding = self.llm_client.generate_embedding(embedding_text)
            
        except Exception as e:
            self.logger.error(f"Error enriching publication with LLM: {e}")
        
        return publication

    def is_publication_expired(self, publication: PublicationData) -> bool:
        """
        Check if publication has expired based on business rules.
        
        Args:
            publication: Publication to check
            
        Returns:
            True if publication has expired
        """
        expiry_days = self.business_rules.publication_expiry_days.get(
            publication.publication_type.value,
            self.business_rules.publication_expiry_days.get("default", 90)
        )
        
        expiry_date = publication.date + timedelta(days=expiry_days)
        return datetime.now() > expiry_date

    def get_active_publications(self, publications: List[PublicationData]) -> List[PublicationData]:
        """
        Filter publications to only active (non-expired) ones.
        
        Args:
            publications: List of publications to filter
            
        Returns:
            List of active publications
        """
        active_publications = []
        
        for pub in publications:
            if not self.is_publication_expired(pub):
                active_publications.append(pub)
            else:
                self.logger.debug(f"Publication {pub.id} has expired")
        
        self.logger.info(f"Filtered {len(publications)} publications to {len(active_publications)} active ones")
        return active_publications

    def get_connector_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all registered connectors.
        
        Returns:
            Dictionary with health status for each connector
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "total_connectors": len(self.connectors),
            "connectors": {}
        }
        
        for name, connector in self.connectors.items():
            try:
                connector_health = connector.get_health_status()
                health_status["connectors"][name] = connector_health
            except Exception as e:
                health_status["connectors"][name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status

    def get_profiling_stats(self) -> Dict[str, Any]:
        """
        Get profiling system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "registered_connectors": list(self.connectors.keys()),
            "business_rules": self.business_rules.dict(),
            "pipeline_config": self.pipeline_config.dict(),
            "has_llm_client": self.llm_client is not None
        }


