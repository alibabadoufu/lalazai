"""
Abstract Base Class for all data connectors.
This module provides the foundation for modular, reusable data connectors.
"""

import abc
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pydantic import ValidationError

from .schemas import (
    ConnectorConfig, ClientProfile, PublicationData, 
    ChatMessage, CRMEntry, RFQData, ReadershipData, ClientKYC
)

# Configure logging
logger = logging.getLogger(__name__)


class BaseConnector(abc.ABC):
    """
    Abstract Base Class for all data connectors.
    
    All connectors must inherit from this class and implement the abstract methods.
    This ensures consistent interface and makes it easy to add new data sources.
    """

    def __init__(self, config: Union[Dict[str, Any], ConnectorConfig]):
        """
        Initialize the connector with configuration.
        
        Args:
            config: Configuration dictionary or ConnectorConfig object
        """
        if isinstance(config, dict):
            self.config = ConnectorConfig(**config)
        else:
            self.config = config
            
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized {self.__class__.__name__} connector")
        
        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate connector configuration."""
        if not self.config.enabled:
            self.logger.warning(f"Connector {self.config.connector_name} is disabled")
            
        if self.config.lookback_days and self.config.lookback_days <= 0:
            raise ValueError("Lookback days must be positive")

    @abc.abstractmethod
    def ingest(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Ingests raw data from the source.
        
        Args:
            client_id: Optional client ID to filter data
            **kwargs: Additional parameters for data ingestion
            
        Returns:
            List of raw data records
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ConnectionError: If unable to connect to data source
            ValueError: If invalid parameters provided
        """
        pass

    @abc.abstractmethod
    def preprocess(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocesses raw data into a standardized format.
        
        Args:
            raw_data: Raw data from ingestion step
            
        Returns:
            List of preprocessed data records
            
        Raises:
            ValidationError: If data doesn't match expected schema
            ValueError: If data preprocessing fails
        """
        pass

    @abc.abstractmethod
    def build_features(self, processed_data: List[Dict[str, Any]], 
                      llm_client=None) -> List[Dict[str, Any]]:
        """
        Builds features from processed data, potentially using LLM.
        
        Args:
            processed_data: Preprocessed data
            llm_client: Optional LLM client for feature extraction
            
        Returns:
            List of feature-enriched data records
            
        Raises:
            RuntimeError: If LLM processing fails
            ValidationError: If features don't match expected schema
        """
        pass

    def get_lookback_date(self) -> datetime:
        """
        Get the lookback date based on configuration.
        
        Returns:
            Datetime representing the earliest date to consider
        """
        if not self.config.lookback_days:
            # Default to 30 days if not specified
            lookback_days = 30
        else:
            lookback_days = self.config.lookback_days
            
        return datetime.now() - timedelta(days=lookback_days)

    def validate_data(self, data: List[Dict[str, Any]], 
                     schema_class) -> List[Any]:
        """
        Validate data against a Pydantic schema.
        
        Args:
            data: Data to validate
            schema_class: Pydantic model class for validation
            
        Returns:
            List of validated objects
            
        Raises:
            ValidationError: If validation fails
        """
        validated_objects = []
        errors = []
        
        for i, record in enumerate(data):
            try:
                validated_obj = schema_class(**record)
                validated_objects.append(validated_obj)
            except ValidationError as e:
                error_msg = f"Validation error in record {i}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        if errors:
            self.logger.warning(f"Found {len(errors)} validation errors out of {len(data)} records")
            if len(errors) == len(data):
                raise ValidationError(f"All records failed validation: {errors[:3]}")  # Show first 3 errors
        
        return validated_objects

    def process_with_error_handling(self, client_id: Optional[str] = None, 
                                  llm_client=None) -> List[Dict[str, Any]]:
        """
        Execute the full connector pipeline with error handling.
        
        Args:
            client_id: Optional client ID to filter data
            llm_client: Optional LLM client for feature extraction
            
        Returns:
            List of processed and feature-enriched data
            
        Raises:
            RuntimeError: If any step fails critically
        """
        try:
            # Step 1: Ingest data
            self.logger.info(f"Starting data ingestion for {self.config.connector_name}")
            raw_data = self.ingest(client_id=client_id)
            self.logger.info(f"Ingested {len(raw_data)} raw records")
            
            if not raw_data:
                self.logger.warning("No raw data ingested")
                return []
            
            # Step 2: Preprocess data
            self.logger.info("Starting data preprocessing")
            processed_data = self.preprocess(raw_data)
            self.logger.info(f"Preprocessed {len(processed_data)} records")
            
            # Step 3: Build features
            self.logger.info("Starting feature building")
            feature_data = self.build_features(processed_data, llm_client=llm_client)
            self.logger.info(f"Built features for {len(feature_data)} records")
            
            return feature_data
            
        except Exception as e:
            error_msg = f"Error in {self.config.connector_name} connector pipeline: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get connector health status.
        
        Returns:
            Dictionary with health information
        """
        return {
            "connector_name": self.config.connector_name,
            "enabled": self.config.enabled,
            "last_health_check": datetime.now().isoformat(),
            "status": "healthy" if self.config.enabled else "disabled",
            "config": self.config.dict()
        }

    @abc.abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass

    def __str__(self) -> str:
        """String representation of the connector."""
        return f"{self.__class__.__name__}(name={self.config.connector_name}, enabled={self.config.enabled})"

    def __repr__(self) -> str:
        """Detailed string representation of the connector."""
        return (f"{self.__class__.__name__}("
                f"name={self.config.connector_name}, "
                f"enabled={self.config.enabled}, "
                f"lookback_days={self.config.lookback_days})")


