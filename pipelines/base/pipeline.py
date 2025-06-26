"""
Base Pipeline Class for Recommendation System

This module defines the abstract base class for all pipeline versions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass


@dataclass
class PipelineResult:
    """Container for pipeline results."""
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    version: str
    execution_time: float
    confidence_scores: Optional[List[float]] = None


class BasePipeline(ABC):
    """Abstract base class for all recommendation pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.version = "base"
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
    @abstractmethod
    def run(self, query: str, context: Dict[str, Any]) -> PipelineResult:
        """
        Run the recommendation pipeline.
        
        Args:
            query: User query or client profile
            context: Additional context information
            
        Returns:
            PipelineResult containing recommendations and metadata
        """
        pass
    
    @abstractmethod
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about this pipeline version.
        
        Returns:
            Dictionary containing pipeline metadata
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate pipeline configuration.
        
        Returns:
            True if configuration is valid
        """
        required_keys = ['name', 'version']
        return all(key in self.config for key in required_keys)
    
    def preprocess_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the input query and context.
        
        Args:
            query: Raw input query
            context: Raw context information
            
        Returns:
            Processed query and context
        """
        return {
            'query': query.strip(),
            'context': context,
            'processed_at': self._get_timestamp()
        }
    
    def postprocess_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Postprocess the pipeline results.
        
        Args:
            results: Raw pipeline results
            
        Returns:
            Processed results
        """
        # Add pipeline version to each result
        for result in results:
            result['pipeline_version'] = self.version
            result['processed_at'] = self._get_timestamp()
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the pipeline.
        
        Returns:
            Health check results
        """
        return {
            'status': 'healthy',
            'version': self.version,
            'config_valid': self.validate_config(),
            'timestamp': self._get_timestamp()
        } 