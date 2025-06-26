"""
Pipeline Factory for Version Management

This factory provides a centralized interface for creating and managing
different versions of recommendation pipelines.
"""

from typing import Dict, Any, List, Optional, Type
import logging
import importlib
from pipelines.base.pipeline import BasePipeline


class PipelineFactory:
    """Factory for creating and managing pipeline instances."""
    
    # Pipeline registry mapping versions to their classes
    PIPELINE_REGISTRY = {
        "v1_basic_rag": {
            "class": "pipelines.v1_basic_rag.pipeline.BasicRAGPipeline",
            "name": "Basic RAG Pipeline",
            "description": "Simple retrieval-augmented generation",
            "status": "stable",
            "min_config_version": "1.0"
        },
        "v2_rag_reflection": {
            "class": "pipelines.v2_rag_reflection.pipeline.RAGReflectionPipeline", 
            "name": "RAG with Reflection Pipeline",
            "description": "Enhanced RAG with self-reflection and refinement",
            "status": "stable",
            "min_config_version": "1.1"
        },
        "v3_rag_graph": {
            "class": "pipelines.v3_rag_graph.pipeline.GraphRAGPipeline",
            "name": "RAG with Knowledge Graph",
            "description": "RAG enhanced with knowledge graph reasoning",
            "status": "experimental",
            "min_config_version": "1.2"
        },
        "v4_agentic_rag": {
            "class": "pipelines.v4_agentic_rag.pipeline.AgenticRAGPipeline",
            "name": "Agentic RAG Pipeline", 
            "description": "Multi-agent RAG with specialized agents",
            "status": "development",
            "min_config_version": "1.3"
        }
    }
    
    # Feature compatibility matrix
    FEATURE_MATRIX = {
        "v1_basic_rag": {
            "reflection": False,
            "graph_reasoning": False,
            "multi_agent": False,
            "batch_processing": True,
            "confidence_scoring": True,
            "diversity_filtering": False
        },
        "v2_rag_reflection": {
            "reflection": True,
            "graph_reasoning": False,
            "multi_agent": False,
            "batch_processing": True,
            "confidence_scoring": True,
            "diversity_filtering": True
        },
        "v3_rag_graph": {
            "reflection": True,
            "graph_reasoning": True,
            "multi_agent": False,
            "batch_processing": True,
            "confidence_scoring": True,
            "diversity_filtering": True
        },
        "v4_agentic_rag": {
            "reflection": True,
            "graph_reasoning": True,
            "multi_agent": True,
            "batch_processing": True,
            "confidence_scoring": True,
            "diversity_filtering": True
        }
    }
    
    def __init__(self):
        """Initialize the pipeline factory."""
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._loaded_classes = {}  # Cache for loaded pipeline classes
        
    @classmethod
    def create_pipeline(cls, version: str, config: Dict[str, Any]) -> BasePipeline:
        """
        Create a pipeline instance based on version.
        
        Args:
            version: Pipeline version identifier
            config: Pipeline configuration dictionary
            
        Returns:
            Initialized pipeline instance
            
        Raises:
            ValueError: If pipeline version is not supported
            ImportError: If pipeline class cannot be imported
        """
        factory = cls()
        return factory._create_pipeline_instance(version, config)
    
    def _create_pipeline_instance(self, version: str, config: Dict[str, Any]) -> BasePipeline:
        """Create a pipeline instance."""
        if version not in self.PIPELINE_REGISTRY:
            available_versions = list(self.PIPELINE_REGISTRY.keys())
            raise ValueError(
                f"Pipeline version '{version}' not supported. "
                f"Available versions: {available_versions}"
            )
        
        pipeline_info = self.PIPELINE_REGISTRY[version]
        
        # Check if pipeline is available (not in development status for production)
        if config.get('environment') == 'production' and pipeline_info['status'] == 'development':
            raise ValueError(f"Pipeline version '{version}' is not available in production")
        
        # Load pipeline class
        pipeline_class = self._load_pipeline_class(version, pipeline_info['class'])
        
        # Validate configuration
        self._validate_config(version, config)
        
        # Create and return instance
        try:
            instance = pipeline_class(config)
            self.logger.info(f"Created pipeline instance: {version}")
            return instance
        except Exception as e:
            self.logger.error(f"Failed to create pipeline instance {version}: {e}")
            raise
    
    def _load_pipeline_class(self, version: str, class_path: str) -> Type[BasePipeline]:
        """Load pipeline class from module path."""
        if version in self._loaded_classes:
            return self._loaded_classes[version]
        
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            pipeline_class = getattr(module, class_name)
            
            # Validate that it's a proper pipeline class
            if not issubclass(pipeline_class, BasePipeline):
                raise ValueError(f"Class {class_name} is not a valid pipeline class")
            
            self._loaded_classes[version] = pipeline_class
            return pipeline_class
            
        except (ImportError, AttributeError, ValueError) as e:
            self.logger.error(f"Failed to load pipeline class {class_path}: {e}")
            raise ImportError(f"Cannot import pipeline class for version {version}: {e}")
    
    def _validate_config(self, version: str, config: Dict[str, Any]) -> None:
        """Validate pipeline configuration."""
        pipeline_info = self.PIPELINE_REGISTRY[version]
        
        # Check minimum config version
        config_version = config.get('version', '1.0')
        min_version = pipeline_info['min_config_version']
        
        if self._compare_versions(config_version, min_version) < 0:
            raise ValueError(
                f"Configuration version {config_version} is too old for pipeline {version}. "
                f"Minimum required: {min_version}"
            )
        
        # Check required configuration keys
        required_keys = ['name', 'version']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
        return 0
    
    @classmethod
    def list_available_versions(cls, include_experimental: bool = False, include_development: bool = False) -> List[str]:
        """
        List all available pipeline versions.
        
        Args:
            include_experimental: Include experimental versions
            include_development: Include development versions
            
        Returns:
            List of available pipeline version identifiers
        """
        versions = []
        
        for version, info in cls.PIPELINE_REGISTRY.items():
            status = info['status']
            
            if status == 'stable':
                versions.append(version)
            elif status == 'experimental' and include_experimental:
                versions.append(version)
            elif status == 'development' and include_development:
                versions.append(version)
        
        return sorted(versions)
    
    @classmethod
    def get_pipeline_info(cls, version: str) -> Dict[str, Any]:
        """
        Get detailed information about a pipeline version.
        
        Args:
            version: Pipeline version identifier
            
        Returns:
            Dictionary containing pipeline information
            
        Raises:
            ValueError: If version is not found
        """
        if version not in cls.PIPELINE_REGISTRY:
            raise ValueError(f"Pipeline version '{version}' not found")
        
        info = cls.PIPELINE_REGISTRY[version].copy()
        info['features'] = cls.FEATURE_MATRIX.get(version, {})
        info['version'] = version
        
        return info
    
    @classmethod
    def get_latest_version(cls, include_experimental: bool = False) -> str:
        """
        Get the latest stable pipeline version.
        
        Args:
            include_experimental: Include experimental versions in consideration
            
        Returns:
            Latest pipeline version identifier
        """
        available_versions = cls.list_available_versions(
            include_experimental=include_experimental,
            include_development=False
        )
        
        if not available_versions:
            raise ValueError("No stable pipeline versions available")
        
        # Sort by version number (assuming format vX_name)
        def version_key(version: str) -> int:
            try:
                return int(version.split('_')[0][1:])  # Extract number from vX
            except (ValueError, IndexError):
                return 0
        
        sorted_versions = sorted(available_versions, key=version_key)
        return sorted_versions[-1]
    
    @classmethod
    def get_recommended_version(cls, requirements: Dict[str, Any]) -> str:
        """
        Get recommended pipeline version based on requirements.
        
        Args:
            requirements: Dictionary of feature requirements
            
        Returns:
            Recommended pipeline version
        """
        required_features = requirements.get('features', [])
        performance_priority = requirements.get('performance_priority', 'balanced')  # 'speed', 'quality', 'balanced'
        stability_requirement = requirements.get('stability', 'stable')  # 'stable', 'experimental', 'any'
        
        # Filter by stability requirement
        if stability_requirement == 'stable':
            candidate_versions = cls.list_available_versions()
        elif stability_requirement == 'experimental':
            candidate_versions = cls.list_available_versions(include_experimental=True)
        else:  # 'any'
            candidate_versions = cls.list_available_versions(include_experimental=True, include_development=True)
        
        # Filter by required features
        compatible_versions = []
        for version in candidate_versions:
            features = cls.FEATURE_MATRIX.get(version, {})
            if all(features.get(feature, False) for feature in required_features):
                compatible_versions.append(version)
        
        if not compatible_versions:
            raise ValueError(f"No pipeline versions support required features: {required_features}")
        
        # Select based on performance priority
        if performance_priority == 'speed':
            # Prefer simpler pipelines for speed
            return min(compatible_versions, key=lambda v: int(v.split('_')[0][1:]))
        elif performance_priority == 'quality':
            # Prefer more advanced pipelines for quality
            return max(compatible_versions, key=lambda v: int(v.split('_')[0][1:]))
        else:  # 'balanced'
            # Return a balanced choice (v2 if available, otherwise latest stable)
            if 'v2_rag_reflection' in compatible_versions:
                return 'v2_rag_reflection'
            return cls.get_latest_version()
    
    @classmethod
    def compare_versions(cls, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two pipeline versions.
        
        Args:
            version1: First pipeline version
            version2: Second pipeline version
            
        Returns:
            Comparison results
        """
        info1 = cls.get_pipeline_info(version1)
        info2 = cls.get_pipeline_info(version2)
        
        features1 = info1['features']
        features2 = info2['features']
        
        # Find feature differences
        added_features = [f for f, v in features2.items() if v and not features1.get(f, False)]
        removed_features = [f for f, v in features1.items() if v and not features2.get(f, False)]
        
        return {
            'version1': version1,
            'version2': version2,
            'added_features': added_features,
            'removed_features': removed_features,
            'compatibility': 'backward' if not removed_features else 'breaking',
            'recommendation': 'upgrade' if len(added_features) > len(removed_features) else 'maintain'
        }
    
    @classmethod
    def create_auto_pipeline(cls, requirements: Dict[str, Any], config: Dict[str, Any]) -> BasePipeline:
        """
        Automatically select and create the best pipeline for given requirements.
        
        Args:
            requirements: Feature and performance requirements
            config: Pipeline configuration
            
        Returns:
            Optimally selected pipeline instance
        """
        recommended_version = cls.get_recommended_version(requirements)
        return cls.create_pipeline(recommended_version, config)
    
    def validate_all_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all registered pipelines.
        
        Returns:
            Validation results for each pipeline
        """
        results = {}
        
        for version in self.PIPELINE_REGISTRY:
            try:
                # Try to load the class
                pipeline_info = self.PIPELINE_REGISTRY[version]
                self._load_pipeline_class(version, pipeline_info['class'])
                
                results[version] = {
                    'status': 'valid',
                    'can_import': True,
                    'error': None
                }
            except Exception as e:
                results[version] = {
                    'status': 'invalid',
                    'can_import': False,
                    'error': str(e)
                }
        
        return results 