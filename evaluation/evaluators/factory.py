"""
Evaluator Factory for Component Evaluators.

This module provides a factory pattern for creating and configuring evaluators,
making it easy to extend the evaluation system with new evaluator types.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union
from enum import Enum

from ..base_evaluator import BaseEvaluator, EvaluationConfig
from .llm_evaluators import LLMFeatureEvaluator
from .connector_evaluators import ConnectorEvaluator
from .recommendation_evaluators import RecommendationEvaluator
from .profile_evaluators import ProfileEvaluator


class EvaluatorType(Enum):
    """Supported evaluator types."""
    LLM_FEATURE = "llm_feature"
    CONNECTOR = "connector"
    RECOMMENDATION = "recommendation" 
    PROFILE = "profile"


class EvaluatorFactory:
    """
    Factory for creating and managing component evaluators.
    
    Provides:
    - Easy evaluator creation and configuration
    - Registry of available evaluator types
    - Batch evaluator creation from configuration
    - Evaluator validation and error handling
    """
    
    # Registry of evaluator classes
    _evaluator_registry: Dict[EvaluatorType, Type[BaseEvaluator]] = {
        EvaluatorType.LLM_FEATURE: LLMFeatureEvaluator,
        EvaluatorType.CONNECTOR: ConnectorEvaluator,
        EvaluatorType.RECOMMENDATION: RecommendationEvaluator,
        EvaluatorType.PROFILE: ProfileEvaluator,
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._created_evaluators: Dict[str, BaseEvaluator] = {}
    
    @classmethod
    def register_evaluator(cls, 
                          evaluator_type: EvaluatorType, 
                          evaluator_class: Type[BaseEvaluator]):
        """
        Register a new evaluator type.
        
        Args:
            evaluator_type: Type identifier for the evaluator
            evaluator_class: Evaluator class to register
        """
        cls._evaluator_registry[evaluator_type] = evaluator_class
    
    def create_evaluator(self, 
                        evaluator_type: Union[EvaluatorType, str],
                        config: EvaluationConfig) -> BaseEvaluator:
        """
        Create a single evaluator instance.
        
        Args:
            evaluator_type: Type of evaluator to create
            config: Configuration for the evaluator
            
        Returns:
            Configured evaluator instance
            
        Raises:
            ValueError: If evaluator type is not supported
            Exception: If evaluator creation fails
        """
        # Convert string to enum if needed
        if isinstance(evaluator_type, str):
            try:
                evaluator_type = EvaluatorType(evaluator_type)
            except ValueError:
                raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
        
        if evaluator_type not in self._evaluator_registry:
            raise ValueError(f"Evaluator type {evaluator_type} is not registered")
        
        try:
            evaluator_class = self._evaluator_registry[evaluator_type]
            evaluator = evaluator_class(config)
            
            # Store reference for later retrieval
            self._created_evaluators[config.evaluator_name] = evaluator
            
            self.logger.info(f"Created evaluator: {config.evaluator_name} ({evaluator_type.value})")
            return evaluator
            
        except Exception as e:
            self.logger.error(f"Failed to create evaluator {config.evaluator_name}: {e}")
            raise
    
    def create_evaluators_from_config(self, 
                                    configs: List[Dict[str, Any]]) -> Dict[str, BaseEvaluator]:
        """
        Create multiple evaluators from configuration list.
        
        Args:
            configs: List of evaluator configurations
            
        Returns:
            Dictionary mapping evaluator names to instances
        """
        evaluators = {}
        
        for config_dict in configs:
            try:
                # Create evaluation config
                config = self._create_evaluation_config(config_dict)
                
                # Get evaluator type
                evaluator_type = config_dict.get('type')
                if not evaluator_type:
                    raise ValueError("Evaluator type is required")
                
                # Create evaluator
                evaluator = self.create_evaluator(evaluator_type, config)
                evaluators[config.evaluator_name] = evaluator
                
            except Exception as e:
                self.logger.error(f"Failed to create evaluator from config {config_dict}: {e}")
                # Continue creating other evaluators
                continue
        
        self.logger.info(f"Created {len(evaluators)} evaluators from {len(configs)} configurations")
        return evaluators
    
    def get_evaluator(self, name: str) -> Optional[BaseEvaluator]:
        """
        Get a previously created evaluator by name.
        
        Args:
            name: Name of the evaluator
            
        Returns:
            Evaluator instance or None if not found
        """
        return self._created_evaluators.get(name)
    
    def get_all_evaluators(self) -> Dict[str, BaseEvaluator]:
        """
        Get all created evaluators.
        
        Returns:
            Dictionary of all evaluator instances
        """
        return self._created_evaluators.copy()
    
    def get_evaluators_by_type(self, 
                              evaluator_type: Union[EvaluatorType, str]) -> List[BaseEvaluator]:
        """
        Get all evaluators of a specific type.
        
        Args:
            evaluator_type: Type of evaluators to retrieve
            
        Returns:
            List of evaluators of the specified type
        """
        if isinstance(evaluator_type, str):
            try:
                evaluator_type = EvaluatorType(evaluator_type)
            except ValueError:
                return []
        
        target_class = self._evaluator_registry.get(evaluator_type)
        if not target_class:
            return []
        
        return [evaluator for evaluator in self._created_evaluators.values() 
                if isinstance(evaluator, target_class)]
    
    def _create_evaluation_config(self, config_dict: Dict[str, Any]) -> EvaluationConfig:
        """
        Create EvaluationConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            EvaluationConfig instance
        """
        # Extract required fields
        evaluator_name = config_dict.get('name')
        if not evaluator_name:
            raise ValueError("Evaluator name is required")
        
        # Create config with defaults
        config = EvaluationConfig(
            evaluator_name=evaluator_name,
            enabled=config_dict.get('enabled', True),
            evaluation_interval_hours=config_dict.get('evaluation_interval_hours', 24),
            sample_size=config_dict.get('sample_size'),
            ground_truth_source=config_dict.get('ground_truth_source'),
            thresholds=config_dict.get('thresholds', {}),
            custom_parameters=config_dict.get('custom_parameters', {})
        )
        
        return config
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """
        Get list of supported evaluator types.
        
        Returns:
            List of evaluator type strings
        """
        return [eval_type.value for eval_type in cls._evaluator_registry.keys()]
    
    def validate_configuration(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate evaluator configuration.
        
        Args:
            config_dict: Configuration to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        if 'name' not in config_dict:
            validation_result['errors'].append("Missing required field: 'name'")
            validation_result['valid'] = False
        
        if 'type' not in config_dict:
            validation_result['errors'].append("Missing required field: 'type'")
            validation_result['valid'] = False
        
        # Check evaluator type
        evaluator_type = config_dict.get('type')
        if evaluator_type and evaluator_type not in self.get_supported_types():
            validation_result['errors'].append(f"Unsupported evaluator type: {evaluator_type}")
            validation_result['valid'] = False
        
        # Check for duplicate names
        evaluator_name = config_dict.get('name')
        if evaluator_name and evaluator_name in self._created_evaluators:
            validation_result['warnings'].append(f"Evaluator with name '{evaluator_name}' already exists")
        
        # Type-specific validation
        if validation_result['valid']:
            type_validation = self._validate_type_specific_config(config_dict)
            validation_result['errors'].extend(type_validation.get('errors', []))
            validation_result['warnings'].extend(type_validation.get('warnings', []))
            if type_validation.get('errors'):
                validation_result['valid'] = False
        
        return validation_result
    
    def _validate_type_specific_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate type-specific configuration requirements.
        
        Args:
            config_dict: Configuration to validate
            
        Returns:
            Dictionary with type-specific validation results
        """
        result = {'errors': [], 'warnings': []}
        evaluator_type = config_dict.get('type')
        
        if evaluator_type == EvaluatorType.LLM_FEATURE.value:
            # LLM evaluators may need specific parameters
            custom_params = config_dict.get('custom_parameters', {})
            if not custom_params.get('llm_model'):
                result['warnings'].append("Consider specifying 'llm_model' in custom_parameters for LLM evaluators")
        
        elif evaluator_type == EvaluatorType.CONNECTOR.value:
            # Connector evaluators may need quality thresholds
            thresholds = config_dict.get('thresholds', {})
            expected_thresholds = ['completeness', 'validity', 'consistency', 'timeliness']
            missing_thresholds = [t for t in expected_thresholds if t not in thresholds]
            if missing_thresholds:
                result['warnings'].append(f"Consider setting thresholds for: {', '.join(missing_thresholds)}")
        
        elif evaluator_type == EvaluatorType.RECOMMENDATION.value:
            # Recommendation evaluators may need business context
            custom_params = config_dict.get('custom_parameters', {})
            if not custom_params.get('enable_llm_judges'):
                result['warnings'].append("Consider enabling LLM judges for enhanced recommendation evaluation")
        
        return result
    
    def create_default_evaluators(self) -> Dict[str, BaseEvaluator]:
        """
        Create a set of default evaluators with sensible configurations.
        
        Returns:
            Dictionary of default evaluators
        """
        default_configs = [
            {
                'name': 'default_llm_evaluator',
                'type': EvaluatorType.LLM_FEATURE.value,
                'enabled': True,
                'evaluation_interval_hours': 24,
                'custom_parameters': {
                    'enable_llm_judges': True,
                    'num_judge_runs': 2
                }
            },
            {
                'name': 'default_connector_evaluator',
                'type': EvaluatorType.CONNECTOR.value,
                'enabled': True,
                'evaluation_interval_hours': 12,
                'thresholds': {
                    'completeness': 0.95,
                    'validity': 0.90,
                    'consistency': 0.85,
                    'timeliness': 0.80
                }
            },
            {
                'name': 'default_recommendation_evaluator',
                'type': EvaluatorType.RECOMMENDATION.value,
                'enabled': True,
                'evaluation_interval_hours': 6,
                'custom_parameters': {
                    'enable_llm_judges': True
                }
            },
            {
                'name': 'default_profile_evaluator',
                'type': EvaluatorType.PROFILE.value,
                'enabled': True,
                'evaluation_interval_hours': 24
            }
        ]
        
        return self.create_evaluators_from_config(default_configs)
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        """
        Get information about available and created evaluators.
        
        Returns:
            Dictionary with evaluator information
        """
        return {
            'supported_types': self.get_supported_types(),
            'created_evaluators': {
                name: {
                    'type': type(evaluator).__name__,
                    'enabled': evaluator.config.enabled,
                    'last_evaluation': evaluator.last_evaluation_time.isoformat() if evaluator.last_evaluation_time else None,
                    'status': evaluator.status.value
                }
                for name, evaluator in self._created_evaluators.items()
            },
            'evaluator_counts_by_type': {
                eval_type.value: len(self.get_evaluators_by_type(eval_type))
                for eval_type in EvaluatorType
            }
        } 