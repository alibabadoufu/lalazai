"""
Component Evaluators for the Recommendation System.

This package provides specialized evaluation for different pipeline components,
organized into separate modules for better maintainability and extensibility.

Modules:
- llm_evaluators: LLM-based evaluators using LLM-as-a-judge
- connector_evaluators: Data connector evaluation
- recommendation_evaluators: Recommendation engine evaluation  
- profile_evaluators: Client profiling evaluation
- factory: Factory for creating evaluators
"""

from .factory import EvaluatorFactory
from .llm_evaluators import LLMFeatureEvaluator, LLMJudge
from .connector_evaluators import ConnectorEvaluator
from .recommendation_evaluators import RecommendationEvaluator
from .profile_evaluators import ProfileEvaluator

__all__ = [
    'EvaluatorFactory',
    'LLMFeatureEvaluator',
    'LLMJudge', 
    'ConnectorEvaluator',
    'RecommendationEvaluator',
    'ProfileEvaluator'
] 