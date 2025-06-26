"""
Evaluation Framework for Intelligent Research Recommendation System.

This package provides comprehensive evaluation capabilities for both offline and online metrics,
enabling measurement of recommendation quality, business impact, and system performance.

Enhanced with LLM-as-a-Judge methodology for qualitative assessment of AI components.

Components:
- BaseEvaluator: Abstract base class for all evaluators
- OfflineEvaluator: Comprehensive offline evaluation framework
- OnlineMetrics: Real-time business impact measurement
- Evaluators: Modular, specialized evaluators for different pipeline components
  - LLM Evaluators: Enhanced with LLM-as-a-judge for qualitative assessment
  - Connector Evaluators: Data ingestion and quality evaluation
  - Recommendation Evaluators: Hybrid traditional and LLM-based recommendation evaluation
  - Profile Evaluators: Client profiling quality assessment
- EvaluatorFactory: Factory pattern for easy evaluator creation and management
"""

from .base_evaluator import BaseEvaluator
from .offline_evaluator import OfflineEvaluator
from .online_metrics import OnlineMetrics

# Import modular evaluators
from .evaluators import (
    EvaluatorFactory,
    LLMFeatureEvaluator,
    LLMJudge,
    ConnectorEvaluator, 
    RecommendationEvaluator,
    ProfileEvaluator
)

# Maintain backward compatibility
from .evaluators.llm_evaluators import LLMFeatureEvaluator
from .evaluators.connector_evaluators import ConnectorEvaluator
from .evaluators.recommendation_evaluators import RecommendationEvaluator
from .evaluators.profile_evaluators import ProfileEvaluator

__all__ = [
    'BaseEvaluator',
    'OfflineEvaluator', 
    'OnlineMetrics',
    'EvaluatorFactory',
    'LLMFeatureEvaluator',
    'LLMJudge',
    'ConnectorEvaluator',
    'RecommendationEvaluator', 
    'ProfileEvaluator'
]

__version__ = '2.0.0'  # Updated version for the new modular architecture 