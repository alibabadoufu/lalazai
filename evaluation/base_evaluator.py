"""
Base Evaluator Class for the Recommendation System.
Provides common functionality and interface for all evaluation components.
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class EvaluationStatus(Enum):
    """Status of evaluation execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    RELEVANCE = "relevance"
    DIVERSITY = "diversity"
    COVERAGE = "coverage"
    ENGAGEMENT = "engagement"
    BUSINESS_IMPACT = "business_impact"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    evaluator_name: str
    component_name: str
    metric_type: MetricType
    value: Union[float, int, str]
    timestamp: datetime
    metadata: Dict[str, Any]
    status: EvaluationStatus = EvaluationStatus.COMPLETED
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evaluator_name": self.evaluator_name,
            "component_name": self.component_name,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "status": self.status.value,
            "error_message": self.error_message
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    evaluator_name: str
    enabled: bool = True
    evaluation_interval_hours: int = 24
    sample_size: Optional[int] = None
    ground_truth_source: Optional[str] = None
    thresholds: Dict[str, float] = None
    custom_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {}
        if self.custom_parameters is None:
            self.custom_parameters = {}


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators in the recommendation system.
    
    Provides common functionality including:
    - Logging and error handling
    - Result storage and retrieval
    - Configuration management
    - Health monitoring
    - Metric calculation utilities
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the base evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.results_history: List[EvaluationResult] = []
        self.last_evaluation_time: Optional[datetime] = None
        self.status = EvaluationStatus.PENDING
        
        # Initialize component-specific setup
        self._initialize()
    
    def _initialize(self):
        """Initialize evaluator-specific components. Override in subclasses."""
        pass
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any], **kwargs) -> List[EvaluationResult]:
        """
        Perform evaluation on the provided data.
        
        Args:
            data: Data to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of evaluation results
        """
        pass
    
    def run_evaluation(self, data: Dict[str, Any], **kwargs) -> List[EvaluationResult]:
        """
        Run evaluation with error handling and logging.
        
        Args:
            data: Data to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of evaluation results
        """
        if not self.config.enabled:
            self.logger.info(f"Evaluator {self.config.evaluator_name} is disabled")
            return []
        
        self.logger.info(f"Starting evaluation: {self.config.evaluator_name}")
        self.status = EvaluationStatus.RUNNING
        start_time = datetime.now()
        
        try:
            results = self.evaluate(data, **kwargs)
            
            # Store results
            self.results_history.extend(results)
            self.last_evaluation_time = start_time
            self.status = EvaluationStatus.COMPLETED
            
            self.logger.info(f"Evaluation completed: {len(results)} metrics calculated")
            return results
            
        except Exception as e:
            self.status = EvaluationStatus.FAILED
            error_msg = f"Evaluation failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Create error result
            error_result = EvaluationResult(
                evaluator_name=self.config.evaluator_name,
                component_name="system",
                metric_type=MetricType.BUSINESS_IMPACT,
                value=0,
                timestamp=start_time,
                metadata={"error": str(e)},
                status=EvaluationStatus.FAILED,
                error_message=error_msg
            )
            
            return [error_result]
    
    def get_latest_results(self, metric_type: Optional[MetricType] = None) -> List[EvaluationResult]:
        """
        Get latest evaluation results.
        
        Args:
            metric_type: Optional filter by metric type
            
        Returns:
            List of latest results
        """
        if not self.results_history:
            return []
        
        # Get results from the last evaluation run
        latest_time = max(result.timestamp for result in self.results_history)
        latest_results = [
            result for result in self.results_history 
            if result.timestamp == latest_time
        ]
        
        if metric_type:
            latest_results = [
                result for result in latest_results 
                if result.metric_type == metric_type
            ]
        
        return latest_results
    
    def get_historical_results(self, 
                             days_back: int = 30,
                             metric_type: Optional[MetricType] = None) -> List[EvaluationResult]:
        """
        Get historical evaluation results.
        
        Args:
            days_back: Number of days to look back
            metric_type: Optional filter by metric type
            
        Returns:
            List of historical results
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        historical_results = [
            result for result in self.results_history 
            if result.timestamp >= cutoff_date
        ]
        
        if metric_type:
            historical_results = [
                result for result in historical_results 
                if result.metric_type == metric_type
            ]
        
        return historical_results
    
    def calculate_trend(self, metric_type: MetricType, days_back: int = 7) -> Dict[str, float]:
        """
        Calculate trend for a specific metric.
        
        Args:
            metric_type: Type of metric to analyze
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        results = self.get_historical_results(days_back, metric_type)
        
        if len(results) < 2:
            return {"trend": 0.0, "confidence": 0.0}
        
        # Sort by timestamp
        results.sort(key=lambda x: x.timestamp)
        
        # Calculate simple linear trend
        values = [float(result.value) for result in results if isinstance(result.value, (int, float))]
        
        if len(values) < 2:
            return {"trend": 0.0, "confidence": 0.0}
        
        # Simple trend calculation (slope)
        n = len(values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate confidence based on data points
        confidence = min(1.0, len(values) / 10.0)  # Max confidence with 10+ data points
        
        return {
            "trend": slope,
            "confidence": confidence,
            "data_points": len(values),
            "latest_value": values[-1],
            "average_value": sum(values) / len(values)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the evaluator.
        
        Returns:
            Health status information
        """
        return {
            "evaluator_name": self.config.evaluator_name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "last_evaluation": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            "total_evaluations": len(self.results_history),
            "recent_errors": len([
                r for r in self.results_history[-10:] 
                if r.status == EvaluationStatus.FAILED
            ])
        }
    
    def export_results(self, 
                      format: str = "json",
                      days_back: Optional[int] = None) -> str:
        """
        Export evaluation results.
        
        Args:
            format: Export format (json, csv)
            days_back: Optional number of days to include
            
        Returns:
            Exported data as string
        """
        if days_back:
            results = self.get_historical_results(days_back)
        else:
            results = self.results_history
        
        if format.lower() == "json":
            return json.dumps([result.to_dict() for result in results], indent=2)
        elif format.lower() == "csv":
            return self._export_csv(results)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv(self, results: List[EvaluationResult]) -> str:
        """Export results as CSV format."""
        if not results:
            return ""
        
        headers = ["timestamp", "evaluator_name", "component_name", "metric_type", "value", "status"]
        lines = [",".join(headers)]
        
        for result in results:
            line = ",".join([
                result.timestamp.isoformat(),
                result.evaluator_name,
                result.component_name,
                result.metric_type.value,
                str(result.value),
                result.status.value
            ])
            lines.append(line)
        
        return "\n".join(lines)
    
    def should_run_evaluation(self) -> bool:
        """
        Check if evaluation should run based on interval.
        
        Returns:
            True if evaluation should run
        """
        if not self.config.enabled:
            return False
        
        if not self.last_evaluation_time:
            return True
        
        time_since_last = datetime.now() - self.last_evaluation_time
        return time_since_last.total_seconds() >= self.config.evaluation_interval_hours * 3600
    
    def create_result(self, 
                     component_name: str,
                     metric_type: MetricType,
                     value: Union[float, int, str],
                     metadata: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Create a standardized evaluation result.
        
        Args:
            component_name: Name of the component being evaluated
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata
            
        Returns:
            EvaluationResult instance
        """
        return EvaluationResult(
            evaluator_name=self.config.evaluator_name,
            component_name=component_name,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata or {}
        ) 