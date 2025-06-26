"""
Data Connector Evaluators.

This module provides evaluation capabilities for data ingestion and connector components,
focusing on ingestion success rates, data quality, and error handling.
"""

import logging
from typing import Dict, Any, List

from ..base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult, MetricType


class ConnectorEvaluator(BaseEvaluator):
    """
    Evaluator for data connector components.
    
    Measures:
    - Ingestion success rate
    - Data throughput
    - Data quality metrics
    - Error rates and handling
    - Processing latency
    """
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.quality_thresholds = {
            'completeness': 0.95,
            'validity': 0.90,
            'consistency': 0.85,
            'timeliness': 0.80
        }
    
    def evaluate(self, data: Dict[str, Any], **kwargs) -> List[EvaluationResult]:
        """
        Evaluate data connector performance.
        
        Args:
            data: Dictionary containing:
                - connector_name: Name of the connector
                - ingested_records: Number of records ingested
                - processing_time: Time taken for ingestion
                - errors: List of errors encountered
                - data_quality_metrics: Quality assessment results
                - total_attempts: Total ingestion attempts
                - successful_ingestions: Successful ingestions
                - total_operations: Total operations performed
                
        Returns:
            List of evaluation results
        """
        results = []
        connector_name = data.get("connector_name", "unknown_connector")
        
        # Evaluate ingestion success rate
        success_rate = self._evaluate_success_rate(data)
        results.append(self.create_result(
            connector_name, MetricType.ACCURACY, success_rate,
            {"evaluation_type": "ingestion_success_rate"}
        ))
        
        # Evaluate throughput
        if "ingested_records" in data and "processing_time" in data:
            throughput = data["ingested_records"] / max(data["processing_time"], 0.001)
            results.append(self.create_result(
                connector_name, MetricType.THROUGHPUT, throughput,
                {"unit": "records_per_second"}
            ))
        
        # Evaluate data quality
        if "data_quality_metrics" in data:
            quality_score = self._evaluate_data_quality(data["data_quality_metrics"])
            results.append(self.create_result(
                connector_name, MetricType.RELEVANCE, quality_score,
                {"evaluation_type": "data_quality"}
            ))
            
            # Individual quality dimensions
            quality_metrics = data["data_quality_metrics"]
            for dimension, score in quality_metrics.items():
                if dimension in self.quality_thresholds:
                    results.append(self.create_result(
                        connector_name, MetricType.PRECISION, score,
                        {
                            "evaluation_type": f"data_quality_{dimension}",
                            "threshold": self.quality_thresholds[dimension],
                            "meets_threshold": score >= self.quality_thresholds[dimension]
                        }
                    ))
        
        # Evaluate error rate
        error_rate = self._evaluate_error_rate(data)
        results.append(self.create_result(
            connector_name, MetricType.PRECISION, 1.0 - error_rate,
            {"evaluation_type": "error_rate", "actual_error_rate": error_rate}
        ))
        
        # Evaluate processing latency
        if "processing_time" in data:
            results.append(self.create_result(
                connector_name, MetricType.LATENCY, data["processing_time"],
                {"unit": "seconds", "evaluation_type": "processing_latency"}
            ))
        
        return results
    
    def _evaluate_success_rate(self, data: Dict[str, Any]) -> float:
        """Calculate ingestion success rate."""
        total_attempts = data.get("total_attempts", 1)
        successful_ingestions = data.get("successful_ingestions", 0)
        
        return successful_ingestions / total_attempts if total_attempts > 0 else 0.0
    
    def _evaluate_data_quality(self, quality_metrics: Dict[str, Any]) -> float:
        """
        Evaluate overall data quality.
        
        Uses weighted average of quality dimensions with configurable weights.
        """
        quality_score = 0.0
        total_weight = 0.0
        
        # Define dimension weights (can be made configurable)
        dimension_weights = {
            'completeness': 0.3,
            'validity': 0.25,
            'consistency': 0.25,
            'timeliness': 0.2
        }
        
        for dimension, weight in dimension_weights.items():
            if dimension in quality_metrics:
                quality_score += quality_metrics[dimension] * weight
                total_weight += weight
        
        return quality_score / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_error_rate(self, data: Dict[str, Any]) -> float:
        """Calculate error rate."""
        errors = data.get("errors", [])
        total_operations = data.get("total_operations", 1)
        
        return len(errors) / total_operations if total_operations > 0 else 0.0
    
    def get_quality_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of data quality metrics.
        
        Args:
            data: Connector evaluation data
            
        Returns:
            Dictionary with quality summary
        """
        quality_metrics = data.get("data_quality_metrics", {})
        
        summary = {
            'overall_quality': self._evaluate_data_quality(quality_metrics),
            'dimensions': {},
            'failing_thresholds': []
        }
        
        for dimension, score in quality_metrics.items():
            if dimension in self.quality_thresholds:
                threshold = self.quality_thresholds[dimension]
                summary['dimensions'][dimension] = {
                    'score': score,
                    'threshold': threshold,
                    'status': 'pass' if score >= threshold else 'fail'
                }
                
                if score < threshold:
                    summary['failing_thresholds'].append({
                        'dimension': dimension,
                        'score': score,
                        'threshold': threshold,
                        'gap': threshold - score
                    })
        
        return summary 