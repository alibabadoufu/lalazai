"""
Offline Evaluator for the Intelligent Research Recommendation System.
Provides comprehensive batch evaluation of all pipeline components.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult, MetricType, EvaluationStatus
from .component_evaluators import (
    LLMFeatureEvaluator,
    ConnectorEvaluator,
    RecommendationEvaluator,
    ProfileEvaluator
)


class OfflineEvaluator:
    """
    Comprehensive offline evaluation framework for the recommendation system.
    
    Orchestrates evaluation of all pipeline components including:
    - Data connectors
    - LLM feature extractors
    - Recommendation engine
    - Client profiling
    - End-to-end pipeline performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the offline evaluator.
        
        Args:
            config: Configuration dictionary containing evaluation settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.evaluators: Dict[str, BaseEvaluator] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Initialize component evaluators
        self._initialize_evaluators()
        
        # Load ground truth data if available
        self.ground_truth_data = self._load_ground_truth_data()
        
        self.logger.info("Offline evaluator initialized with {} component evaluators".format(
            len(self.evaluators)
        ))
    
    def _initialize_evaluators(self):
        """Initialize all component evaluators."""
        evaluator_configs = self.config.get("evaluators", {})
        
        # LLM Feature Evaluators
        for connector_name in ["bloomberg_chat", "refinitiv_chat", "crm", "rfq", "kyc", "readership"]:
            if evaluator_configs.get(f"llm_{connector_name}", {}).get("enabled", True):
                config = EvaluationConfig(
                    evaluator_name=f"llm_{connector_name}",
                    enabled=True,
                    evaluation_interval_hours=evaluator_configs.get(f"llm_{connector_name}", {}).get("interval", 24),
                    custom_parameters=evaluator_configs.get(f"llm_{connector_name}", {})
                )
                self.evaluators[f"llm_{connector_name}"] = LLMFeatureEvaluator(config)
        
        # Connector Evaluators
        for connector_name in ["bloomberg_chat", "refinitiv_chat", "publications", "crm", "rfq", "kyc", "readership"]:
            if evaluator_configs.get(f"connector_{connector_name}", {}).get("enabled", True):
                config = EvaluationConfig(
                    evaluator_name=f"connector_{connector_name}",
                    enabled=True,
                    evaluation_interval_hours=evaluator_configs.get(f"connector_{connector_name}", {}).get("interval", 12),
                    custom_parameters=evaluator_configs.get(f"connector_{connector_name}", {})
                )
                self.evaluators[f"connector_{connector_name}"] = ConnectorEvaluator(config)
        
        # Recommendation Evaluator
        if evaluator_configs.get("recommendations", {}).get("enabled", True):
            config = EvaluationConfig(
                evaluator_name="recommendations",
                enabled=True,
                evaluation_interval_hours=evaluator_configs.get("recommendations", {}).get("interval", 6),
                custom_parameters=evaluator_configs.get("recommendations", {})
            )
            self.evaluators["recommendations"] = RecommendationEvaluator(config)
        
        # Profile Evaluator
        if evaluator_configs.get("profiling", {}).get("enabled", True):
            config = EvaluationConfig(
                evaluator_name="profiling",
                enabled=True,
                evaluation_interval_hours=evaluator_configs.get("profiling", {}).get("interval", 24),
                custom_parameters=evaluator_configs.get("profiling", {})
            )
            self.evaluators["profiling"] = ProfileEvaluator(config)
    
    def _load_ground_truth_data(self) -> Dict[str, Any]:
        """Load ground truth data for evaluation."""
        ground_truth_path = self.config.get("ground_truth_path")
        if not ground_truth_path or not os.path.exists(ground_truth_path):
            self.logger.warning("No ground truth data available for evaluation")
            return {}
        
        try:
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
            self.logger.info(f"Loaded ground truth data from {ground_truth_path}")
            return ground_truth
        except Exception as e:
            self.logger.error(f"Failed to load ground truth data: {e}")
            return {}
    
    def run_full_evaluation(self, 
                           pipeline_data: Dict[str, Any],
                           parallel: bool = True) -> Dict[str, List[EvaluationResult]]:
        """
        Run comprehensive evaluation of all pipeline components.
        
        Args:
            pipeline_data: Complete pipeline execution data
            parallel: Whether to run evaluations in parallel
            
        Returns:
            Dictionary mapping evaluator names to their results
        """
        self.logger.info("Starting full pipeline evaluation")
        start_time = datetime.now()
        
        evaluation_results = {}
        
        if parallel:
            evaluation_results = self._run_parallel_evaluation(pipeline_data)
        else:
            evaluation_results = self._run_sequential_evaluation(pipeline_data)
        
        # Store evaluation run metadata
        evaluation_run = {
            "timestamp": start_time.isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "evaluators_run": list(evaluation_results.keys()),
            "total_metrics": sum(len(results) for results in evaluation_results.values()),
            "pipeline_data_summary": self._summarize_pipeline_data(pipeline_data)
        }
        
        self.evaluation_history.append(evaluation_run)
        
        self.logger.info(f"Full evaluation completed in {evaluation_run['duration_seconds']:.2f} seconds")
        return evaluation_results
    
    def _run_parallel_evaluation(self, pipeline_data: Dict[str, Any]) -> Dict[str, List[EvaluationResult]]:
        """Run evaluations in parallel using ThreadPoolExecutor."""
        evaluation_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit evaluation tasks
            future_to_evaluator = {}
            
            for evaluator_name, evaluator in self.evaluators.items():
                if evaluator.should_run_evaluation():
                    component_data = self._extract_component_data(pipeline_data, evaluator_name)
                    if component_data:
                        future = executor.submit(evaluator.run_evaluation, component_data)
                        future_to_evaluator[future] = evaluator_name
            
            # Collect results
            for future in as_completed(future_to_evaluator):
                evaluator_name = future_to_evaluator[future]
                try:
                    results = future.result()
                    evaluation_results[evaluator_name] = results
                    self.logger.info(f"Completed evaluation for {evaluator_name}: {len(results)} metrics")
                except Exception as e:
                    self.logger.error(f"Evaluation failed for {evaluator_name}: {e}")
                    evaluation_results[evaluator_name] = []
        
        return evaluation_results
    
    def _run_sequential_evaluation(self, pipeline_data: Dict[str, Any]) -> Dict[str, List[EvaluationResult]]:
        """Run evaluations sequentially."""
        evaluation_results = {}
        
        for evaluator_name, evaluator in self.evaluators.items():
            if evaluator.should_run_evaluation():
                component_data = self._extract_component_data(pipeline_data, evaluator_name)
                if component_data:
                    try:
                        results = evaluator.run_evaluation(component_data)
                        evaluation_results[evaluator_name] = results
                        self.logger.info(f"Completed evaluation for {evaluator_name}: {len(results)} metrics")
                    except Exception as e:
                        self.logger.error(f"Evaluation failed for {evaluator_name}: {e}")
                        evaluation_results[evaluator_name] = []
        
        return evaluation_results
    
    def _extract_component_data(self, pipeline_data: Dict[str, Any], evaluator_name: str) -> Optional[Dict[str, Any]]:
        """Extract relevant data for a specific evaluator."""
        if evaluator_name.startswith("llm_"):
            # Extract LLM feature extraction data
            connector_name = evaluator_name[4:]  # Remove "llm_" prefix
            return pipeline_data.get("llm_features", {}).get(connector_name, {})
        
        elif evaluator_name.startswith("connector_"):
            # Extract connector performance data
            connector_name = evaluator_name[10:]  # Remove "connector_" prefix
            return pipeline_data.get("connectors", {}).get(connector_name, {})
        
        elif evaluator_name == "recommendations":
            # Extract recommendation data
            return pipeline_data.get("recommendations", {})
        
        elif evaluator_name == "profiling":
            # Extract profiling data
            return pipeline_data.get("profiling", {})
        
        else:
            self.logger.warning(f"Unknown evaluator type: {evaluator_name}")
            return None
    
    def _summarize_pipeline_data(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of pipeline data for logging."""
        summary = {}
        
        # Summarize connectors
        connectors = pipeline_data.get("connectors", {})
        summary["connectors"] = {
            name: {
                "records_processed": data.get("ingested_records", 0),
                "processing_time": data.get("processing_time", 0),
                "errors": len(data.get("errors", []))
            }
            for name, data in connectors.items()
        }
        
        # Summarize recommendations
        recommendations = pipeline_data.get("recommendations", {})
        summary["recommendations"] = {
            "total_clients": len(recommendations),
            "total_recommendations": sum(len(recs.get("recommendations", [])) 
                                       for recs in recommendations.values())
        }
        
        # Summarize LLM features
        llm_features = pipeline_data.get("llm_features", {})
        summary["llm_features"] = {
            name: {
                "features_extracted": len(data.get("extracted_features", {})),
                "token_usage": data.get("token_usage", 0)
            }
            for name, data in llm_features.items()
        }
        
        return summary
    
    def evaluate_component(self, 
                          component_name: str, 
                          component_data: Dict[str, Any]) -> List[EvaluationResult]:
        """
        Evaluate a specific component.
        
        Args:
            component_name: Name of the component to evaluate
            component_data: Data for the component
            
        Returns:
            List of evaluation results
        """
        if component_name not in self.evaluators:
            self.logger.error(f"No evaluator found for component: {component_name}")
            return []
        
        evaluator = self.evaluators[component_name]
        return evaluator.run_evaluation(component_data)
    
    def get_evaluation_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get summary of recent evaluations.
        
        Args:
            days_back: Number of days to include in summary
            
        Returns:
            Evaluation summary
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Get recent evaluation runs
        recent_runs = [
            run for run in self.evaluation_history
            if datetime.fromisoformat(run["timestamp"]) >= cutoff_date
        ]
        
        # Aggregate metrics from all evaluators
        all_metrics = {}
        for evaluator_name, evaluator in self.evaluators.items():
            recent_results = evaluator.get_historical_results(days_back)
            
            # Group by metric type
            for result in recent_results:
                metric_key = f"{evaluator_name}_{result.metric_type.value}"
                if metric_key not in all_metrics:
                    all_metrics[metric_key] = []
                all_metrics[metric_key].append(result.value)
        
        # Calculate summary statistics
        summary = {
            "evaluation_period_days": days_back,
            "total_evaluation_runs": len(recent_runs),
            "evaluators_active": len([e for e in self.evaluators.values() if e.config.enabled]),
            "metrics_summary": {}
        }
        
        for metric_key, values in all_metrics.items():
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                summary["metrics_summary"][metric_key] = {
                    "count": len(numeric_values),
                    "average": sum(numeric_values) / len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "latest": numeric_values[-1] if numeric_values else None
                }
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all evaluators.
        
        Returns:
            Health status summary
        """
        health_status = {
            "overall_status": "healthy",
            "evaluators": {},
            "last_full_evaluation": None,
            "total_evaluators": len(self.evaluators)
        }
        
        failed_evaluators = 0
        
        for evaluator_name, evaluator in self.evaluators.items():
            evaluator_health = evaluator.health_check()
            health_status["evaluators"][evaluator_name] = evaluator_health
            
            if evaluator_health["status"] == "failed":
                failed_evaluators += 1
        
        # Determine overall status
        if failed_evaluators > len(self.evaluators) * 0.5:
            health_status["overall_status"] = "critical"
        elif failed_evaluators > 0:
            health_status["overall_status"] = "degraded"
        
        # Get last full evaluation
        if self.evaluation_history:
            health_status["last_full_evaluation"] = self.evaluation_history[-1]["timestamp"]
        
        return health_status
    
    def export_evaluation_report(self, 
                                days_back: int = 30,
                                format: str = "json") -> str:
        """
        Export comprehensive evaluation report.
        
        Args:
            days_back: Number of days to include
            format: Export format (json, html)
            
        Returns:
            Formatted evaluation report
        """
        summary = self.get_evaluation_summary(days_back)
        health_status = self.get_health_status()
        
        # Collect detailed results from all evaluators
        detailed_results = {}
        for evaluator_name, evaluator in self.evaluators.items():
            results = evaluator.get_historical_results(days_back)
            detailed_results[evaluator_name] = [result.to_dict() for result in results]
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "evaluation_period_days": days_back,
            "summary": summary,
            "health_status": health_status,
            "detailed_results": detailed_results,
            "evaluation_history": self.evaluation_history[-10:]  # Last 10 runs
        }
        
        if format.lower() == "json":
            return json.dumps(report, indent=2)
        elif format.lower() == "html":
            return self._generate_html_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML evaluation report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Recommendation System Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .status-healthy {{ color: green; }}
                .status-degraded {{ color: orange; }}
                .status-critical {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Recommendation System Evaluation Report</h1>
                <p>Generated: {report_data['report_generated']}</p>
                <p>Evaluation Period: {report_data['evaluation_period_days']} days</p>
            </div>
            
            <div class="section">
                <h2>Health Status</h2>
                <p class="status-{report_data['health_status']['overall_status']}">
                    Overall Status: {report_data['health_status']['overall_status'].upper()}
                </p>
                <p>Active Evaluators: {report_data['health_status']['total_evaluators']}</p>
            </div>
            
            <div class="section">
                <h2>Metrics Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Count</th><th>Average</th><th>Min</th><th>Max</th><th>Latest</th></tr>
        """
        
        for metric_name, metric_data in report_data['summary']['metrics_summary'].items():
            html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{metric_data['count']}</td>
                        <td>{metric_data['average']:.3f}</td>
                        <td>{metric_data['min']:.3f}</td>
                        <td>{metric_data['max']:.3f}</td>
                        <td>{metric_data['latest']:.3f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def add_evaluator(self, name: str, evaluator: BaseEvaluator):
        """
        Add a custom evaluator.
        
        Args:
            name: Name of the evaluator
            evaluator: Evaluator instance
        """
        self.evaluators[name] = evaluator
        self.logger.info(f"Added custom evaluator: {name}")
    
    def remove_evaluator(self, name: str):
        """
        Remove an evaluator.
        
        Args:
            name: Name of the evaluator to remove
        """
        if name in self.evaluators:
            del self.evaluators[name]
            self.logger.info(f"Removed evaluator: {name}")
        else:
            self.logger.warning(f"Evaluator not found: {name}")
