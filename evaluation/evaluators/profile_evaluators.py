"""
Client Profile Evaluators.

This module provides evaluation capabilities for client profiling components,
focusing on profile completeness, freshness, and data consistency.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult, MetricType


class ProfileEvaluator(BaseEvaluator):
    """
    Evaluator for client profiling components.
    
    Measures:
    - Profile completeness
    - Profile accuracy
    - Update frequency (freshness)
    - Consistency across data sources
    - Profile confidence scores
    """
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        
        # Essential profile fields for completeness evaluation
        self.essential_fields = [
            "client_name", "sector", "risk_tolerance", "investment_objectives",
            "geographic_preferences", "asset_class_preferences", "aum"
        ]
        
        # Optional fields that enhance profile quality
        self.optional_fields = [
            "investment_themes", "client_type", "regulatory_constraints",
            "investment_horizon", "liquidity_preferences", "esg_preferences"
        ]
        
        # Freshness thresholds (in days)
        self.freshness_thresholds = {
            'excellent': 7,   # Updated within a week
            'good': 30,       # Updated within a month
            'fair': 90,       # Updated within 3 months
            'poor': 180       # Updated within 6 months
        }
    
    def evaluate(self, data: Dict[str, Any], **kwargs) -> List[EvaluationResult]:
        """
        Evaluate client profiling quality.
        
        Args:
            data: Dictionary containing:
                - client_id: Client identifier
                - profile: Client profile data
                - data_sources: List of data sources used
                - last_updated: Last update timestamp
                - confidence_scores: Confidence in profile attributes
                - validation_results: Results from profile validation
                
        Returns:
            List of evaluation results
        """
        results = []
        client_id = data.get("client_id", "unknown_client")
        
        # Evaluate completeness
        completeness = self._evaluate_completeness(data)
        results.append(self.create_result(
            f"profile_{client_id}", MetricType.COVERAGE, completeness,
            {"evaluation_type": "profile_completeness"}
        ))
        
        # Evaluate freshness
        freshness = self._evaluate_freshness(data)
        freshness_category = self._categorize_freshness(data)
        results.append(self.create_result(
            f"profile_{client_id}", MetricType.RELEVANCE, freshness,
            {
                "evaluation_type": "profile_freshness",
                "freshness_category": freshness_category,
                "last_updated": data.get("last_updated")
            }
        ))
        
        # Evaluate consistency across sources
        if "data_sources" in data and len(data["data_sources"]) > 1:
            consistency = self._evaluate_consistency_across_sources(data)
            results.append(self.create_result(
                f"profile_{client_id}", MetricType.PRECISION, consistency,
                {
                    "evaluation_type": "cross_source_consistency",
                    "num_sources": len(data["data_sources"])
                }
            ))
        
        # Evaluate confidence scores
        if "confidence_scores" in data:
            avg_confidence = np.mean(list(data["confidence_scores"].values()))
            results.append(self.create_result(
                f"profile_{client_id}", MetricType.ACCURACY, avg_confidence,
                {
                    "evaluation_type": "profile_confidence",
                    "confidence_breakdown": data["confidence_scores"]
                }
            ))
        
        # Evaluate enrichment quality
        enrichment_score = self._evaluate_enrichment(data)
        results.append(self.create_result(
            f"profile_{client_id}", MetricType.COVERAGE, enrichment_score,
            {"evaluation_type": "profile_enrichment"}
        ))
        
        # Evaluate validation results if available
        if "validation_results" in data:
            validation_score = self._evaluate_validation_results(data["validation_results"])
            results.append(self.create_result(
                f"profile_{client_id}", MetricType.ACCURACY, validation_score,
                {
                    "evaluation_type": "profile_validation",
                    "validation_details": data["validation_results"]
                }
            ))
        
        return results
    
    def _evaluate_completeness(self, data: Dict[str, Any]) -> float:
        """Evaluate completeness of client profile."""
        profile = data.get("profile", {})
        
        # Check essential fields
        completed_essential = sum(1 for field in self.essential_fields 
                                 if profile.get(field) is not None and profile.get(field) != "")
        essential_score = completed_essential / len(self.essential_fields)
        
        return essential_score
    
    def _evaluate_enrichment(self, data: Dict[str, Any]) -> float:
        """Evaluate profile enrichment with optional fields."""
        profile = data.get("profile", {})
        
        # Check optional fields for enrichment
        completed_optional = sum(1 for field in self.optional_fields 
                               if profile.get(field) is not None and profile.get(field) != "")
        optional_score = completed_optional / len(self.optional_fields)
        
        # Combined score: 70% essential, 30% optional
        essential_score = self._evaluate_completeness(data)
        return 0.7 * essential_score + 0.3 * optional_score
    
    def _evaluate_freshness(self, data: Dict[str, Any]) -> float:
        """Evaluate freshness of profile data."""
        last_updated = data.get("last_updated")
        
        if not last_updated:
            return 0.0
        
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            except ValueError:
                self.logger.warning(f"Invalid date format: {last_updated}")
                return 0.0
        
        # Calculate days since last update
        days_since_update = (datetime.now(last_updated.tzinfo) - last_updated).days
        
        # Freshness score based on thresholds
        if days_since_update <= self.freshness_thresholds['excellent']:
            return 1.0
        elif days_since_update <= self.freshness_thresholds['good']:
            return 0.8
        elif days_since_update <= self.freshness_thresholds['fair']:
            return 0.6
        elif days_since_update <= self.freshness_thresholds['poor']:
            return 0.4
        else:
            return 0.2
    
    def _categorize_freshness(self, data: Dict[str, Any]) -> str:
        """Categorize freshness level."""
        last_updated = data.get("last_updated")
        
        if not last_updated:
            return "unknown"
        
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            except ValueError:
                return "unknown"
        
        days_since_update = (datetime.now(last_updated.tzinfo) - last_updated).days
        
        for category, threshold in self.freshness_thresholds.items():
            if days_since_update <= threshold:
                return category
        
        return "very_poor"
    
    def _evaluate_consistency_across_sources(self, data: Dict[str, Any]) -> float:
        """Evaluate consistency of profile data across different sources."""
        profile = data.get("profile", {})
        data_sources = data.get("data_sources", [])
        
        if len(data_sources) < 2:
            return 1.0
        
        # Check for source-specific data if available
        source_profiles = data.get("source_profiles", {})
        
        if not source_profiles:
            # Simple consistency check based on confidence scores
            confidence_scores = data.get("confidence_scores", {})
            if confidence_scores:
                # Higher average confidence suggests better consistency
                return min(np.mean(list(confidence_scores.values())), 1.0)
            return 0.8  # Default moderate consistency
        
        # Compare key attributes across sources
        consistency_scores = []
        key_attributes = ["sector", "risk_tolerance", "aum", "investment_objectives"]
        
        for attr in key_attributes:
            if attr in profile:
                attr_values = []
                for source, source_profile in source_profiles.items():
                    if attr in source_profile:
                        attr_values.append(source_profile[attr])
                
                if len(attr_values) > 1:
                    # Simple consistency: check if all values are the same
                    consistency = 1.0 if len(set(str(v) for v in attr_values)) == 1 else 0.5
                    consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _evaluate_validation_results(self, validation_results: Dict[str, Any]) -> float:
        """Evaluate profile validation results."""
        if not validation_results:
            return 0.0
        
        total_validations = validation_results.get('total_validations', 0)
        passed_validations = validation_results.get('passed_validations', 0)
        
        if total_validations == 0:
            return 1.0
        
        return passed_validations / total_validations
    
    def get_profile_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a comprehensive summary of profile evaluation.
        
        Args:
            data: Profile evaluation data
            
        Returns:
            Dictionary with profile evaluation summary
        """
        profile = data.get("profile", {})
        
        # Field completion analysis
        essential_completion = {}
        for field in self.essential_fields:
            essential_completion[field] = {
                'completed': field in profile and profile[field] is not None and profile[field] != "",
                'value': profile.get(field, "Not provided")
            }
        
        optional_completion = {}
        for field in self.optional_fields:
            optional_completion[field] = {
                'completed': field in profile and profile[field] is not None and profile[field] != "",
                'value': profile.get(field, "Not provided")
            }
        
        summary = {
            'profile_completeness': {
                'essential_fields': essential_completion,
                'optional_fields': optional_completion,
                'completeness_score': self._evaluate_completeness(data),
                'enrichment_score': self._evaluate_enrichment(data)
            },
            'profile_freshness': {
                'last_updated': data.get("last_updated"),
                'freshness_score': self._evaluate_freshness(data),
                'freshness_category': self._categorize_freshness(data)
            },
            'data_sources': data.get("data_sources", []),
            'confidence_scores': data.get("confidence_scores", {}),
            'validation_status': data.get("validation_results", {})
        }
        
        # Add consistency evaluation if multiple sources
        if len(data.get("data_sources", [])) > 1:
            summary['consistency_score'] = self._evaluate_consistency_across_sources(data)
        
        return summary
    
    def get_improvement_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get recommendations for improving profile quality.
        
        Args:
            data: Profile evaluation data
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        profile = data.get("profile", {})
        
        # Check for missing essential fields
        missing_essential = [field for field in self.essential_fields 
                           if not profile.get(field)]
        
        if missing_essential:
            recommendations.append({
                'type': 'completeness',
                'priority': 'high',
                'description': f'Complete missing essential fields: {", ".join(missing_essential)}',
                'fields': missing_essential
            })
        
        # Check for missing optional fields
        missing_optional = [field for field in self.optional_fields 
                          if not profile.get(field)]
        
        if missing_optional:
            recommendations.append({
                'type': 'enrichment',
                'priority': 'medium',
                'description': f'Consider adding optional fields for enrichment: {", ".join(missing_optional[:3])}',
                'fields': missing_optional[:3]  # Top 3 recommendations
            })
        
        # Check freshness
        freshness_category = self._categorize_freshness(data)
        if freshness_category in ['poor', 'very_poor']:
            recommendations.append({
                'type': 'freshness',
                'priority': 'high',
                'description': 'Profile data is outdated and should be refreshed',
                'last_updated': data.get("last_updated")
            })
        elif freshness_category == 'fair':
            recommendations.append({
                'type': 'freshness',
                'priority': 'medium',
                'description': 'Consider updating profile data soon',
                'last_updated': data.get("last_updated")
            })
        
        # Check confidence scores
        confidence_scores = data.get("confidence_scores", {})
        low_confidence_fields = [field for field, score in confidence_scores.items() 
                               if score < 0.7]
        
        if low_confidence_fields:
            recommendations.append({
                'type': 'confidence',
                'priority': 'medium',
                'description': f'Verify and improve confidence for: {", ".join(low_confidence_fields)}',
                'fields': low_confidence_fields
            })
        
        return recommendations 