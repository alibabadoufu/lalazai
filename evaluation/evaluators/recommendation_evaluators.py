"""
Recommendation Engine Evaluators.

This module provides evaluation capabilities for recommendation components,
including both traditional metrics and LLM-as-a-judge evaluation for recommendation quality.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Set, Optional
from collections import Counter

from ..base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult, MetricType
from ...llm_services.llm_client import LLMClient
from .llm_evaluators import LLMJudge, JudgmentCriteria, JudgmentPrompt, JudgmentType


class RecommendationEvaluator(BaseEvaluator):
    """
    Enhanced evaluator for recommendation engine components.
    
    Combines traditional recommendation metrics with LLM-as-a-judge evaluation:
    - Traditional metrics: Relevance, diversity, coverage, ranking quality
    - LLM judges: Personalization quality, business relevance, content appropriateness
    
    Measures:
    - Recommendation relevance
    - Diversity of recommendations
    - Coverage of client interests  
    - Ranking quality
    - Personalization effectiveness (LLM judge)
    - Business relevance (LLM judge)
    """
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.llm_client = LLMClient()
        self.judges = {}
        self._initialize_llm_judges()
    
    def _initialize_llm_judges(self):
        """Initialize LLM judges for qualitative recommendation assessment."""
        
        # Personalization Judge
        personalization_criteria = JudgmentCriteria(
            name="personalization",
            description="How well recommendations are personalized to the specific client's profile and preferences",
            judgment_type=JudgmentType.SCALE,
            scale_range=(1, 5)
        )
        
        personalization_prompt = JudgmentPrompt(
            system_prompt="""You are an expert financial advisor evaluating the personalization quality of investment recommendations.
Your task is to assess how well the recommendations match the specific client's profile, risk tolerance, and investment objectives.

Guidelines:
- Consider client's sector preferences, risk tolerance, investment objectives
- Evaluate if recommendations align with client's AUM, geographic preferences, and investment themes
- Assess whether the recommendations show understanding of client's unique situation
- Consider appropriateness of recommendation complexity for client sophistication level""",
            
            evaluation_template="""## Evaluation Task: {criteria_name}

**Criteria**: {criteria_description}

**Client Profile**:
{input_text}

**Recommendations**:
{output_text}

Please evaluate the personalization quality on a scale of 1-5:
- 1: Generic recommendations, no personalization evident
- 2: Minimal personalization, few client-specific considerations
- 3: Moderate personalization, some alignment with client profile
- 4: Good personalization, recommendations well-matched to client
- 5: Excellent personalization, recommendations perfectly tailored to client

Think step by step and provide your reasoning before giving the final score.

{output_format}""",
            
            examples=[],
            
            output_format="""Respond in JSON format:
{
    "score": <1-5>,
    "reasoning": "<detailed explanation>",
    "confidence": <0-1>,
    "personalization_analysis": {
        "matched_preferences": ["list of client preferences that recommendations match"],
        "missed_opportunities": ["list of personalization opportunities missed"],
        "appropriateness_factors": ["factors making recommendations appropriate for this client"]
    }
}"""
        )
        
        self.judges['personalization'] = LLMJudge(
            self.llm_client, personalization_criteria, personalization_prompt, num_runs=2
        )
        
        # Business Relevance Judge
        business_relevance_criteria = JudgmentCriteria(
            name="business_relevance",
            description="How relevant the recommendations are for the client's business context and current market conditions",
            judgment_type=JudgmentType.SCALE,
            scale_range=(1, 5)
        )
        
        business_relevance_prompt = JudgmentPrompt(
            system_prompt="""You are an expert investment strategist evaluating the business relevance of financial recommendations.
Your task is to assess how well recommendations align with current market conditions and the client's business context.

Guidelines:
- Consider current market trends and economic conditions
- Evaluate relevance to client's industry and business cycle
- Assess timing appropriateness of recommendations
- Consider regulatory environment and compliance factors""",
            
            evaluation_template="""## Evaluation Task: {criteria_name}

**Criteria**: {criteria_description}

**Client Business Context**:
{input_text}

**Recommendations**:
{output_text}

Please evaluate the business relevance on a scale of 1-5:
- 1: Poor business relevance, recommendations don't fit current context
- 2: Limited relevance, few recommendations aligned with business needs
- 3: Moderate relevance, some recommendations appropriate for business context
- 4: Good relevance, most recommendations well-suited to business situation
- 5: Excellent relevance, all recommendations highly appropriate for business context

Think step by step and provide your reasoning before giving the final score.

{output_format}""",
            
            examples=[],
            
            output_format="""Respond in JSON format:
{
    "score": <1-5>,
    "reasoning": "<detailed explanation>",
    "confidence": <0-1>,
    "business_analysis": {
        "market_alignment": ["how recommendations align with market conditions"],
        "industry_fit": ["relevance to client's industry sector"],
        "timing_assessment": ["appropriateness of recommendation timing"]
    }
}"""
        )
        
        self.judges['business_relevance'] = LLMJudge(
            self.llm_client, business_relevance_criteria, business_relevance_prompt, num_runs=2
        )
    
    def evaluate(self, data: Dict[str, Any], **kwargs) -> List[EvaluationResult]:
        """
        Evaluate recommendation quality using both traditional metrics and LLM judges.
        
        Args:
            data: Dictionary containing:
                - client_id: Client identifier
                - recommendations: List of recommended publications
                - client_profile: Client profile information
                - ranking_scores: Relevance scores for recommendations
                - ground_truth_preferences: Known client preferences (optional)
                
        Returns:
            List of evaluation results
        """
        results = []
        client_id = data.get("client_id", "unknown_client")
        
        # Traditional metrics
        relevance = self._evaluate_relevance(data)
        results.append(self.create_result(
            f"recommendations_{client_id}", MetricType.RELEVANCE, relevance,
            {"evaluation_type": "recommendation_relevance"}
        ))
        
        diversity = self._evaluate_diversity(data)
        results.append(self.create_result(
            f"recommendations_{client_id}", MetricType.DIVERSITY, diversity,
            {"evaluation_type": "recommendation_diversity"}
        ))
        
        coverage = self._evaluate_coverage(data)
        results.append(self.create_result(
            f"recommendations_{client_id}", MetricType.COVERAGE, coverage,
            {"evaluation_type": "client_interest_coverage"}
        ))
        
        if "ranking_scores" in data:
            ranking_quality = self._evaluate_ranking_quality(data)
            results.append(self.create_result(
                f"recommendations_{client_id}", MetricType.PRECISION, ranking_quality,
                {"evaluation_type": "ranking_quality"}
            ))
        
        # LLM-based evaluation
        if data.get("client_profile") and data.get("recommendations"):
            llm_results = self._evaluate_with_llm_judges(data)
            results.extend(llm_results)
        
        return results
    
    def _evaluate_with_llm_judges(self, data: Dict[str, Any]) -> List[EvaluationResult]:
        """Evaluate recommendations using LLM judges."""
        results = []
        client_id = data.get("client_id", "unknown_client")
        
        # Prepare data for LLM judges
        input_data = {
            "client_profile": data.get("client_profile", {}),
            "client_business_context": {
                "sector": data.get("client_profile", {}).get("sector"),
                "aum": data.get("client_profile", {}).get("aum"),
                "risk_tolerance": data.get("client_profile", {}).get("risk_tolerance"),
                "investment_objectives": data.get("client_profile", {}).get("investment_objectives")
            }
        }
        
        output_data = {
            "recommendations": data.get("recommendations", [])
        }
        
        # Run each LLM judge
        for judge_name, judge in self.judges.items():
            try:
                self.logger.info(f"Running {judge_name} judge for client {client_id}")
                
                judgment = judge.judge(input_data, output_data)
                
                # Convert to evaluation result
                result = self.create_result(
                    f"recommendations_{client_id}",
                    MetricType.BUSINESS_IMPACT if judge_name == "business_relevance" else MetricType.ENGAGEMENT,
                    judgment.get('score', 0.0),
                    {
                        "evaluation_type": f"llm_judge_{judge_name}",
                        "reasoning": judgment.get('reasoning', ''),
                        "confidence": judgment.get('confidence', 0.0),
                        "consistency_score": judgment.get('consistency_score', 1.0),
                        "judge_metadata": judgment
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"LLM judge {judge_name} failed: {e}")
                error_result = self.create_result(
                    f"recommendations_{client_id}",
                    MetricType.BUSINESS_IMPACT,
                    0.0,
                    {
                        "evaluation_type": f"llm_judge_{judge_name}_error",
                        "error": str(e)
                    }
                )
                results.append(error_result)
        
        return results
    
    def _evaluate_relevance(self, data: Dict[str, Any]) -> float:
        """Evaluate relevance of recommendations to client profile using traditional metrics."""
        recommendations = data.get("recommendations", [])
        client_profile = data.get("client_profile", {})
        
        if not recommendations or not client_profile:
            return 0.0
        
        # Extract client interests
        client_sectors = set(client_profile.get("sectors", []))
        client_themes = set(client_profile.get("investment_themes", []))
        
        relevance_scores = []
        
        for rec in recommendations:
            score = 0.0
            
            # Check sector alignment
            rec_sectors = set(rec.get("sectors", []))
            if client_sectors and rec_sectors:
                sector_overlap = len(client_sectors.intersection(rec_sectors))
                score += 0.6 * (sector_overlap / len(client_sectors))
            
            # Check theme alignment
            rec_themes = set(rec.get("themes", []))
            if client_themes and rec_themes:
                theme_overlap = len(client_themes.intersection(rec_themes))
                score += 0.4 * (theme_overlap / len(client_themes))
            
            relevance_scores.append(min(score, 1.0))
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def _evaluate_diversity(self, data: Dict[str, Any]) -> float:
        """Evaluate diversity of recommendations."""
        recommendations = data.get("recommendations", [])
        
        if len(recommendations) <= 1:
            return 1.0
        
        # Calculate diversity based on different attributes
        sectors = [rec.get("sector", "unknown") for rec in recommendations]
        authors = [rec.get("author", "unknown") for rec in recommendations]
        pub_types = [rec.get("publication_type", "unknown") for rec in recommendations]
        
        # Calculate entropy for each attribute
        sector_diversity = self._calculate_entropy(sectors)
        author_diversity = self._calculate_entropy(authors)  
        type_diversity = self._calculate_entropy(pub_types)
        
        # Weighted average diversity across attributes
        return (sector_diversity * 0.5 + author_diversity * 0.3 + type_diversity * 0.2)
    
    def _calculate_entropy(self, items: List[str]) -> float:
        """Calculate entropy (diversity measure) for a list of items."""
        if not items:
            return 0.0
        
        counts = Counter(items)
        total = len(items)
        
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _evaluate_coverage(self, data: Dict[str, Any]) -> float:
        """Evaluate how well recommendations cover client interests."""
        recommendations = data.get("recommendations", [])
        client_profile = data.get("client_profile", {})
        
        if not recommendations or not client_profile:
            return 0.0
        
        # Get all client interests
        client_interests = set()
        client_interests.update(client_profile.get("sectors", []))
        client_interests.update(client_profile.get("investment_themes", []))
        client_interests.update(client_profile.get("geographic_preferences", []))
        
        if not client_interests:
            return 1.0  # No specific interests to cover
        
        # Get all topics covered by recommendations
        covered_topics = set()
        for rec in recommendations:
            covered_topics.update(rec.get("sectors", []))
            covered_topics.update(rec.get("themes", []))
            covered_topics.update(rec.get("regions", []))
        
        # Calculate coverage
        covered_interests = client_interests.intersection(covered_topics)
        return len(covered_interests) / len(client_interests)
    
    def _evaluate_ranking_quality(self, data: Dict[str, Any]) -> float:
        """Evaluate quality of recommendation ranking."""
        ranking_scores = data.get("ranking_scores", [])
        
        if len(ranking_scores) <= 1:
            return 1.0
        
        # Check if scores are in descending order (higher scores should be ranked first)
        correctly_ordered = 0
        total_pairs = 0
        
        for i in range(len(ranking_scores)):
            for j in range(i + 1, len(ranking_scores)):
                if ranking_scores[i] >= ranking_scores[j]:
                    correctly_ordered += 1
                total_pairs += 1
        
        return correctly_ordered / total_pairs if total_pairs > 0 else 1.0
    
    def get_recommendation_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a comprehensive summary of recommendation evaluation.
        
        Args:
            data: Recommendation evaluation data
            
        Returns:
            Dictionary with evaluation summary
        """
        recommendations = data.get("recommendations", [])
        
        summary = {
            'total_recommendations': len(recommendations),
            'traditional_metrics': {
                'relevance': self._evaluate_relevance(data),
                'diversity': self._evaluate_diversity(data),
                'coverage': self._evaluate_coverage(data)
            },
            'recommendation_breakdown': {
                'sectors': list(set(rec.get("sector", "unknown") for rec in recommendations)),
                'publication_types': list(set(rec.get("publication_type", "unknown") for rec in recommendations)),
                'authors': list(set(rec.get("author", "unknown") for rec in recommendations))
            }
        }
        
        if "ranking_scores" in data:
            summary['traditional_metrics']['ranking_quality'] = self._evaluate_ranking_quality(data)
        
        return summary 