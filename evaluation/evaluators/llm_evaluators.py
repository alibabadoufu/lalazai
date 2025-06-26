"""
LLM-based Evaluators using LLM-as-a-Judge methodology.

This module implements state-of-the-art LLM evaluation techniques including:
- LLM-as-a-Judge for qualitative assessment
- Multi-criteria evaluation with specialized judges
- Chain-of-thought reasoning for better accuracy
- Bias mitigation and consistency improvements
"""

import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult, MetricType
from ...llm_services.llm_client import LLMClient


class JudgmentType(Enum):
    """Types of LLM judgment."""
    BINARY = "binary"
    SCALE = "scale"
    CATEGORICAL = "categorical"
    PAIRWISE = "pairwise"


@dataclass
class JudgmentCriteria:
    """Criteria for LLM judgment."""
    name: str
    description: str
    judgment_type: JudgmentType
    scale_range: Optional[Tuple[int, int]] = None
    categories: Optional[List[str]] = None
    weight: float = 1.0


@dataclass
class JudgmentPrompt:
    """Structured prompt for LLM judge."""
    system_prompt: str
    evaluation_template: str
    examples: List[Dict[str, Any]]
    output_format: str


class LLMJudge:
    """
    LLM-as-a-Judge implementation for qualitative evaluation.
    
    Features:
    - Multiple judgment types (binary, scale, categorical, pairwise)
    - Chain-of-thought reasoning
    - Bias mitigation through prompt engineering
    - Consistency validation through multiple runs
    - Structured output parsing
    """
    
    def __init__(self, 
                 llm_client: LLMClient,
                 criteria: JudgmentCriteria,
                 prompt: JudgmentPrompt,
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 num_runs: int = 1):
        """
        Initialize LLM Judge.
        
        Args:
            llm_client: LLM client for making API calls
            criteria: Judgment criteria
            prompt: Structured prompt for evaluation
            temperature: Sampling temperature (low for consistency)
            max_tokens: Maximum tokens for response
            num_runs: Number of runs for consistency (>1 for ensemble)
        """
        self.llm_client = llm_client
        self.criteria = criteria
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_runs = num_runs
        self.logger = logging.getLogger(__name__)
    
    def judge(self, 
              input_data: Dict[str, Any], 
              output_data: Dict[str, Any],
              reference_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform LLM-based judgment.
        
        Args:
            input_data: Input to the system being evaluated
            output_data: Output from the system being evaluated  
            reference_data: Optional reference/ground truth data
            
        Returns:
            Dictionary containing judgment results
        """
        try:
            # Build evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(
                input_data, output_data, reference_data
            )
            
            # Run multiple evaluations for consistency
            judgments = []
            for i in range(self.num_runs):
                self.logger.debug(f"Running judgment {i+1}/{self.num_runs}")
                response = self.llm_client.generate_completion(
                    prompt=evaluation_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Parse structured output
                judgment = self._parse_judgment(response.content if hasattr(response, 'content') else response)
                if judgment:
                    judgments.append(judgment)
            
            if not judgments:
                raise ValueError("No valid judgments obtained")
            
            # Aggregate multiple judgments
            final_judgment = self._aggregate_judgments(judgments)
            
            # Add metadata
            final_judgment.update({
                'criteria_name': self.criteria.name,
                'judgment_type': self.criteria.judgment_type.value,
                'num_runs': len(judgments),
                'consistency_score': self._calculate_consistency(judgments) if len(judgments) > 1 else 1.0
            })
            
            return final_judgment
            
        except Exception as e:
            self.logger.error(f"LLM judgment failed: {e}")
            return {
                'score': 0.0,
                'reasoning': f"Judgment failed: {str(e)}",
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _build_evaluation_prompt(self, 
                                input_data: Dict[str, Any],
                                output_data: Dict[str, Any],
                                reference_data: Optional[Dict[str, Any]]) -> str:
        """Build the evaluation prompt from template."""
        
        # Format input, output, and reference data
        input_text = self._format_data_for_prompt(input_data)
        output_text = self._format_data_for_prompt(output_data)
        reference_text = self._format_data_for_prompt(reference_data) if reference_data else "N/A"
        
        # Build the complete prompt
        evaluation_text = self.prompt.evaluation_template.format(
            criteria_name=self.criteria.name,
            criteria_description=self.criteria.description,
            input_text=input_text,
            output_text=output_text,
            reference_text=reference_text,
            output_format=self.prompt.output_format
        )
        
        # Add examples if provided
        examples_text = ""
        if self.prompt.examples:
            examples_text = "\n\n## Examples:\n"
            for i, example in enumerate(self.prompt.examples, 1):
                examples_text += f"\n### Example {i}:\n"
                examples_text += f"Input: {example.get('input', 'N/A')}\n"
                examples_text += f"Output: {example.get('output', 'N/A')}\n"
                examples_text += f"Judgment: {example.get('judgment', 'N/A')}\n"
        
        full_prompt = f"{self.prompt.system_prompt}\n\n{examples_text}\n\n{evaluation_text}"
        
        return full_prompt
    
    def _format_data_for_prompt(self, data: Dict[str, Any]) -> str:
        """Format data dictionary for inclusion in prompt."""
        if not data:
            return "N/A"
        
        # Handle different data types appropriately
        formatted_parts = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                formatted_parts.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                formatted_parts.append(f"{key}: {str(value)}")
        
        return "\n".join(formatted_parts)
    
    def _parse_judgment(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse structured judgment from LLM response."""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Look for JSON block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif response_text.startswith("{") and response_text.endswith("}"):
                json_text = response_text
            else:
                # Try to extract structured information from text
                return self._parse_text_judgment(response_text)
            
            judgment = json.loads(json_text)
            
            # Validate required fields
            if 'score' not in judgment:
                self.logger.warning("No score found in judgment")
                return None
            
            # Normalize score based on judgment type
            judgment['score'] = self._normalize_score(judgment['score'])
            
            return judgment
            
        except Exception as e:
            self.logger.warning(f"Failed to parse judgment: {e}")
            return self._parse_text_judgment(response_text)
    
    def _parse_text_judgment(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse judgment from unstructured text response."""
        # Simple text parsing for fallback
        score = 0.0
        reasoning = response_text
        confidence = 0.5
        
        # Look for score indicators
        if self.criteria.judgment_type == JudgmentType.BINARY:
            if any(word in response_text.lower() for word in ['yes', 'correct', 'good', 'helpful', 'relevant']):
                score = 1.0
            elif any(word in response_text.lower() for word in ['no', 'incorrect', 'bad', 'unhelpful', 'irrelevant']):
                score = 0.0
        
        return {
            'score': score,
            'reasoning': reasoning,
            'confidence': confidence,
            'parsed_from_text': True
        }
    
    def _normalize_score(self, score: Union[int, float, str]) -> float:
        """Normalize score to 0-1 range based on judgment type."""
        if self.criteria.judgment_type == JudgmentType.BINARY:
            if isinstance(score, str):
                return 1.0 if score.lower() in ['yes', 'true', 'correct', 'good'] else 0.0
            return float(score)
        
        elif self.criteria.judgment_type == JudgmentType.SCALE:
            if self.criteria.scale_range:
                min_val, max_val = self.criteria.scale_range
                return (float(score) - min_val) / (max_val - min_val)
            return min(max(float(score) / 5.0, 0.0), 1.0)  # Default 1-5 scale
        
        elif self.criteria.judgment_type == JudgmentType.CATEGORICAL:
            if self.criteria.categories and isinstance(score, str):
                try:
                    index = self.criteria.categories.index(score)
                    return index / (len(self.criteria.categories) - 1)
                except ValueError:
                    return 0.0
        
        return float(score)
    
    def _aggregate_judgments(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple judgments into final result."""
        if len(judgments) == 1:
            return judgments[0]
        
        # Aggregate scores
        scores = [j['score'] for j in judgments]
        final_score = np.mean(scores)
        
        # Aggregate confidence
        confidences = [j.get('confidence', 0.5) for j in judgments]
        final_confidence = np.mean(confidences)
        
        # Combine reasoning
        reasonings = [j.get('reasoning', '') for j in judgments if j.get('reasoning')]
        final_reasoning = f"Aggregated from {len(judgments)} runs: " + "; ".join(reasonings[:3])
        
        return {
            'score': final_score,
            'reasoning': final_reasoning,
            'confidence': final_confidence,
            'score_std': np.std(scores),
            'individual_scores': scores
        }
    
    def _calculate_consistency(self, judgments: List[Dict[str, Any]]) -> float:
        """Calculate consistency score across multiple judgments."""
        scores = [j['score'] for j in judgments]
        if len(scores) < 2:
            return 1.0
        
        # Use coefficient of variation (std/mean) as consistency metric
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 1.0 if std_score == 0 else 0.0
        
        cv = std_score / mean_score
        # Convert to consistency score (1 = perfect consistency, 0 = no consistency)
        return max(0.0, 1.0 - cv)


class LLMFeatureEvaluator(BaseEvaluator):
    """
    Enhanced LLM Feature Evaluator using LLM-as-a-Judge methodology.
    
    Evaluates LLM-based feature extraction with multiple specialized judges:
    - Accuracy Judge: Compares against ground truth
    - Relevance Judge: Assesses relevance to input
    - Completeness Judge: Checks if all required features are extracted
    - Coherence Judge: Evaluates logical consistency
    - Factuality Judge: Verifies factual correctness
    """
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.llm_client = LLMClient()
        self.judges = {}
        self._initialize_judges()
    
    def _initialize_judges(self):
        """Initialize specialized LLM judges."""
        
        # Accuracy Judge (reference-based)
        accuracy_criteria = JudgmentCriteria(
            name="accuracy",
            description="How accurately the extracted features match the ground truth or reference",
            judgment_type=JudgmentType.SCALE,
            scale_range=(1, 5)
        )
        
        accuracy_prompt = JudgmentPrompt(
            system_prompt="""You are an expert evaluator assessing the accuracy of AI-extracted features. 
Your task is to compare extracted features against reference data and provide a detailed assessment.

Guidelines:
- Compare each extracted feature with the corresponding reference
- Consider semantic similarity, not just exact matches
- Account for different valid representations of the same information
- Be objective and consistent in your evaluation""",
            
            evaluation_template="""## Evaluation Task: {criteria_name}

**Criteria**: {criteria_description}

**Input Data**:
{input_text}

**Extracted Features**:
{output_text}

**Reference Features**:
{reference_text}

Please evaluate the accuracy of the extracted features on a scale of 1-5:
- 1: Completely inaccurate, major errors or missing information
- 2: Mostly inaccurate, some correct elements but significant issues
- 3: Partially accurate, mix of correct and incorrect features
- 4: Mostly accurate, minor discrepancies or missing details
- 5: Highly accurate, matches reference with minor acceptable variations

Think step by step and provide your reasoning before giving the final score.

{output_format}""",
            
            examples=[
                {
                    "input": "Financial report mentioning Q3 revenue of $10M and 15% growth",
                    "output": "{'revenue': '$10M', 'period': 'Q3', 'growth_rate': '15%'}",
                    "judgment": "{'score': 5, 'reasoning': 'All key financial metrics correctly extracted with proper formatting', 'confidence': 0.95}"
                }
            ],
            
            output_format="""Respond in JSON format:
{
    "score": <1-5>,
    "reasoning": "<detailed explanation>",
    "confidence": <0-1>,
    "accuracy_breakdown": {
        "correct_features": ["list of correct features"],
        "incorrect_features": ["list of incorrect features"],
        "missing_features": ["list of missing features"]
    }
}"""
        )
        
        self.judges['accuracy'] = LLMJudge(
            self.llm_client, accuracy_criteria, accuracy_prompt, num_runs=2
        )
        
        # Relevance Judge (reference-free)
        relevance_criteria = JudgmentCriteria(
            name="relevance",
            description="How relevant the extracted features are to the input content",
            judgment_type=JudgmentType.SCALE,
            scale_range=(1, 5)
        )
        
        relevance_prompt = JudgmentPrompt(
            system_prompt="""You are an expert evaluator assessing the relevance of AI-extracted features.
Your task is to determine how well the extracted features capture the important information from the input.

Guidelines:
- Focus on whether extracted features are pertinent to the input content
- Consider completeness - are important aspects covered?
- Evaluate if irrelevant information was incorrectly extracted
- Assess the overall utility of the extracted features""",
            
            evaluation_template="""## Evaluation Task: {criteria_name}

**Criteria**: {criteria_description}

**Input Data**:
{input_text}

**Extracted Features**:
{output_text}

Please evaluate the relevance of the extracted features on a scale of 1-5:
- 1: Irrelevant, features don't relate to input content
- 2: Mostly irrelevant, few relevant features extracted
- 3: Partially relevant, some important aspects captured
- 4: Mostly relevant, captures key information with minor gaps
- 5: Highly relevant, comprehensively captures all important aspects

Think step by step and provide your reasoning before giving the final score.

{output_format}""",
            
            examples=[],
            
            output_format="""Respond in JSON format:
{
    "score": <1-5>,
    "reasoning": "<detailed explanation>",
    "confidence": <0-1>,
    "relevance_analysis": {
        "relevant_features": ["list of relevant features"],
        "irrelevant_features": ["list of irrelevant features"],
        "missing_important_aspects": ["list of missing aspects"]
    }
}"""
        )
        
        self.judges['relevance'] = LLMJudge(
            self.llm_client, relevance_criteria, relevance_prompt, num_runs=2
        )
        
        # Factuality Judge
        factuality_criteria = JudgmentCriteria(
            name="factuality",
            description="Whether the extracted features contain factual errors or hallucinations",
            judgment_type=JudgmentType.BINARY
        )
        
        factuality_prompt = JudgmentPrompt(
            system_prompt="""You are an expert fact-checker evaluating AI-extracted features for factual accuracy.
Your task is to identify any factual errors, hallucinations, or unsupported claims in the extracted features.

Guidelines:
- Compare extracted features against the source input
- Flag any information not present or contradicted by the input
- Be strict about factual accuracy - when in doubt, flag as potentially inaccurate
- Consider both explicit and implicit factual claims""",
            
            evaluation_template="""## Evaluation Task: {criteria_name}

**Criteria**: {criteria_description}

**Input Data**:
{input_text}

**Extracted Features**:
{output_text}

Please determine if the extracted features are factually accurate (contain no hallucinations or errors):
- YES: All features are factually accurate and supported by the input
- NO: Contains factual errors, hallucinations, or unsupported claims

Think step by step and provide your reasoning before giving the final judgment.

{output_format}""",
            
            examples=[],
            
            output_format="""Respond in JSON format:
{
    "score": "<YES or NO>",
    "reasoning": "<detailed explanation>",
    "confidence": <0-1>,
    "factuality_issues": ["list of specific factual problems found, if any"]
}"""
        )
        
        self.judges['factuality'] = LLMJudge(
            self.llm_client, factuality_criteria, factuality_prompt, num_runs=3
        )
    
    def evaluate(self, data: Dict[str, Any], **kwargs) -> List[EvaluationResult]:
        """
        Evaluate LLM feature extraction using multiple specialized judges.
        
        Args:
            data: Dictionary containing:
                - component_name: Name of the component
                - input_data: Input data for feature extraction
                - extracted_features: Features extracted by LLM
                - reference_features: Ground truth features (optional)
                - processing_time: Time taken for extraction
                - token_usage: Number of tokens used
                
        Returns:
            List of evaluation results
        """
        results = []
        component_name = data.get("component_name", "unknown_llm_component")
        
        input_data = data.get("input_data", {})
        output_data = {"extracted_features": data.get("extracted_features", {})}
        reference_data = {"reference_features": data.get("reference_features")} if data.get("reference_features") else None
        
        # Run each judge
        for judge_name, judge in self.judges.items():
            try:
                self.logger.info(f"Running {judge_name} judge for {component_name}")
                
                # Skip accuracy judge if no reference data
                if judge_name == "accuracy" and not reference_data:
                    continue
                
                judgment = judge.judge(input_data, output_data, reference_data)
                
                # Convert to evaluation result
                result = self.create_result(
                    component_name,
                    MetricType.ACCURACY if judge_name == "accuracy" else 
                    MetricType.RELEVANCE if judge_name == "relevance" else
                    MetricType.PRECISION,  # factuality
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
                self.logger.error(f"Judge {judge_name} failed: {e}")
                # Create error result
                error_result = self.create_result(
                    component_name,
                    MetricType.ACCURACY,
                    0.0,
                    {
                        "evaluation_type": f"llm_judge_{judge_name}_error",
                        "error": str(e)
                    }
                )
                results.append(error_result)
        
        # Add traditional metrics for comparison
        if "processing_time" in data:
            results.append(self.create_result(
                component_name, MetricType.LATENCY, data["processing_time"],
                {"unit": "seconds", "evaluation_type": "processing_latency"}
            ))
        
        if "token_usage" in data and input_data:
            efficiency = len(data.get("extracted_features", {})) / max(data["token_usage"], 1)
            results.append(self.create_result(
                component_name, MetricType.THROUGHPUT, efficiency,
                {"unit": "features_per_token", "evaluation_type": "token_efficiency"}
            ))
        
        return results 