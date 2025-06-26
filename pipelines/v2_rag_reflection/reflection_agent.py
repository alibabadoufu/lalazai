"""
Reflection Agent for RAG Pipeline Enhancement

This agent provides self-reflection and iterative refinement capabilities
to improve the quality of recommendations.
"""

import logging
from typing import Dict, Any, List, Optional
import time


class ReflectionAgent:
    """Agent for reflecting on and refining RAG results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reflection agent.
        
        Args:
            config: Configuration dictionary for the reflection agent
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Reflection parameters
        self.max_iterations = config.get('max_iterations', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.reflection_model = config.get('reflection_model', 'gpt-4')
        
        # Track iteration count for metadata
        self.last_iteration_count = 0
        
        # Load reflection prompts
        self.reflection_prompts = self._load_reflection_prompts()
        
        self.logger.info("Reflection agent initialized")
    
    def reflect_and_refine(
        self, 
        initial_results: List[Dict[str, Any]], 
        query: str, 
        context: Dict[str, Any],
        max_iterations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reflect on initial results and iteratively refine them.
        
        Args:
            initial_results: Initial recommendations from the pipeline
            query: Original user query
            context: Additional context information
            max_iterations: Override default max iterations
            
        Returns:
            Refined list of recommendations
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        current_results = initial_results.copy()
        self.last_iteration_count = 0
        
        self.logger.info(f"Starting reflection process with {len(current_results)} initial results")
        
        for iteration in range(max_iterations):
            self.last_iteration_count = iteration + 1
            
            # Step 1: Analyze current results
            analysis = self._analyze_results(current_results, query, context)
            
            # Step 2: Check if refinement is needed
            if self._should_stop_refinement(analysis, current_results):
                self.logger.info(f"Refinement complete after {iteration + 1} iterations")
                break
            
            # Step 3: Generate refinement suggestions
            refinement_suggestions = self._generate_refinement_suggestions(
                analysis, current_results, query, context
            )
            
            # Step 4: Apply refinements
            current_results = self._apply_refinements(
                current_results, refinement_suggestions, query, context
            )
            
            self.logger.info(f"Completed iteration {iteration + 1}, refined results count: {len(current_results)}")
        
        # Final quality enhancement
        final_results = self._final_quality_enhancement(current_results, query, context)
        
        self.logger.info(f"Reflection process completed with {len(final_results)} final results")
        return final_results
    
    def _analyze_results(self, results: List[Dict[str, Any]], query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current results to identify improvement opportunities.
        
        Args:
            results: Current recommendation results
            query: User query
            context: Additional context
            
        Returns:
            Analysis results with improvement suggestions
        """
        analysis = {
            'overall_quality': self._assess_overall_quality(results),
            'coverage_gaps': self._identify_coverage_gaps(results, query, context),
            'redundancy_issues': self._identify_redundancy(results),
            'relevance_issues': self._identify_relevance_issues(results, query, context),
            'confidence_distribution': self._analyze_confidence_distribution(results)
        }
        
        self.logger.debug(f"Analysis completed: {analysis}")
        return analysis
    
    def _should_stop_refinement(self, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> bool:
        """
        Determine if refinement should stop based on analysis.
        
        Args:
            analysis: Analysis results
            results: Current results
            
        Returns:
            True if refinement should stop
        """
        # Stop if overall quality is high enough
        if analysis['overall_quality'] >= self.confidence_threshold:
            return True
        
        # Stop if no significant improvement opportunities
        if (
            len(analysis['coverage_gaps']) == 0 and
            len(analysis['redundancy_issues']) == 0 and
            len(analysis['relevance_issues']) == 0
        ):
            return True
        
        # Stop if confidence is uniformly high
        confidence_dist = analysis['confidence_distribution']
        if confidence_dist['min'] >= self.confidence_threshold * 0.9:
            return True
        
        return False
    
    def _generate_refinement_suggestions(
        self, 
        analysis: Dict[str, Any], 
        results: List[Dict[str, Any]], 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate specific refinement suggestions based on analysis.
        
        Args:
            analysis: Results analysis
            results: Current results
            query: User query
            context: Additional context
            
        Returns:
            Refinement suggestions
        """
        suggestions = {
            'remove_low_confidence': [],
            'remove_redundant': [],
            'boost_relevant': [],
            'add_missing_coverage': [],
            'rerank_suggestions': []
        }
        
        # Identify low confidence items for removal
        for i, result in enumerate(results):
            confidence = result.get('confidence', 0.0)
            if confidence < self.confidence_threshold * 0.5:
                suggestions['remove_low_confidence'].append(i)
        
        # Identify redundant items
        for issue in analysis['redundancy_issues']:
            suggestions['remove_redundant'].extend(issue['duplicate_indices'])
        
        # Identify highly relevant items to boost
        for i, result in enumerate(results):
            if self._is_highly_relevant(result, query, context):
                suggestions['boost_relevant'].append(i)
        
        # Suggest coverage improvements
        for gap in analysis['coverage_gaps']:
            suggestions['add_missing_coverage'].append(gap)
        
        return suggestions
    
    def _apply_refinements(
        self, 
        results: List[Dict[str, Any]], 
        suggestions: Dict[str, Any], 
        query: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply refinement suggestions to the results.
        
        Args:
            results: Current results
            suggestions: Refinement suggestions
            query: User query
            context: Additional context
            
        Returns:
            Refined results
        """
        refined_results = results.copy()
        
        # Remove low confidence items
        indices_to_remove = set(suggestions['remove_low_confidence'] + suggestions['remove_redundant'])
        refined_results = [r for i, r in enumerate(refined_results) if i not in indices_to_remove]
        
        # Boost confidence of highly relevant items
        for i in suggestions['boost_relevant']:
            if i < len(refined_results):
                current_confidence = refined_results[i].get('confidence', 0.0)
                refined_results[i]['confidence'] = min(1.0, current_confidence * 1.2)
                refined_results[i]['reflection_boost'] = True
        
        # Re-rank based on updated confidences
        refined_results.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Add reflection metadata
        for result in refined_results:
            result['reflection_applied'] = True
            result['reflection_iteration'] = self.last_iteration_count
        
        return refined_results
    
    def _final_quality_enhancement(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply final quality enhancements to the results.
        
        Args:
            results: Results after reflection iterations
            query: User query
            context: Additional context
            
        Returns:
            Final enhanced results
        """
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add final quality score
            enhanced_result['final_quality_score'] = self._calculate_final_quality_score(
                result, query, context
            )
            
            # Add explanation for why this was recommended
            enhanced_result['recommendation_reasoning'] = self._generate_reasoning(
                result, query, context
            )
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _assess_overall_quality(self, results: List[Dict[str, Any]]) -> float:
        """Assess the overall quality of current results."""
        if not results:
            return 0.0
        
        confidences = [r.get('confidence', 0.0) for r in results]
        return sum(confidences) / len(confidences)
    
    def _identify_coverage_gaps(self, results: List[Dict[str, Any]], query: str, context: Dict[str, Any]) -> List[str]:
        """Identify potential coverage gaps in the results."""
        # Simplified implementation - in practice, this would use more sophisticated analysis
        covered_topics = set()
        for result in results:
            topic = result.get('metadata', {}).get('topic', 'unknown')
            covered_topics.add(topic)
        
        # This is a placeholder - real implementation would analyze query context
        # to identify missing important topics
        gaps = []
        if len(covered_topics) < 3:  # Simple heuristic
            gaps.append("Insufficient topic diversity")
        
        return gaps
    
    def _identify_redundancy(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify redundant items in the results."""
        redundancy_issues = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                similarity = self._calculate_similarity(results[i], results[j])
                if similarity > 0.8:  # High similarity threshold
                    redundancy_issues.append({
                        'type': 'high_similarity',
                        'items': [i, j],
                        'similarity': similarity,
                        'duplicate_indices': [j]  # Remove the later one
                    })
        
        return redundancy_issues
    
    def _identify_relevance_issues(self, results: List[Dict[str, Any]], query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify relevance issues in the results."""
        issues = []
        
        for i, result in enumerate(results):
            relevance_score = self._calculate_relevance_score(result, query, context)
            if relevance_score < 0.6:  # Low relevance threshold
                issues.append({
                    'index': i,
                    'issue': 'low_relevance',
                    'score': relevance_score
                })
        
        return issues
    
    def _analyze_confidence_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze the confidence score distribution."""
        if not results:
            return {'min': 0.0, 'max': 0.0, 'avg': 0.0, 'std': 0.0}
        
        confidences = [r.get('confidence', 0.0) for r in results]
        
        return {
            'min': min(confidences),
            'max': max(confidences),
            'avg': sum(confidences) / len(confidences),
            'std': self._calculate_std(confidences)
        }
    
    def _calculate_similarity(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> float:
        """Calculate similarity between two results."""
        # Simplified similarity calculation
        # In practice, this would use more sophisticated methods like embedding similarity
        
        title1 = result1.get('title', '').lower()
        title2 = result2.get('title', '').lower()
        
        # Simple word overlap similarity
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str, context: Dict[str, Any]) -> float:
        """Calculate relevance score for a result."""
        # Simplified relevance calculation
        # In practice, this would use LLM-based scoring
        
        base_confidence = result.get('confidence', 0.0)
        
        # Simple keyword matching boost
        query_words = set(query.lower().split())
        title_words = set(result.get('title', '').lower().split())
        
        keyword_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0.0
        
        return min(1.0, base_confidence + keyword_overlap * 0.2)
    
    def _is_highly_relevant(self, result: Dict[str, Any], query: str, context: Dict[str, Any]) -> bool:
        """Check if a result is highly relevant."""
        relevance_score = self._calculate_relevance_score(result, query, context)
        return relevance_score >= 0.8
    
    def _calculate_final_quality_score(self, result: Dict[str, Any], query: str, context: Dict[str, Any]) -> float:
        """Calculate a final quality score for a result."""
        confidence = result.get('confidence', 0.0)
        relevance = self._calculate_relevance_score(result, query, context)
        
        # Weighted combination
        return 0.7 * confidence + 0.3 * relevance
    
    def _generate_reasoning(self, result: Dict[str, Any], query: str, context: Dict[str, Any]) -> str:
        """Generate explanation for why this result was recommended."""
        reasoning_parts = []
        
        confidence = result.get('confidence', 0.0)
        if confidence > 0.8:
            reasoning_parts.append("High confidence match")
        
        if result.get('reflection_boost', False):
            reasoning_parts.append("Enhanced through reflection process")
        
        relevance = self._calculate_relevance_score(result, query, context)
        if relevance > 0.7:
            reasoning_parts.append("Strong relevance to query")
        
        if not reasoning_parts:
            reasoning_parts.append("Standard recommendation criteria")
        
        return "; ".join(reasoning_parts)
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _load_reflection_prompts(self) -> Dict[str, str]:
        """Load reflection prompt templates."""
        # In practice, these would be loaded from files
        return {
            'analysis_prompt': """
            Analyze the following recommendation results for quality and relevance:
            Query: {query}
            Results: {results}
            
            Identify potential issues with:
            1. Relevance to the query
            2. Result quality and confidence
            3. Diversity and coverage
            4. Redundancy or duplication
            """,
            'refinement_prompt': """
            Based on the analysis, suggest improvements for these recommendations:
            Current Results: {results}
            Issues Identified: {issues}
            
            Provide specific suggestions for:
            1. Which results to remove or demote
            2. Which results to boost or promote  
            3. What types of content are missing
            """
        } 