"""
RAG with Reflection Pipeline - Version 2

Enhanced RAG with self-reflection and refinement capabilities.
This version adds reflection mechanisms to improve recommendation quality.
"""

import time
from typing import Dict, Any, List
from pipelines.base.pipeline import BasePipeline, PipelineResult
from pipelines.v2_rag_reflection.reflection_agent import ReflectionAgent


class RAGReflectionPipeline(BasePipeline):
    """Version 2: RAG with reflection mechanism."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG with Reflection Pipeline."""
        super().__init__(config)
        self.version = "v2_rag_reflection"
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize pipeline components."""
        from pipelines.base.retriever import Retriever
        from pipelines.base.ranker import Ranker
        
        # Initialize base components
        retriever_config = self.config.get('retriever', {})
        ranker_config = self.config.get('ranker', {})
        reflection_config = self.config.get('reflection', {})
        
        self.retriever = Retriever(retriever_config)
        self.ranker = Ranker(ranker_config)
        self.reflection_agent = ReflectionAgent(reflection_config)
        
        self.logger.info(f"Initialized {self.version} pipeline components")
    
    def run(self, query: str, context: Dict[str, Any]) -> PipelineResult:
        """
        Run the RAG pipeline with reflection.
        
        Args:
            query: User query or client description
            context: Additional context information
            
        Returns:
            PipelineResult with refined recommendations
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess input
            processed_input = self.preprocess_query(query, context)
            self.logger.info(f"Preprocessed query for {self.version}")
            
            # Step 2: Initial retrieval (broader search)
            initial_candidates = self.retriever.retrieve(
                query=processed_input['query'],
                context=processed_input['context'],
                top_k=self.config.get('retriever', {}).get('top_k', 30)  # More candidates for reflection
            )
            self.logger.info(f"Retrieved {len(initial_candidates)} initial candidates")
            
            # Step 3: Initial ranking
            initial_results = self.ranker.rank(
                candidates=initial_candidates,
                query=processed_input['query'],
                context=processed_input['context'],
                max_results=self.config.get('ranker', {}).get('initial_results', 10)
            )
            self.logger.info(f"Initial ranking produced {len(initial_results)} results")
            
            # Step 4: Reflection and refinement
            refined_results = self.reflection_agent.reflect_and_refine(
                initial_results=initial_results,
                query=processed_input['query'],
                context=processed_input['context'],
                max_iterations=self.config.get('reflection', {}).get('max_iterations', 3)
            )
            self.logger.info(f"Reflection process completed with {len(refined_results)} refined results")
            
            # Step 5: Final selection
            final_results = self._select_final_results(
                refined_results,
                max_results=self.config.get('ranker', {}).get('max_results', 5)
            )
            
            # Step 6: Post-process results
            final_results = self.postprocess_results(final_results)
            
            execution_time = time.time() - start_time
            
            return PipelineResult(
                recommendations=final_results,
                metadata={
                    'pipeline_version': self.version,
                    'num_initial_candidates': len(initial_candidates),
                    'num_initial_results': len(initial_results),
                    'num_final': len(final_results),
                    'retrieval_strategy': 'enhanced_vector_search',
                    'ranking_strategy': 'llm_scoring_with_reflection',
                    'reflection_iterations': self.reflection_agent.last_iteration_count,
                    'confidence_improvement': self._calculate_confidence_improvement(initial_results, final_results)
                },
                version=self.version,
                execution_time=execution_time,
                confidence_scores=[r.get('confidence', 0.0) for r in final_results]
            )
            
        except Exception as e:
            self.logger.error(f"Error in {self.version} pipeline: {e}")
            raise
    
    def _select_final_results(self, refined_results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """Select the final results based on refined scores."""
        # Sort by confidence score (descending)
        sorted_results = sorted(
            refined_results,
            key=lambda x: x.get('confidence', 0.0),
            reverse=True
        )
        
        # Apply diversity filtering if configured
        if self.config.get('diversity_filtering', False):
            sorted_results = self._apply_diversity_filtering(sorted_results, max_results)
        
        return sorted_results[:max_results]
    
    def _apply_diversity_filtering(self, results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """Apply diversity filtering to avoid redundant recommendations."""
        # Simple diversity filtering based on topic/sector
        diverse_results = []
        used_sectors = set()
        
        for result in results:
            sector = result.get('metadata', {}).get('sector', 'unknown')
            
            # Allow some duplication but prefer diversity
            if sector not in used_sectors or len(diverse_results) < max_results // 2:
                diverse_results.append(result)
                used_sectors.add(sector)
                
                if len(diverse_results) >= max_results:
                    break
        
        # Fill remaining slots if needed
        for result in results:
            if result not in diverse_results and len(diverse_results) < max_results:
                diverse_results.append(result)
        
        return diverse_results
    
    def _calculate_confidence_improvement(self, initial_results: List[Dict[str, Any]], final_results: List[Dict[str, Any]]) -> float:
        """Calculate the average confidence improvement from reflection."""
        if not initial_results or not final_results:
            return 0.0
        
        initial_avg = sum(r.get('confidence', 0.0) for r in initial_results) / len(initial_results)
        final_avg = sum(r.get('confidence', 0.0) for r in final_results) / len(final_results)
        
        return final_avg - initial_avg
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about this pipeline version."""
        return {
            'name': 'RAG with Reflection Pipeline',
            'version': self.version,
            'description': 'Enhanced RAG with self-reflection and iterative refinement',
            'capabilities': [
                'Vector-based document retrieval',
                'LLM-based relevance scoring',
                'Self-reflection and critique',
                'Iterative result refinement',
                'Confidence improvement tracking',
                'Optional diversity filtering'
            ],
            'advantages': [
                'Higher quality recommendations',
                'Self-correcting mechanism',
                'Better handling of ambiguous queries',
                'Confidence-aware ranking'
            ],
            'limitations': [
                'Increased computational cost',
                'Longer execution time',
                'Requires reflection-capable LLM'
            ],
            'use_cases': [
                'High-stakes recommendations',
                'Complex client requirements',
                'Quality-over-speed scenarios'
            ],
            'config_schema': {
                'retriever': {
                    'top_k': 'Number of candidates to retrieve (default: 30)',
                    'similarity_threshold': 'Minimum similarity score (default: 0.6)'
                },
                'ranker': {
                    'initial_results': 'Initial results before reflection (default: 10)',
                    'max_results': 'Maximum final recommendations (default: 5)'
                },
                'reflection': {
                    'max_iterations': 'Maximum reflection iterations (default: 3)',
                    'confidence_threshold': 'Confidence threshold for iteration (default: 0.8)',
                    'reflection_model': 'LLM model for reflection (default: gpt-4)'
                },
                'diversity_filtering': 'Enable diversity filtering (default: false)'
            }
        } 