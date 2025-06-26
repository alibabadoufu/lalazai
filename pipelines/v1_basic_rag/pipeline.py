"""
Basic RAG Pipeline - Version 1

Simple retrieval-augmented generation approach for recommendation system.
This version provides the foundational RAG functionality.
"""

import time
from typing import Dict, Any, List
from pipelines.base.pipeline import BasePipeline, PipelineResult


class BasicRAGPipeline(BasePipeline):
    """Version 1: Basic RAG implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Basic RAG Pipeline."""
        super().__init__(config)
        self.version = "v1_basic_rag"
        
        # Initialize components based on config
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize pipeline components."""
        # Import components (will be imported from base or utils)
        from pipelines.base.retriever import Retriever
        from pipelines.base.ranker import Ranker
        
        # Initialize with config
        retriever_config = self.config.get('retriever', {})
        ranker_config = self.config.get('ranker', {})
        
        self.retriever = Retriever(retriever_config)
        self.ranker = Ranker(ranker_config)
        
        self.logger.info(f"Initialized {self.version} pipeline components")
    
    def run(self, query: str, context: Dict[str, Any]) -> PipelineResult:
        """
        Run the basic RAG pipeline.
        
        Args:
            query: User query or client description
            context: Additional context (client profile, preferences, etc.)
            
        Returns:
            PipelineResult with recommendations
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess input
            processed_input = self.preprocess_query(query, context)
            self.logger.info(f"Preprocessed query for {self.version}")
            
            # Step 2: Retrieve candidate documents/publications
            candidates = self.retriever.retrieve(
                query=processed_input['query'],
                context=processed_input['context'],
                top_k=self.config.get('retriever', {}).get('top_k', 20)
            )
            self.logger.info(f"Retrieved {len(candidates)} candidates")
            
            # Step 3: Rank and score candidates
            ranked_results = self.ranker.rank(
                candidates=candidates,
                query=processed_input['query'],
                context=processed_input['context'],
                max_results=self.config.get('ranker', {}).get('max_results', 5)
            )
            self.logger.info(f"Ranked to {len(ranked_results)} final recommendations")
            
            # Step 4: Post-process results
            final_results = self.postprocess_results(ranked_results)
            
            execution_time = time.time() - start_time
            
            return PipelineResult(
                recommendations=final_results,
                metadata={
                    'pipeline_version': self.version,
                    'num_candidates': len(candidates),
                    'num_final': len(final_results),
                    'retrieval_strategy': 'basic_vector_search',
                    'ranking_strategy': 'llm_scoring'
                },
                version=self.version,
                execution_time=execution_time,
                confidence_scores=[r.get('confidence', 0.0) for r in final_results]
            )
            
        except Exception as e:
            self.logger.error(f"Error in {self.version} pipeline: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about this pipeline version."""
        return {
            'name': 'Basic RAG Pipeline',
            'version': self.version,
            'description': 'Simple retrieval-augmented generation for recommendations',
            'capabilities': [
                'Vector-based document retrieval',
                'LLM-based relevance scoring',
                'Basic ranking and filtering'
            ],
            'limitations': [
                'No reflection or self-correction',
                'Single-stage retrieval only',
                'Limited context understanding'
            ],
            'use_cases': [
                'Simple recommendation tasks',
                'Baseline comparisons',
                'Quick prototyping'
            ],
            'config_schema': {
                'retriever': {
                    'top_k': 'Number of candidates to retrieve (default: 20)',
                    'similarity_threshold': 'Minimum similarity score (default: 0.7)'
                },
                'ranker': {
                    'max_results': 'Maximum final recommendations (default: 5)',
                    'scoring_model': 'LLM model for scoring (default: gpt-4)'
                }
            }
        }
    
    def batch_process(self, queries: List[Dict[str, Any]]) -> List[PipelineResult]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query dictionaries with 'query' and 'context' keys
            
        Returns:
            List of PipelineResults
        """
        results = []
        
        for i, query_data in enumerate(queries):
            self.logger.info(f"Processing batch item {i+1}/{len(queries)}")
            
            result = self.run(
                query=query_data['query'],
                context=query_data.get('context', {})
            )
            results.append(result)
            
        return results 