"""
Enhanced Example Usage of the Modular Evaluation System.

This example demonstrates how to use the new modular evaluation system with
LLM-as-a-judge capabilities for comprehensive AI component assessment.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

from evaluation import (
    EvaluatorFactory, 
    LLMFeatureEvaluator,
    RecommendationEvaluator,
    ConnectorEvaluator,
    ProfileEvaluator
)
from evaluation.base_evaluator import EvaluationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedEvaluationDemo:
    """Demonstration of the enhanced modular evaluation system."""
    
    def __init__(self):
        self.factory = EvaluatorFactory()
        self.evaluators = {}
    
    def setup_evaluators(self):
        """Set up evaluators using the factory pattern."""
        logger.info("Setting up evaluators using factory pattern...")
        
        # Method 1: Create evaluators using factory with custom configs
        custom_configs = [
            {
                'name': 'llm_feature_evaluator',
                'type': 'llm_feature',
                'enabled': True,
                'evaluation_interval_hours': 12,
                'custom_parameters': {
                    'enable_llm_judges': True,
                    'num_judge_runs': 3,  # More runs for higher confidence
                    'llm_model': 'gpt-4'
                }
            },
            {
                'name': 'recommendation_evaluator', 
                'type': 'recommendation',
                'enabled': True,
                'evaluation_interval_hours': 6,
                'custom_parameters': {
                    'enable_llm_judges': True
                }
            },
            {
                'name': 'connector_evaluator',
                'type': 'connector',
                'enabled': True,
                'thresholds': {
                    'completeness': 0.98,  # High standards
                    'validity': 0.95,
                    'consistency': 0.90,
                    'timeliness': 0.85
                }
            }
        ]
        
        self.evaluators = self.factory.create_evaluators_from_config(custom_configs)
        
        # Method 2: Create individual evaluators
        profile_config = EvaluationConfig(
            evaluator_name='profile_evaluator',
            enabled=True,
            evaluation_interval_hours=24
        )
        
        profile_evaluator = self.factory.create_evaluator('profile', profile_config)
        self.evaluators['profile_evaluator'] = profile_evaluator
        
        logger.info(f"Created {len(self.evaluators)} evaluators")
    
    def demonstrate_llm_feature_evaluation(self):
        """Demonstrate LLM feature evaluation with LLM-as-a-judge."""
        logger.info("\n=== LLM Feature Evaluation with LLM-as-a-Judge ===")
        
        evaluator = self.evaluators.get('llm_feature_evaluator')
        if not evaluator:
            logger.error("LLM Feature evaluator not found")
            return
        
        # Sample data for LLM feature extraction evaluation
        evaluation_data = {
            'component_name': 'financial_feature_extractor',
            'input_data': {
                'content': 'BNP Paribas Q3 2024 earnings report shows revenue of €12.5B, up 8% YoY. Net income increased to €2.8B. The bank expanded operations in Asia-Pacific region with focus on sustainable finance.',
                'document_type': 'earnings_report',
                'client_context': 'institutional_investor'
            },
            'extracted_features': {
                'revenue': '€12.5B',
                'revenue_growth': '8% YoY',
                'net_income': '€2.8B',
                'period': 'Q3 2024',
                'geographic_expansion': 'Asia-Pacific',
                'strategic_focus': 'sustainable finance',
                'company': 'BNP Paribas'
            },
            'reference_features': {  # Ground truth for accuracy judge
                'revenue': '€12.5B',
                'revenue_growth': '8% YoY', 
                'net_income': '€2.8B',
                'period': 'Q3 2024',
                'geographic_expansion': 'Asia-Pacific',
                'strategic_focus': 'sustainable finance',
                'company': 'BNP Paribas'
            },
            'processing_time': 2.3,
            'token_usage': 450
        }
        
        # Run evaluation
        results = evaluator.run_evaluation(evaluation_data)
        
        # Display results
        self._display_results("LLM Feature Evaluation", results)
    
    def demonstrate_recommendation_evaluation(self):
        """Demonstrate recommendation evaluation with hybrid metrics."""
        logger.info("\n=== Recommendation Evaluation with LLM Judges ===")
        
        evaluator = self.evaluators.get('recommendation_evaluator')
        if not evaluator:
            logger.error("Recommendation evaluator not found")
            return
        
        # Sample data for recommendation evaluation
        evaluation_data = {
            'client_id': 'institutional_client_001',
            'client_profile': {
                'client_name': 'Global Pension Fund',
                'sector': 'Financial Services',
                'aum': '€50B',
                'risk_tolerance': 'Conservative',
                'investment_objectives': ['capital_preservation', 'steady_income'],
                'sectors': ['technology', 'healthcare', 'renewable_energy'],
                'investment_themes': ['ESG', 'digital_transformation'],
                'geographic_preferences': ['Europe', 'North America']
            },
            'recommendations': [
                {
                    'title': 'ESG Investment Opportunities in European Tech',
                    'author': 'Sustainable Finance Team',
                    'publication_type': 'research_report',
                    'sectors': ['technology'],
                    'themes': ['ESG', 'digital_transformation'], 
                    'regions': ['Europe'],
                    'relevance_score': 0.92
                },
                {
                    'title': 'Healthcare Innovation Trends 2024',
                    'author': 'Healthcare Research',
                    'publication_type': 'market_outlook',
                    'sectors': ['healthcare'],
                    'themes': ['digital_transformation'],
                    'regions': ['North America'],
                    'relevance_score': 0.87
                },
                {
                    'title': 'Renewable Energy Infrastructure Bonds',
                    'author': 'Fixed Income Research',
                    'publication_type': 'investment_strategy',
                    'sectors': ['renewable_energy'],
                    'themes': ['ESG'],
                    'regions': ['Europe'],
                    'relevance_score': 0.85
                }
            ],
            'ranking_scores': [0.92, 0.87, 0.85]
        }
        
        # Run evaluation
        results = evaluator.run_evaluation(evaluation_data)
        
        # Display results
        self._display_results("Recommendation Evaluation", results)
        
        # Get recommendation summary
        if hasattr(evaluator, 'get_recommendation_summary'):
            summary = evaluator.get_recommendation_summary(evaluation_data)
            logger.info(f"Recommendation Summary: {json.dumps(summary, indent=2)}")
    
    def demonstrate_connector_evaluation(self):
        """Demonstrate data connector evaluation."""
        logger.info("\n=== Data Connector Evaluation ===")
        
        evaluator = self.evaluators.get('connector_evaluator')
        if not evaluator:
            logger.error("Connector evaluator not found")
            return
        
        # Sample data for connector evaluation
        evaluation_data = {
            'connector_name': 'bloomberg_chat_connector',
            'ingested_records': 1250,
            'processing_time': 45.2,
            'total_attempts': 1300,
            'successful_ingestions': 1250,
            'total_operations': 1300,
            'errors': ['timeout_error', 'rate_limit_exceeded'],
            'data_quality_metrics': {
                'completeness': 0.96,
                'validity': 0.94,
                'consistency': 0.91,
                'timeliness': 0.88
            }
        }
        
        # Run evaluation
        results = evaluator.run_evaluation(evaluation_data)
        
        # Display results
        self._display_results("Connector Evaluation", results)
        
        # Get quality summary
        if hasattr(evaluator, 'get_quality_summary'):
            summary = evaluator.get_quality_summary(evaluation_data)
            logger.info(f"Quality Summary: {json.dumps(summary, indent=2)}")
    
    def demonstrate_profile_evaluation(self):
        """Demonstrate client profile evaluation."""
        logger.info("\n=== Client Profile Evaluation ===")
        
        evaluator = self.evaluators.get('profile_evaluator')
        if not evaluator:
            logger.error("Profile evaluator not found")
            return
        
        # Sample data for profile evaluation
        evaluation_data = {
            'client_id': 'client_001',
            'profile': {
                'client_name': 'ABC Investment Management',
                'sector': 'Asset Management',
                'risk_tolerance': 'Moderate',
                'investment_objectives': ['growth', 'diversification'],
                'geographic_preferences': ['Global'],
                'asset_class_preferences': ['equities', 'fixed_income'],
                'aum': '€15B',
                'investment_themes': ['sustainable_investing'],
                'esg_preferences': 'high_importance'
            },
            'data_sources': ['CRM', 'Bloomberg', 'Internal_Research'],
            'last_updated': '2024-03-15T10:30:00Z',
            'confidence_scores': {
                'sector': 0.95,
                'risk_tolerance': 0.88,
                'aum': 0.92,
                'investment_objectives': 0.85
            },
            'validation_results': {
                'total_validations': 10,
                'passed_validations': 9
            }
        }
        
        # Run evaluation
        results = evaluator.run_evaluation(evaluation_data)
        
        # Display results
        self._display_results("Profile Evaluation", results)
        
        # Get profile summary and recommendations
        if hasattr(evaluator, 'get_profile_summary'):
            summary = evaluator.get_profile_summary(evaluation_data)
            logger.info(f"Profile Summary: {json.dumps(summary, indent=2)}")
        
        if hasattr(evaluator, 'get_improvement_recommendations'):
            recommendations = evaluator.get_improvement_recommendations(evaluation_data)
            logger.info(f"Improvement Recommendations: {json.dumps(recommendations, indent=2)}")
    
    def demonstrate_factory_features(self):
        """Demonstrate factory pattern features."""
        logger.info("\n=== Factory Pattern Features ===")
        
        # Get evaluator information
        info = self.factory.get_evaluator_info()
        logger.info(f"Evaluator Info: {json.dumps(info, indent=2)}")
        
        # Validate configuration
        test_config = {
            'name': 'test_evaluator',
            'type': 'llm_feature',
            'enabled': True
        }
        
        validation_result = self.factory.validate_configuration(test_config)
        logger.info(f"Config Validation: {json.dumps(validation_result, indent=2)}")
        
        # Create default evaluators
        logger.info("Creating default evaluators...")
        default_evaluators = self.factory.create_default_evaluators()
        logger.info(f"Created {len(default_evaluators)} default evaluators")
    
    def _display_results(self, title: str, results):
        """Display evaluation results in a readable format."""
        logger.info(f"\n{title} Results:")
        logger.info("-" * 50)
        
        for result in results:
            logger.info(f"Component: {result.component_name}")
            logger.info(f"Metric: {result.metric_type.value}")
            logger.info(f"Value: {result.value}")
            logger.info(f"Status: {result.status.value}")
            
            # Display metadata
            if result.metadata:
                interesting_metadata = {
                    k: v for k, v in result.metadata.items() 
                    if k in ['evaluation_type', 'reasoning', 'confidence', 'consistency_score']
                }
                if interesting_metadata:
                    logger.info(f"Metadata: {json.dumps(interesting_metadata, indent=2)}")
            
            logger.info("-" * 30)
    
    def run_complete_demo(self):
        """Run the complete evaluation demonstration."""
        logger.info("🚀 Starting Enhanced Evaluation System Demo")
        logger.info("=" * 60)
        
        try:
            self.setup_evaluators()
            self.demonstrate_factory_features()
            self.demonstrate_llm_feature_evaluation()
            self.demonstrate_recommendation_evaluation()
            self.demonstrate_connector_evaluation()
            self.demonstrate_profile_evaluation()
            
            logger.info("\n✅ Demo completed successfully!")
            logger.info("The modular evaluation system with LLM-as-a-judge is ready for use.")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise


def main():
    """Main entry point for the demo."""
    demo = EnhancedEvaluationDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main() 