#!/usr/bin/env python3
"""
Main entry point for the Reorganized Intelligent Research Recommendation System.

This demonstrates the new pipeline system with version management and 
better separation of concerns.
"""

import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/recommendation_system.log')
    ]
)

from pipelines.factory import PipelineFactory


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def demo_pipeline_factory():
    """Demonstrate the new pipeline factory system."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== Pipeline Factory Demonstration ===")
    
    # 1. List available pipeline versions
    logger.info("Available pipeline versions:")
    stable_versions = PipelineFactory.list_available_versions()
    experimental_versions = PipelineFactory.list_available_versions(include_experimental=True)
    
    for version in stable_versions:
        logger.info(f"  ✓ {version} (stable)")
    
    for version in experimental_versions:
        if version not in stable_versions:
            logger.info(f"  ⚠ {version} (experimental)")
    
    # 2. Get pipeline information
    logger.info("\n=== Pipeline Information ===")
    for version in stable_versions:
        try:
            info = PipelineFactory.get_pipeline_info(version)
            logger.info(f"{version}:")
            logger.info(f"  Name: {info['name']}")
            logger.info(f"  Description: {info['description']}")
            logger.info(f"  Status: {info['status']}")
            
            features = info.get('features', {})
            enabled_features = [f for f, enabled in features.items() if enabled]
            logger.info(f"  Features: {', '.join(enabled_features)}")
        except Exception as e:
            logger.error(f"Error getting info for {version}: {e}")
    
    # 3. Demonstrate automatic pipeline selection
    logger.info("\n=== Automatic Pipeline Selection ===")
    
    requirements = {
        'features': ['reflection'],
        'performance_priority': 'quality',
        'stability': 'stable'
    }
    
    try:
        recommended_version = PipelineFactory.get_recommended_version(requirements)
        logger.info(f"Recommended pipeline for requirements {requirements}: {recommended_version}")
    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")
    
    # 4. Compare pipeline versions
    logger.info("\n=== Pipeline Comparison ===")
    if len(stable_versions) >= 2:
        try:
            comparison = PipelineFactory.compare_versions(stable_versions[0], stable_versions[1])
            logger.info(f"Comparing {comparison['version1']} vs {comparison['version2']}:")
            logger.info(f"  Added features: {comparison['added_features']}")
            logger.info(f"  Removed features: {comparison['removed_features']}")
            logger.info(f"  Compatibility: {comparison['compatibility']}")
            logger.info(f"  Recommendation: {comparison['recommendation']}")
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")


def demo_pipeline_execution():
    """Demonstrate pipeline execution with different versions."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n=== Pipeline Execution Demonstration ===")
    
    # Sample query and context
    sample_query = "I need research on AI applications in financial risk management"
    sample_context = {
        "client_id": "demo_client",
        "sector": "financial_services",
        "risk_tolerance": "moderate",
        "previous_interests": ["artificial_intelligence", "risk_management"]
    }
    
    # Try different pipeline versions
    versions_to_test = ["v1_basic_rag", "v2_rag_reflection"]
    
    for version in versions_to_test:
        logger.info(f"\n--- Testing {version} ---")
        
        try:
            # Load configuration for this version
            config_path = f"configs/pipelines/{version}.yaml"
            if Path(config_path).exists():
                config = load_config(config_path)
                config['environment'] = 'demo'
                
                # Create pipeline instance
                pipeline = PipelineFactory.create_pipeline(version, config)
                
                # Get pipeline info
                info = pipeline.get_pipeline_info()
                logger.info(f"Pipeline: {info['name']}")
                logger.info(f"Description: {info['description']}")
                
                # Note: In a real implementation, we would call pipeline.run()
                # For this demo, we'll just show the setup is working
                logger.info(f"✓ Pipeline {version} created successfully")
                
                # Demonstrate health check
                health = pipeline.health_check()
                logger.info(f"Health check: {health['status']}")
                
            else:
                logger.warning(f"Configuration file not found: {config_path}")
                
        except Exception as e:
            logger.error(f"Error testing {version}: {e}")


def demo_auto_pipeline():
    """Demonstrate automatic pipeline selection and creation."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n=== Auto Pipeline Demonstration ===")
    
    # Define different scenarios
    scenarios = [
        {
            "name": "Speed Priority",
            "requirements": {
                'features': [],
                'performance_priority': 'speed',
                'stability': 'stable'
            }
        },
        {
            "name": "Quality Priority", 
            "requirements": {
                'features': ['reflection'],
                'performance_priority': 'quality',
                'stability': 'stable'
            }
        },
        {
            "name": "Balanced",
            "requirements": {
                'features': [],
                'performance_priority': 'balanced',
                'stability': 'stable'
            }
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"\n--- {scenario['name']} Scenario ---")
        try:
            recommended_version = PipelineFactory.get_recommended_version(scenario['requirements'])
            logger.info(f"Recommended version: {recommended_version}")
            
            # Could create the pipeline here with auto selection
            # config = load_config("configs/global_config.yaml")  
            # pipeline = PipelineFactory.create_auto_pipeline(scenario['requirements'], config)
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario['name']}: {e}")


def main():
    """Main demonstration function."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Reorganized Recommendation System Demo")
    logger.info("=" * 60)
    
    try:
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Demo 1: Pipeline Factory Features
        demo_pipeline_factory()
        
        # Demo 2: Pipeline Execution
        demo_pipeline_execution()
        
        # Demo 3: Auto Pipeline Selection
        demo_auto_pipeline()
        
        logger.info("\n" + "=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("\nNew Project Structure Benefits:")
        logger.info("✓ Versioned pipelines (v1, v2, etc.)")
        logger.info("✓ Centralized configuration management") 
        logger.info("✓ Pipeline factory for easy instantiation")
        logger.info("✓ Automatic pipeline selection based on requirements")
        logger.info("✓ Better separation of concerns")
        logger.info("✓ Support for reflection and advanced features")
        logger.info("✓ Production-ready structure with monitoring and evaluation")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 