#!/usr/bin/env python3
"""
Project Reorganization Script

This script migrates the existing project structure to the new organized structure
supporting versioned pipelines and better separation of concerns.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# New directories to create
NEW_DIRECTORIES = [
    'configs/environments',
    'configs/pipelines', 
    'configs/models',
    'data/raw',
    'data/processed',
    'data/features',
    'data/embeddings',
    'data/schemas',
    'data/connectors/bloomberg',
    'data/connectors/refinitiv',
    'data/connectors/crm',
    'data/connectors/publications',
    'data/connectors/utils',
    'pipelines/base',
    'pipelines/v1_basic_rag',
    'pipelines/v2_rag_reflection',
    'pipelines/v3_rag_graph',
    'pipelines/v4_agentic_rag',
    'pipelines/utils',
    'llm/clients',
    'llm/prompts/templates',
    'llm/prompts/versions/v1',
    'llm/prompts/versions/v2',
    'llm/prompts/versions/v3',
    'llm/autotuning/experiments',
    'llm/utils',
    'evaluation/framework',
    'evaluation/metrics',
    'evaluation/evaluators',
    'evaluation/experiments/results',
    'evaluation/experiments/notebooks',
    'evaluation/reports/templates',
    'evaluation/reports/generated',
    'scripts/deployment/docker',
    'scripts/deployment/kubernetes',
    'scripts/scheduling',
    'scripts/data',
    'scripts/monitoring',
    'scripts/utilities',
    'orchestration/airflow/dags',
    'orchestration/airflow/plugins',
    'orchestration/prefect/flows',
    'orchestration/prefect/tasks',
    'orchestration/schedulers',
    'autotuning/optimizers',
    'autotuning/objectives',
    'autotuning/experiments/results',
    'autotuning/configs',
    'monitoring/metrics',
    'monitoring/dashboards/grafana',
    'monitoring/alerts',
    'monitoring/logs',
    'api/v1',
    'api/middleware',
    'api/utils',
    'tests/unit',
    'tests/integration',
    'tests/e2e',
    'tests/fixtures',
    'notebooks/exploratory',
    'notebooks/experiments',
    'notebooks/analysis',
    'notebooks/tutorials',
    'docs/api',
    'docs/architecture',
    'docs/deployment',
    'docs/tutorials',
    'docs/development',
    'tools/ci_cd/github_actions',
    'tools/ci_cd/jenkins',
    'tools/ci_cd/gitlab_ci',
    'tools/quality',
    'tools/utilities'
]


def create_directory_structure():
    """Create the new directory structure."""
    logger.info("Creating new directory structure...")
    
    for directory in NEW_DIRECTORIES:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
        
        # Create __init__.py files for Python packages
        if any(python_dir in directory for python_dir in ['pipelines', 'llm', 'evaluation', 'data', 'autotuning', 'monitoring', 'api']):
            init_file = path / '__init__.py'
            if not init_file.exists():
                init_file.touch()
                logger.info(f"Created __init__.py: {init_file}")


def backup_current_structure():
    """Create a backup of the current structure."""
    backup_dir = Path('backup_original_structure')
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    logger.info("Creating backup of current structure...")
    
    # Copy important directories
    for item in ['config', 'data_connectors', 'core_pipeline', 'llm_services', 'evaluation', 'orchestration', 'prompts']:
        if Path(item).exists():
            shutil.copytree(item, backup_dir / item)
            logger.info(f"Backed up: {item}")


def move_data_connectors():
    """Move and reorganize data connectors."""
    logger.info("Reorganizing data connectors...")
    
    # Move base connector
    if Path('data_connectors/base_connector.py').exists():
        shutil.copy2('data_connectors/base_connector.py', 'data/connectors/base.py')
        logger.info("Moved base_connector.py to data/connectors/base.py")
    
    # Move schemas
    if Path('data_connectors/schemas.py').exists():
        shutil.copy2('data_connectors/schemas.py', 'data/schemas/__init__.py')
        logger.info("Moved schemas.py to data/schemas/__init__.py")
    
    # Move specific connectors
    connector_mappings = {
        'bloomberg_chat': 'data/connectors/bloomberg',
        'refinitiv_chat': 'data/connectors/refinitiv',
    }
    
    for old_name, new_path in connector_mappings.items():
        old_path = Path(f'data_connectors/{old_name}')
        if old_path.exists():
            # Copy all files from old directory to new
            for file in old_path.glob('*.py'):
                shutil.copy2(file, f'{new_path}/{file.name}')
                logger.info(f"Moved {file} to {new_path}/{file.name}")
    
    # Move individual connector files
    individual_connectors = [
        'crm_connector.py',
        'kyc_connector.py', 
        'publications_connector.py',
        'readership_connector.py',
        'rfq_connector.py'
    ]
    
    for connector in individual_connectors:
        old_file = Path(f'data_connectors/{connector}')
        if old_file.exists():
            # Determine target directory based on connector type
            if 'crm' in connector:
                target = 'data/connectors/crm/'
            elif 'publications' in connector:
                target = 'data/connectors/publications/'
            else:
                target = 'data/connectors/utils/'
            
            Path(target).mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_file, f'{target}{connector}')
            logger.info(f"Moved {connector} to {target}")


def reorganize_configs():
    """Reorganize configuration files."""
    logger.info("Reorganizing configuration files...")
    
    # Move existing configs
    config_files = [
        'business_rules.yaml',
        'evaluation_config.yaml', 
        'pipeline_config.yaml',
        'prompts.yaml'
    ]
    
    for config_file in config_files:
        old_path = Path(f'config/{config_file}')
        if old_path.exists():
            shutil.copy2(old_path, f'configs/{config_file}')
            logger.info(f"Moved {config_file} to configs/")


def main():
    """Main migration function."""
    logger.info("Starting project reorganization...")
    
    try:
        # Step 1: Backup current structure
        backup_current_structure()
        
        # Step 2: Create new directory structure
        create_directory_structure()
        
        # Step 3: Reorganize components
        move_data_connectors()
        reorganize_configs()
        
        logger.info("Basic reorganization completed!")
        logger.info("Next steps: Run the full reorganization script")
        
    except Exception as e:
        logger.error(f"Error during reorganization: {e}")
        raise


if __name__ == "__main__":
    main() 