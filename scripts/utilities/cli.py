#!/usr/bin/env python3
"""
Command Line Interface for the Recommendation System

Provides easy access to pipeline management, evaluation, and system operations.
"""

import click
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from pipelines.factory import PipelineFactory


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', default='configs/global_config.yaml', help='Global config file')
@click.pass_context
def cli(ctx, verbose, config):
    """Recommendation System CLI - Manage pipelines, run evaluations, and more."""
    
    # Set up logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose


@cli.group()
def pipeline():
    """Pipeline management commands."""
    pass


@pipeline.command('list')
@click.option('--include-experimental', '-e', is_flag=True, help='Include experimental versions')
@click.option('--include-development', '-d', is_flag=True, help='Include development versions')
def list_pipelines(include_experimental, include_development):
    """List available pipeline versions."""
    versions = PipelineFactory.list_available_versions(
        include_experimental=include_experimental,
        include_development=include_development
    )
    
    click.echo("Available Pipeline Versions:")
    click.echo("=" * 40)
    
    for version in versions:
        try:
            info = PipelineFactory.get_pipeline_info(version)
            status_emoji = {
                'stable': '✅',
                'experimental': '⚠️',
                'development': '🚧'
            }.get(info['status'], '❓')
            
            click.echo(f"{status_emoji} {version}")
            click.echo(f"    Name: {info['name']}")
            click.echo(f"    Description: {info['description']}")
            click.echo(f"    Status: {info['status']}")
            
            features = info.get('features', {})
            enabled_features = [f for f, enabled in features.items() if enabled]
            if enabled_features:
                click.echo(f"    Features: {', '.join(enabled_features)}")
            click.echo()
        except Exception as e:
            click.echo(f"❌ {version} - Error: {e}")


@pipeline.command('info')
@click.argument('version')
def pipeline_info(version):
    """Get detailed information about a pipeline version."""
    try:
        info = PipelineFactory.get_pipeline_info(version)
        
        click.echo(f"Pipeline: {version}")
        click.echo("=" * 50)
        click.echo(f"Name: {info['name']}")
        click.echo(f"Description: {info['description']}")
        click.echo(f"Status: {info['status']}")
        click.echo(f"Class: {info['class']}")
        click.echo(f"Min Config Version: {info['min_config_version']}")
        
        click.echo("\nFeatures:")
        features = info.get('features', {})
        for feature, enabled in features.items():
            status = "✅" if enabled else "❌"
            click.echo(f"  {status} {feature}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@pipeline.command('recommend')
@click.option('--features', multiple=True, help='Required features (can specify multiple)')
@click.option('--priority', type=click.Choice(['speed', 'quality', 'balanced']), default='balanced')
@click.option('--stability', type=click.Choice(['stable', 'experimental', 'any']), default='stable')
def recommend_pipeline(features, priority, stability):
    """Get pipeline recommendation based on requirements."""
    requirements = {
        'features': list(features),
        'performance_priority': priority,
        'stability': stability
    }
    
    try:
        recommended = PipelineFactory.get_recommended_version(requirements)
        info = PipelineFactory.get_pipeline_info(recommended)
        
        click.echo(f"Recommended Pipeline: {recommended}")
        click.echo(f"Name: {info['name']}")
        click.echo(f"Description: {info['description']}")
        click.echo(f"Status: {info['status']}")
        
        click.echo(f"\nRequirements matched:")
        click.echo(f"  Features: {list(features) if features else 'None'}")
        click.echo(f"  Priority: {priority}")
        click.echo(f"  Stability: {stability}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@pipeline.command('compare')
@click.argument('version1')
@click.argument('version2')
def compare_pipelines(version1, version2):
    """Compare two pipeline versions."""
    try:
        comparison = PipelineFactory.compare_versions(version1, version2)
        
        click.echo(f"Comparison: {version1} vs {version2}")
        click.echo("=" * 50)
        
        if comparison['added_features']:
            click.echo(f"✅ Added features in {version2}:")
            for feature in comparison['added_features']:
                click.echo(f"  + {feature}")
        
        if comparison['removed_features']:
            click.echo(f"❌ Removed features in {version2}:")
            for feature in comparison['removed_features']:
                click.echo(f"  - {feature}")
        
        click.echo(f"\nCompatibility: {comparison['compatibility']}")
        click.echo(f"Recommendation: {comparison['recommendation']}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@pipeline.command('validate')
def validate_pipelines():
    """Validate all registered pipelines."""
    click.echo("Validating all pipelines...")
    click.echo("=" * 40)
    
    factory = PipelineFactory()
    results = factory.validate_all_pipelines()
    
    valid_count = sum(1 for r in results.values() if r['status'] == 'valid')
    total_count = len(results)
    
    for version, result in results.items():
        if result['status'] == 'valid':
            click.echo(f"✅ {version} - Valid")
        else:
            click.echo(f"❌ {version} - Invalid: {result['error']}")
    
    click.echo(f"\nSummary: {valid_count}/{total_count} pipelines valid")


@cli.group()
def run():
    """Run pipeline operations."""
    pass


@run.command('demo')
@click.option('--pipeline', '-p', default='auto', help='Pipeline version to use (default: auto)')
@click.option('--query', '-q', default='AI applications in finance', help='Demo query')
def run_demo(pipeline, query):
    """Run a demo of the pipeline system."""
    click.echo(f"Running demo with pipeline: {pipeline}")
    click.echo(f"Query: {query}")
    click.echo("=" * 50)
    
    try:
        if pipeline == 'auto':
            requirements = {'performance_priority': 'balanced', 'stability': 'stable'}
            recommended = PipelineFactory.get_recommended_version(requirements)
            click.echo(f"Auto-selected pipeline: {recommended}")
            pipeline = recommended
        
        # Load config
        config_path = f"configs/pipelines/{pipeline}.yaml"
        if not Path(config_path).exists():
            click.echo(f"Config file not found: {config_path}")
            click.echo("Using default configuration...")
            config = {'name': pipeline, 'version': '1.0'}
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        
        # Create pipeline
        instance = PipelineFactory.create_pipeline(pipeline, config)
        info = instance.get_pipeline_info()
        
        click.echo(f"✅ Pipeline created: {info['name']}")
        click.echo(f"Description: {info['description']}")
        
        # Health check
        health = instance.health_check()
        click.echo(f"Health: {health['status']}")
        
        click.echo(f"\nDemo completed successfully!")
        
    except Exception as e:
        click.echo(f"Demo failed: {e}", err=True)
        sys.exit(1)


@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command('validate')
@click.argument('config_file')
def validate_config(config_file):
    """Validate a configuration file."""
    try:
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        
        click.echo(f"✅ Configuration file {config_file} is valid YAML")
        click.echo(f"Keys found: {list(config_data.keys())}")
        
    except Exception as e:
        click.echo(f"❌ Invalid configuration: {e}", err=True)
        sys.exit(1)


@config.command('show')
@click.argument('config_file')
def show_config(config_file):
    """Display configuration file contents."""
    try:
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        
        click.echo(f"Configuration: {config_file}")
        click.echo("=" * 50)
        click.echo(yaml.dump(config_data, default_flow_style=False))
        
    except Exception as e:
        click.echo(f"Error reading config: {e}", err=True)
        sys.exit(1)


@cli.command('version')
def version():
    """Show system version information."""
    click.echo("Recommendation System v2.0.0")
    click.echo("Reorganized with versioned pipelines")
    
    # Show available versions
    versions = PipelineFactory.list_available_versions(include_experimental=True)
    click.echo(f"Available pipeline versions: {len(versions)}")
    for v in versions:
        click.echo(f"  - {v}")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main() 