"""
Feriq CLI - Main Entry Point

Command-line interface for interacting with the Feriq framework.
"""

import click
import asyncio
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .commands import (
    init_group,
    model_group,
    agent_group,
    goal_group,
    workflow_group,
    status_group,
    interactive_group
)
from .utils import CliContext, print_banner, print_success, print_error, print_info
from .models import ModelManager


# CLI context initialization is handled by Click's context system


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
@click.version_option(version='1.0.0', prog_name='feriq')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool, debug: bool):
    """
    Feriq - Collaborative AI Agents Framework CLI
    
    A comprehensive command-line interface for managing and executing
    multi-agent workflows with intelligent coordination and reasoning.
    """
    # Initialize CLI context
    ctx.ensure_object(CliContext)
    if isinstance(ctx.obj, dict):
        ctx.obj = CliContext()
    
    ctx.obj.verbose = verbose
    ctx.obj.debug = debug
    
    if config:
        ctx.obj.load_config(config)
    
    # Print banner on first run
    if ctx.invoked_subcommand != 'version':
        print_banner()


# Add command groups
cli.add_command(init_group)
cli.add_command(model_group)
cli.add_command(agent_group)
cli.add_command(goal_group)
cli.add_command(workflow_group)
cli.add_command(status_group)
cli.add_command(interactive_group)

# Add reasoning commands
try:
    from feriq.cli.reasoning_commands import add_reasoning_commands
    add_reasoning_commands(cli)
except ImportError:
    pass  # Reasoning commands not available

# Add reasoning planning commands
try:
    from feriq.cli.reasoning_planning_commands import add_reasoning_planning_commands
    add_reasoning_planning_commands(cli)
except ImportError:
    pass  # Reasoning planning commands not available

# Add comprehensive listing commands
try:
    from feriq.cli.list_commands import add_list_commands
    add_list_commands(cli)
except ImportError:
    pass  # List commands not available


@cli.command()
@click.option('--output', '-o', type=click.Choice(['json', 'yaml', 'table']), default='table')
def version(output):
    """Show version information."""
    version_info = {
        'feriq_version': '1.0.0',
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': sys.platform,
        'build_date': '2025-10-14',
        'components': [
            'Framework Core',
            'Dynamic Role Designer',
            'Task Designer & Allocator',
            'Plan Designer',
            'Plan Observer',
            'Workflow Orchestrator',
            'Choreographer',
            'Reasoner'
        ]
    }
    
    if output == 'json':
        click.echo(json.dumps(version_info, indent=2))
    elif output == 'yaml':
        import yaml
        click.echo(yaml.dump(version_info, default_flow_style=False))
    else:
        print_info("Feriq Framework Information")
        print(f"Version: {version_info['feriq_version']}")
        print(f"Python: {version_info['python_version']}")
        print(f"Platform: {version_info['platform']}")
        print(f"Build Date: {version_info['build_date']}")
        print("\nComponents:")
        for component in version_info['components']:
            print(f"  ✓ {component}")


@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), default='json')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def config(format, output):
    """Generate or show configuration template."""
    config_template = {
        'framework': {
            'name': 'feriq-project',
            'description': 'Feriq collaborative AI agents project',
            'version': '1.0.0'
        },
        'logging': {
            'level': 'INFO',
            'structured': True,
            'audit_enabled': True,
            'file_path': 'logs/feriq.log'
        },
        'models': {
            'default_provider': 'ollama',
            'providers': {
                'ollama': {
                    'base_url': 'http://localhost:11434',
                    'default_model': 'llama2'
                },
                'openai': {
                    'api_key': '${OPENAI_API_KEY}',
                    'default_model': 'gpt-3.5-turbo'
                },
                'anthropic': {
                    'api_key': '${ANTHROPIC_API_KEY}',
                    'default_model': 'claude-3-sonnet'
                }
            }
        },
        'orchestrator': {
            'max_concurrent_workflows': 5,
            'default_strategy': 'dynamic',
            'resource_limits': {
                'compute_nodes': 10,
                'memory_gb': 64,
                'storage_gb': 1000
            }
        },
        'reasoner': {
            'default_reasoning_type': 'probabilistic',
            'confidence_threshold': 0.7,
            'knowledge_base_path': 'data/knowledge_base.json'
        },
        'choreographer': {
            'message_timeout': 300,
            'max_coordination_participants': 10,
            'coordination_patterns': [
                'pipeline',
                'scatter_gather',
                'consensus',
                'broadcast'
            ]
        }
    }
    
    if format == 'yaml':
        import yaml
        content = yaml.dump(config_template, default_flow_style=False, indent=2)
    else:
        content = json.dumps(config_template, indent=2)
    
    if output:
        with open(output, 'w') as f:
            f.write(content)
        print_success(f"Configuration template saved to {output}")
    else:
        click.echo(content)


@cli.command()
@click.option('--check-models', is_flag=True, help='Check available LLM models')
@click.option('--check-components', is_flag=True, help='Check framework components')
@click.option('--check-dependencies', is_flag=True, help='Check Python dependencies')
def doctor(check_models, check_components, check_dependencies):
    """Run diagnostic checks on the Feriq installation."""
    print_info("Running Feriq diagnostic checks...")
    
    all_checks = not any([check_models, check_components, check_dependencies])
    
    if all_checks or check_dependencies:
        print("\n🔍 Checking Python dependencies...")
        _check_dependencies()
    
    if all_checks or check_components:
        print("\n🔍 Checking framework components...")
        _check_components()
    
    if all_checks or check_models:
        print("\n🔍 Checking available LLM models...")
        _check_models()
    
    print_success("\n✅ Diagnostic checks completed!")


def _check_dependencies():
    """Check if required dependencies are installed."""
    required_deps = [
        'click', 'pydantic', 'networkx', 'structlog', 
        'asyncio', 'aiofiles', 'requests', 'PyYAML'
    ]
    
    for dep in required_deps:
        try:
            __import__(dep.replace('-', '_'))
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} (missing)")


def _check_components():
    """Check if framework components are available."""
    components = [
        ('Framework Core', 'feriq.core.framework'),
        ('Dynamic Role Designer', 'feriq.components.role_designer'),
        ('Task Designer', 'feriq.components.task_designer'),
        ('Plan Designer', 'feriq.components.plan_designer'),
        ('Plan Observer', 'feriq.components.plan_observer'),
        ('Workflow Orchestrator', 'feriq.components.orchestrator'),
        ('Choreographer', 'feriq.components.choreographer'),
        ('Reasoner', 'feriq.components.reasoner')
    ]
    
    for name, module in components:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} ({e})")


def _check_models():
    """Check available LLM models."""
    from .models import ModelManager
    
    model_manager = ModelManager()
    
    # Check Ollama
    try:
        ollama_models = model_manager.list_ollama_models()
        if ollama_models:
            print(f"  ✓ Ollama ({len(ollama_models)} models available)")
            for model in ollama_models[:3]:  # Show first 3
                print(f"    - {model}")
            if len(ollama_models) > 3:
                print(f"    ... and {len(ollama_models) - 3} more")
        else:
            print("  ⚠ Ollama (no models found)")
    except Exception as e:
        print(f"  ✗ Ollama (connection failed: {e})")
    
    # Check OpenAI
    try:
        import os
        if os.getenv('OPENAI_API_KEY'):
            print("  ✓ OpenAI (API key configured)")
        else:
            print("  ⚠ OpenAI (API key not configured)")
    except Exception:
        print("  ✗ OpenAI (not available)")
    
    # Check Anthropic
    try:
        import os
        if os.getenv('ANTHROPIC_API_KEY'):
            print("  ✓ Anthropic (API key configured)")
        else:
            print("  ⚠ Anthropic (API key not configured)")
    except Exception:
        print("  ✗ Anthropic (not available)")


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        print_error("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()