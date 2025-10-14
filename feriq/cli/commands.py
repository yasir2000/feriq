"""
CLI Commands for Feriq Framework

All command groups and individual commands for the Feriq CLI.
"""

import os
import json
import asyncio
from typing import Optional
from pathlib import Path
import click

from .utils import (
    CliContext, print_success, print_error, print_warning, print_info,
    print_header, print_table, confirm_action, get_user_input,
    ensure_project_directory, load_project_config, save_project_config,
    create_directory_structure, select_model_interactive
)
from .models import ModelManager, setup_model_interactive, list_models_command, pull_ollama_model


# Context object - temporarily remove to fix Path issue
# pass_context = click.make_pass_decorator(CliContext)


@click.group(name='init')
def init_group():
    """Initialize and setup commands."""
    pass


@init_group.command()
@click.option('--name', help='Project name')
@click.option('--template', type=click.Choice(['basic', 'advanced', 'research']), 
              default='basic', help='Project template')
@click.option('--model-setup', is_flag=True, help='Setup model configuration during init')
def project(name, template, model_setup):
    """Initialize a new Feriq project."""
    
    if not name:
        name = get_user_input("Project name", default=Path.cwd().name)
    
    project_path = Path.cwd() / name
    
    if project_path.exists() and any(project_path.iterdir()):
        if not confirm_action(f"Directory {name} exists and is not empty. Continue?"):
            return
    
    project_path.mkdir(exist_ok=True)
    os.chdir(project_path)
    
    print_info(f"Initializing Feriq project: {name}")
    
    # Create directory structure
    create_directory_structure(project_path)
    
    # Create project configuration
    config = {
        'name': name,
        'version': '0.1.0',
        'template': template,
        'framework': {
            'version': '0.1.0',
            'features': ['role_designer', 'task_designer', 'plan_designer', 
                        'plan_observer', 'workflow_executor', 'reasoner']
        },
        'models': {},
        'agents': {},
        'goals': {},
        'workflows': {}
    }
    
    # Model setup
    if model_setup:
        model_config = setup_model_interactive()
        if model_config:
            config['models']['default'] = model_config
    
    save_project_config(config)
    
    # Create example files based on template
    create_template_files(project_path, template)
    
    print_success(f"Feriq project '{name}' initialized successfully!")
    
    if not model_setup:
        print_info("Run 'feriq model setup' to configure LLM models")


def create_template_files(project_path: Path, template: str):
    """Create template files based on template type."""
    
    # Basic example agent
    agent_example = """# Example Agent Configuration

name: research_agent
role: Research Specialist
goal: Conduct thorough research on given topics
backstory: |
  You are an experienced research specialist with expertise in gathering,
  analyzing, and synthesizing information from various sources.

capabilities:
  - information_gathering
  - data_analysis
  - report_generation

tools:
  - web_search
  - document_analysis
  - summarization

model:
  provider: ollama
  name: llama2
"""
    
    with open(project_path / 'agents' / 'example_agent.yaml', 'w') as f:
        f.write(agent_example)
    
    # Example goal
    goal_example = """# Example Goal Configuration

name: research_goal
title: Comprehensive Market Research
description: |
  Conduct comprehensive market research for a new product launch,
  including competitor analysis, target audience identification,
  and market trends analysis.

objectives:
  - Analyze competitor landscape
  - Identify target demographics
  - Research market trends
  - Provide actionable insights

success_criteria:
  - Complete competitor analysis report
  - Target audience profile
  - Market trend summary
  - Strategic recommendations

priority: high
estimated_duration: "4 hours"
"""
    
    with open(project_path / 'goals' / 'example_goal.yaml', 'w') as f:
        f.write(goal_example)
    
    # Example workflow
    workflow_example = """# Example Workflow Configuration

name: research_workflow
title: Market Research Workflow
description: Automated workflow for conducting market research

stages:
  - name: planning
    description: Plan research approach
    agents: [research_agent]
    tasks:
      - define_research_scope
      - identify_data_sources
    
  - name: data_collection
    description: Collect research data
    agents: [research_agent]
    tasks:
      - gather_competitor_data
      - collect_market_data
      - survey_target_audience
    
  - name: analysis
    description: Analyze collected data
    agents: [research_agent]
    tasks:
      - analyze_competitors
      - identify_trends
      - segment_audience
    
  - name: reporting
    description: Generate research report
    agents: [research_agent]
    tasks:
      - create_summary
      - generate_insights
      - provide_recommendations

dependencies:
  - planning -> data_collection
  - data_collection -> analysis
  - analysis -> reporting
"""
    
    with open(project_path / 'workflows' / 'example_workflow.yaml', 'w') as f:
        f.write(workflow_example)


@click.group(name='model')
def model_group():
    """Model management commands."""
    pass


@model_group.command()
def setup():
    """Setup model configuration interactively."""
    model_config = setup_model_interactive()
    
    if model_config:
        config = load_project_config()
        if 'models' not in config:
            config['models'] = {}
        
        config['models']['default'] = model_config
        save_project_config(config)
        
        print_success("Model configuration saved to project")


@model_group.command()
def list():
    """List available models."""
    list_models_command()


@model_group.command()
@click.argument('model_name')
def pull(model_name):
    """Pull an Ollama model."""
    if pull_ollama_model(model_name):
        print_success(f"Model {model_name} pulled successfully")
    else:
        print_error(f"Failed to pull model {model_name}")


@model_group.command()
@click.argument('provider')
@click.argument('model_name')
def test(provider, model_name):
    """Test a specific model."""
    manager = ModelManager()
    
    print_info(f"Testing {provider}:{model_name}...")
    
    if manager.test_model(provider, model_name):
        print_success("Model test successful!")
    else:
        print_error("Model test failed!")


@click.group(name='agent')
def agent_group():
    """Agent management commands."""
    pass


@agent_group.command()
def list():
    """List all agents in the project."""
    try:
        current_dir_str = os.getcwd()
        agents_dir_str = os.path.join(current_dir_str, 'agents')
        
        if not os.path.exists(agents_dir_str):
            print_warning("No agents directory found")
            return
        
        # Use os.listdir instead of Path.glob to avoid Click issues
        agent_files = []
        for filename in os.listdir(agents_dir_str):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                agent_files.append(os.path.join(agents_dir_str, filename))
        
        if not agent_files:
            print_info("No agents found")
            return
        
        # Simple list output instead of table
        print_header("Project Agents")
        for agent_file_path in agent_files:
            try:
                import yaml
                with open(agent_file_path, 'r') as f:
                    agent_config = yaml.safe_load(f)
                
                name = agent_config.get('name', 'Unknown')
                role = agent_config.get('role', 'Unknown')
                filename = os.path.basename(agent_file_path)
                print(f"  📋 {name} ({role}) - {filename}")
                
            except Exception as e:
                filename = os.path.basename(agent_file_path)
                print(f"  ❌ {filename} - Error reading file")
        
    except Exception as e:
        print_error(f"Error listing agents: {e}")
        import traceback
        traceback.print_exc()


@agent_group.command()
@click.argument('name')
def create(name):
    """Create a new agent."""
    ensure_project_directory()
    
    agent_path = Path.cwd() / 'agents' / f'{name}.yaml'
    
    if agent_path.exists():
        if not confirm_action(f"Agent {name} already exists. Overwrite?"):
            return
    
    # Interactive agent creation
    role = get_user_input("Agent role")
    goal = get_user_input("Agent goal")
    backstory = get_user_input("Agent backstory")
    
    agent_config = {
        'name': name,
        'role': role,
        'goal': goal,
        'backstory': backstory,
        'capabilities': [],
        'tools': [],
        'model': {
            'provider': 'ollama',
            'name': 'llama2'
        }
    }
    
    import yaml
    with open(agent_path, 'w') as f:
        yaml.dump(agent_config, f, default_flow_style=False, indent=2)
    
    print_success(f"Agent {name} created at {agent_path}")


@agent_group.command()
@click.argument('name')
@click.option('--goal', help='Goal to execute')
def run(name, goal):
    """Run an agent with a specific goal."""
    ensure_project_directory()
    
    print_info(f"Running agent {name}...")
    
    # This would integrate with the framework
    from ..core.framework import FeriqFramework
    from ..utils.config import Config
    
    config = Config()
    project_config = load_project_config()
    if project_config:
        config.update(project_config)
    
    framework = FeriqFramework(config)
    
    # Load and run agent
    # Implementation would go here
    print_warning("Agent execution not yet implemented")


@click.group(name='goal')
def goal_group():
    """Goal management commands."""
    pass


@goal_group.command()
def list():
    """List all goals in the project."""
    ensure_project_directory()
    
    goal_files = get_yaml_files_in_directory('goals')
    
    if not goal_files:
        print_info("No goals found")
        return
    
    headers = ['Name', 'Title', 'Priority', 'File']
    rows = []
    
    for goal_file_path in goal_files:
        try:
            import yaml
            with open(goal_file_path, 'r') as f:
                goal_config = yaml.safe_load(f)
            
            filename = os.path.basename(goal_file_path)
            rows.append([
                goal_config.get('name', 'Unknown'),
                goal_config.get('title', 'Unknown'),
                goal_config.get('priority', 'medium'),
                filename
            ])
        except Exception as e:
            filename = os.path.basename(goal_file_path)
            rows.append(['Error', 'Error', 'Error', filename])
    
    print_table(headers, rows, "Project Goals")


@goal_group.command()
@click.argument('name')
def create(name):
    """Create a new goal."""
    ensure_project_directory()
    
    goal_path = Path.cwd() / 'goals' / f'{name}.yaml'
    
    if goal_path.exists():
        if not confirm_action(f"Goal {name} already exists. Overwrite?"):
            return
    
    # Interactive goal creation
    title = get_user_input("Goal title")
    description = get_user_input("Goal description")
    priority = click.prompt(
        "Priority", 
        type=click.Choice(['low', 'medium', 'high', 'critical']),
        default='medium'
    )
    
    goal_config = {
        'name': name,
        'title': title,
        'description': description,
        'priority': priority,
        'objectives': [],
        'success_criteria': [],
        'estimated_duration': '1 hour'
    }
    
    import yaml
    with open(goal_path, 'w') as f:
        yaml.dump(goal_config, f, default_flow_style=False, indent=2)
    
    print_success(f"Goal {name} created at {goal_path}")


@click.group(name='workflow')
def workflow_group():
    """Workflow management commands."""
    pass


@workflow_group.command()
def list():
    """List all workflows in the project."""
    ensure_project_directory()
    
    workflow_files = get_yaml_files_in_directory('workflows')
    
    if not workflow_files:
        print_info("No workflows found")
        return
    
    headers = ['Name', 'Title', 'Stages', 'File']
    rows = []
    
    for workflow_file_path in workflow_files:
        try:
            import yaml
            with open(workflow_file_path, 'r') as f:
                workflow_config = yaml.safe_load(f)
            
            stages = len(workflow_config.get('stages', []))
            filename = os.path.basename(workflow_file_path)
            
            rows.append([
                workflow_config.get('name', 'Unknown'),
                workflow_config.get('title', 'Unknown'),
                str(stages),
                filename
            ])
        except Exception as e:
            filename = os.path.basename(workflow_file_path)
            rows.append(['Error', 'Error', 'Error', filename])
    
    print_table(headers, rows, "Project Workflows")


@workflow_group.command()
@click.argument('name')
def run(name):
    """Run a workflow."""
    ensure_project_directory()
    
    print_info(f"Running workflow {name}...")
    
    # This would integrate with the framework
    print_warning("Workflow execution not yet implemented")


@click.group(name='status')
def status_group():
    """Status and monitoring commands."""
    pass


def get_yaml_files_in_directory(directory_name: str) -> list:
    """Get list of YAML files in a directory using os module to avoid Click issues."""
    try:
        current_dir_str = os.getcwd()
        target_dir_str = os.path.join(current_dir_str, directory_name)
        
        if not os.path.exists(target_dir_str):
            return []
        
        yaml_files = []
        for filename in os.listdir(target_dir_str):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                yaml_files.append(os.path.join(target_dir_str, filename))
        
        return yaml_files
    except Exception:
        return []


def count_resource_files(resource_dir_name: str) -> int:
    """Count YAML files in a resource directory without using Path objects in Click context."""
    try:
        current_dir_str = os.getcwd()
        resource_dir_str = os.path.join(current_dir_str, resource_dir_name)
        
        if not os.path.exists(resource_dir_str):
            return 0
        
        count = 0
        for filename in os.listdir(resource_dir_str):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                count += 1
        
        return count
    except Exception:
        return 0


@status_group.command()
def show():
    """Show project status."""
    ensure_project_directory()
    
    config = load_project_config()
    
    print_header("Project Status")
    print(f"Name: {config.get('name', 'Unknown')}")
    print(f"Version: {config.get('version', 'Unknown')}")
    print(f"Template: {config.get('template', 'Unknown')}")
    
    # Count resources using os module to avoid Click issues
    agents_count = count_resource_files('agents')
    goals_count = count_resource_files('goals')
    workflows_count = count_resource_files('workflows')
    
    print_header("Resources")
    print(f"Agents: {agents_count}")
    print(f"Goals: {goals_count}")
    print(f"Workflows: {workflows_count}")
    
    # Model status
    if 'models' in config:
        print_header("Models")
        for name, model_config in config['models'].items():
            provider = model_config.get('provider', 'unknown')
            model = model_config.get('model', 'unknown')
            print(f"{name}: {provider}:{model}")


@click.group(name='interactive')
def interactive_group():
    """Interactive mode commands."""
    pass


@interactive_group.command()
def start():
    """Start interactive mode."""
    from .utils import print_banner
    
    print_banner()
    print_info("Starting Feriq interactive mode...")
    print_info("Type 'help' for available commands, 'exit' to quit")
    
    # Initialize framework without context for now
    from ..core.framework import FeriqFramework
    from ..utils.config import Config
    
    config = Config()
    project_config = load_project_config()
    if project_config:
        config.update(project_config)
    
    framework = FeriqFramework(config)
    
    while True:
        try:
            user_input = input("\nferiq> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print_info("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print_interactive_help()
                continue
            
            # Process command
            process_interactive_command(user_input, framework)
            
        except KeyboardInterrupt:
            print_info("\nGoodbye!")
            break
        except EOFError:
            print_info("\nGoodbye!")
            break


def print_interactive_help():
    """Print interactive mode help."""
    help_text = """
Available commands:
  help                    - Show this help
  status                  - Show project status
  agents list             - List agents
  goals list              - List goals
  workflows list          - List workflows
  models list             - List models
  run agent <name>        - Run an agent
  run workflow <name>     - Run a workflow
  chat                    - Start chat mode
  exit                    - Exit interactive mode
"""
    print(help_text)


def process_interactive_command(command: str, framework):
    """Process an interactive command."""
    parts = command.split()
    
    if not parts:
        return
    
    cmd = parts[0].lower()
    
    if cmd == 'status':
        # Show status
        print_info("Project status...")
    elif cmd == 'agents':
        if len(parts) > 1 and parts[1] == 'list':
            print_info("Listing agents...")
    elif cmd == 'chat':
        start_chat_mode(framework)
    else:
        print_warning(f"Unknown command: {command}")


def start_chat_mode(framework):
    """Start chat mode with the framework."""
    print_info("Starting chat mode... (type 'exit' to return)")
    
    while True:
        try:
            message = input("\nchat> ").strip()
            
            if not message:
                continue
            
            if message.lower() in ['exit', 'quit']:
                break
            
            # Process chat message with framework
            print_info("Processing message...")
            # Implementation would integrate with framework
            
        except KeyboardInterrupt:
            break