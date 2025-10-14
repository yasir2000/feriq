"""
CLI Commands for Listing Framework Components Output

Comprehensive listing capabilities for all Feriq framework components.
"""

import click
import os
import json
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from feriq.cli.utils import print_success, print_error, print_info, print_header, print_table


@click.group(name='list')
def list_group():
    """List outputs and artifacts from all Feriq framework components."""
    pass


@list_group.command('components')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed component information')
def list_components(detailed: bool):
    """List all available Feriq framework components."""
    
    print_header("🏗️  Feriq Framework Components")
    
    components = [
        {
            "emoji": "🎭",
            "name": "Dynamic Role Designer",
            "description": "Automatically creates and assigns roles based on task requirements and context",
            "status": "✅ Available",
            "outputs": ["role_definitions", "role_assignments", "role_templates"]
        },
        {
            "emoji": "📋", 
            "name": "Task Designer & Allocator",
            "description": "Intelligently breaks down goals into tasks and optimally assigns them to agents",
            "status": "✅ Available",
            "outputs": ["task_breakdowns", "task_assignments", "allocation_reports"]
        },
        {
            "emoji": "📊",
            "name": "Plan Designer", 
            "description": "Creates comprehensive execution plans with resource allocation and timeline management",
            "status": "✅ Available",
            "outputs": ["execution_plans", "resource_allocations", "timeline_schedules"]
        },
        {
            "emoji": "👁️",
            "name": "Plan Observer",
            "description": "Real-time monitoring of plan execution with alerts and performance metrics", 
            "status": "✅ Available",
            "outputs": ["execution_logs", "performance_metrics", "status_reports", "alerts"]
        },
        {
            "emoji": "🎯",
            "name": "Goal-Oriented Agents",
            "description": "Intelligent agents that work towards specific goals with learning and adaptation",
            "status": "✅ Available", 
            "outputs": ["agent_configs", "goal_progress", "learning_logs", "adaptations"]
        },
        {
            "emoji": "🎼",
            "name": "Workflow Orchestrator",
            "description": "Central coordinator for workflow execution and resource management",
            "status": "✅ Available",
            "outputs": ["workflow_definitions", "execution_results", "resource_usage", "coordination_logs"]
        },
        {
            "emoji": "💃",
            "name": "Choreographer", 
            "description": "Manages agent interactions, coordination patterns, and communication protocols",
            "status": "✅ Available",
            "outputs": ["interaction_patterns", "communication_logs", "coordination_matrices"]
        },
        {
            "emoji": "🧠",
            "name": "Reasoner",
            "description": "Advanced reasoning engine for decision-making, problem-solving, and strategic planning",
            "status": "✅ Available",
            "outputs": ["reasoning_results", "decision_trees", "strategic_recommendations", "problem_solutions"]
        }
    ]
    
    if detailed:
        for component in components:
            print(f"\n{component['emoji']} {component['name']}")
            print(f"   Description: {component['description']}")
            print(f"   Status: {component['status']}")
            print(f"   Outputs: {', '.join(component['outputs'])}")
    else:
        for component in components:
            print(f"{component['emoji']} {component['name']} - {component['status']}")
    
    print(f"\n💡 Use 'feriq list <component>' to see specific outputs")
    print(f"💡 Use 'feriq list actions' to see available actions")


@list_group.command('roles')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table')
@click.option('--filter', help='Filter roles by type or status')
def list_roles(format: str, filter: Optional[str]):
    """🎭 List outputs from Dynamic Role Designer."""
    
    print_header("🎭 Dynamic Role Designer Outputs")
    
    # Check for role definitions in project
    roles_data = get_component_outputs('roles', [
        'role_definitions.yaml',
        'role_assignments.json', 
        'role_templates.yaml',
        'dynamic_roles.json'
    ])
    
    if not roles_data:
        print_info("No role designer outputs found. Create roles with 'feriq agent create' or 'feriq role design'")
        return
    
    # Display role definitions
    if 'role_definitions.yaml' in roles_data:
        roles = roles_data['role_definitions.yaml']
        if format == 'json':
            print(json.dumps(roles, indent=2))
        elif format == 'yaml':
            print(yaml.dump(roles, default_flow_style=False))
        else:
            display_roles_table(roles, filter)
    
    # Show role assignments if available
    if 'role_assignments.json' in roles_data:
        print("\n📋 Role Assignments:")
        assignments = roles_data['role_assignments.json']
        for agent, role in assignments.items():
            print(f"  • {agent}: {role}")
    
    # Show available templates
    if 'role_templates.yaml' in roles_data:
        print("\n📝 Available Role Templates:")
        templates = roles_data['role_templates.yaml']
        for template_name in templates.keys():
            print(f"  • {template_name}")


@list_group.command('tasks')
@click.option('--status', type=click.Choice(['pending', 'in_progress', 'completed', 'failed']), help='Filter by task status')
@click.option('--agent', help='Filter by assigned agent')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table')
def list_tasks(status: Optional[str], agent: Optional[str], format: str):
    """📋 List outputs from Task Designer & Allocator."""
    
    print_header("📋 Task Designer & Allocator Outputs")
    
    tasks_data = get_component_outputs('tasks', [
        'task_breakdowns.json',
        'task_assignments.json',
        'allocation_reports.yaml',
        'task_dependencies.json'
    ])
    
    if not tasks_data:
        print_info("No task designer outputs found. Create tasks with 'feriq goal create' or 'feriq task design'")
        return
    
    # Display task breakdowns
    if 'task_breakdowns.json' in tasks_data:
        tasks = tasks_data['task_breakdowns.json']
        filtered_tasks = filter_tasks(tasks, status, agent)
        
        if format == 'json':
            print(json.dumps(filtered_tasks, indent=2))
        elif format == 'yaml':
            print(yaml.dump(filtered_tasks, default_flow_style=False))
        else:
            display_tasks_table(filtered_tasks)
    
    # Show allocation summary
    if 'allocation_reports.yaml' in tasks_data:
        print("\n📊 Task Allocation Summary:")
        reports = tasks_data['allocation_reports.yaml']
        for report in reports.get('allocation_summary', []):
            print(f"  • {report}")


@list_group.command('plans')
@click.option('--active-only', is_flag=True, help='Show only active plans')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table')
def list_plans(active_only: bool, format: str):
    """📊 List outputs from Plan Designer."""
    
    print_header("📊 Plan Designer Outputs")
    
    plans_data = get_component_outputs('plans', [
        'execution_plans.json',
        'resource_allocations.yaml',
        'timeline_schedules.json',
        'plan_templates.yaml'
    ])
    
    if not plans_data:
        print_info("No plan designer outputs found. Create plans with 'feriq plan create' or 'feriq workflow design'")
        return
    
    # Display execution plans
    if 'execution_plans.json' in plans_data:
        plans = plans_data['execution_plans.json']
        
        if active_only:
            plans = {k: v for k, v in plans.items() if v.get('status') == 'active'}
        
        if format == 'json':
            print(json.dumps(plans, indent=2))
        elif format == 'yaml':
            print(yaml.dump(plans, default_flow_style=False))
        else:
            display_plans_table(plans)
    
    # Show resource allocations
    if 'resource_allocations.yaml' in plans_data:
        print("\n💰 Resource Allocations:")
        allocations = plans_data['resource_allocations.yaml']
        for resource, allocation in allocations.items():
            print(f"  • {resource}: {allocation}")


@list_group.command('observations')
@click.option('--recent', '-r', type=int, default=10, help='Number of recent observations to show')
@click.option('--level', type=click.Choice(['info', 'warning', 'error', 'critical']), help='Filter by observation level')
def list_observations(recent: int, level: Optional[str]):
    """👁️ List outputs from Plan Observer."""
    
    print_header("👁️ Plan Observer Outputs")
    
    observations_data = get_component_outputs('observations', [
        'execution_logs.json',
        'performance_metrics.json',
        'status_reports.yaml',
        'alerts.json'
    ])
    
    if not observations_data:
        print_info("No plan observer outputs found. Start plan execution to generate observations.")
        return
    
    # Display execution logs
    if 'execution_logs.json' in observations_data:
        logs = observations_data['execution_logs.json']
        filtered_logs = filter_logs(logs, recent, level)
        
        print("📝 Recent Execution Logs:")
        for log in filtered_logs:
            timestamp = log.get('timestamp', 'Unknown')
            level_emoji = get_level_emoji(log.get('level', 'info'))
            message = log.get('message', '')
            print(f"  {level_emoji} [{timestamp}] {message}")
    
    # Show performance metrics
    if 'performance_metrics.json' in observations_data:
        print("\n📈 Performance Metrics:")
        metrics = observations_data['performance_metrics.json']
        for metric, value in metrics.items():
            print(f"  • {metric}: {value}")
    
    # Show active alerts
    if 'alerts.json' in observations_data:
        alerts = observations_data['alerts.json']
        active_alerts = [a for a in alerts if a.get('status') == 'active']
        if active_alerts:
            print(f"\n🚨 Active Alerts ({len(active_alerts)}):")
            for alert in active_alerts:
                severity = alert.get('severity', 'info')
                message = alert.get('message', '')
                print(f"  🔔 [{severity.upper()}] {message}")


@list_group.command('agents')
@click.option('--status', type=click.Choice(['active', 'idle', 'busy', 'offline']), help='Filter by agent status')
@click.option('--role', help='Filter by agent role')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table')
def list_agents(status: Optional[str], role: Optional[str], format: str):
    """🎯 List Goal-Oriented Agents outputs."""
    
    print_header("🎯 Goal-Oriented Agents Outputs")
    
    agents_data = get_component_outputs('agents', [
        'agent_configs.yaml',
        'goal_progress.json',
        'learning_logs.json',
        'adaptations.yaml'
    ])
    
    if not agents_data:
        print_info("No agent outputs found. Create agents with 'feriq agent create'")
        return
    
    # Display agent configurations
    if 'agent_configs.yaml' in agents_data:
        agents = agents_data['agent_configs.yaml']
        filtered_agents = filter_agents(agents, status, role)
        
        if format == 'json':
            print(json.dumps(filtered_agents, indent=2))
        elif format == 'yaml':
            print(yaml.dump(filtered_agents, default_flow_style=False))
        else:
            display_agents_table(filtered_agents)
    
    # Show goal progress
    if 'goal_progress.json' in agents_data:
        print("\n🎯 Goal Progress Summary:")
        progress = agents_data['goal_progress.json']
        for agent, goals in progress.items():
            print(f"  • {agent}: {len(goals)} active goals")
    
    # Show recent adaptations
    if 'adaptations.yaml' in agents_data:
        print("\n🔄 Recent Adaptations:")
        adaptations = agents_data['adaptations.yaml']
        for adaptation in adaptations.get('recent', [])[:5]:
            print(f"  • {adaptation}")


@list_group.command('workflows')
@click.option('--status', type=click.Choice(['running', 'completed', 'failed', 'paused']), help='Filter by workflow status')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table')
def list_workflows(status: Optional[str], format: str):
    """🎼 List Workflow Orchestrator outputs."""
    
    print_header("🎼 Workflow Orchestrator Outputs")
    
    workflows_data = get_component_outputs('workflows', [
        'workflow_definitions.yaml',
        'execution_results.json',
        'resource_usage.json',
        'coordination_logs.json'
    ])
    
    if not workflows_data:
        print_info("No workflow orchestrator outputs found. Create workflows with 'feriq workflow create'")
        return
    
    # Display workflow definitions
    if 'workflow_definitions.yaml' in workflows_data:
        workflows = workflows_data['workflow_definitions.yaml']
        
        if status:
            workflows = {k: v for k, v in workflows.items() if v.get('status') == status}
        
        if format == 'json':
            print(json.dumps(workflows, indent=2))
        elif format == 'yaml':
            print(yaml.dump(workflows, default_flow_style=False))
        else:
            display_workflows_table(workflows)
    
    # Show resource usage
    if 'resource_usage.json' in workflows_data:
        print("\n💻 Resource Usage:")
        usage = workflows_data['resource_usage.json']
        for resource, metrics in usage.items():
            print(f"  • {resource}: {metrics}")


@list_group.command('interactions')
@click.option('--pattern', help='Filter by interaction pattern')
@click.option('--recent', '-r', type=int, default=20, help='Number of recent interactions to show')
def list_interactions(pattern: Optional[str], recent: int):
    """💃 List Choreographer outputs."""
    
    print_header("💃 Choreographer Outputs")
    
    interactions_data = get_component_outputs('interactions', [
        'interaction_patterns.yaml',
        'communication_logs.json',
        'coordination_matrices.json'
    ])
    
    if not interactions_data:
        print_info("No choreographer outputs found. Agent interactions will generate outputs automatically.")
        return
    
    # Display interaction patterns
    if 'interaction_patterns.yaml' in interactions_data:
        patterns = interactions_data['interaction_patterns.yaml']
        
        if pattern:
            patterns = {k: v for k, v in patterns.items() if pattern.lower() in k.lower()}
        
        print("🔗 Interaction Patterns:")
        for pattern_name, pattern_data in patterns.items():
            print(f"  • {pattern_name}: {pattern_data.get('description', 'No description')}")
    
    # Show recent communications
    if 'communication_logs.json' in interactions_data:
        logs = interactions_data['communication_logs.json']
        recent_logs = logs[-recent:] if len(logs) > recent else logs
        
        print(f"\n💬 Recent Communications ({len(recent_logs)}):")
        for log in recent_logs:
            sender = log.get('sender', 'Unknown')
            receiver = log.get('receiver', 'Unknown')
            message_type = log.get('type', 'message')
            print(f"  • {sender} → {receiver}: {message_type}")


@list_group.command('reasoning')
@click.option('--type', type=click.Choice(['inductive', 'deductive', 'probabilistic', 'causal', 'abductive', 'analogical', 'temporal', 'spatial', 'hybrid', 'collaborative']), help='Filter by reasoning type')
@click.option('--recent', '-r', type=int, default=10, help='Number of recent reasoning results to show')
def list_reasoning(type: Optional[str], recent: int):
    """🧠 List Reasoner outputs."""
    
    print_header("🧠 Reasoner Outputs")
    
    reasoning_data = get_component_outputs('reasoning', [
        'reasoning_results.json',
        'decision_trees.yaml',
        'strategic_recommendations.json',
        'problem_solutions.json'
    ])
    
    if not reasoning_data:
        print_info("No reasoner outputs found. Use 'feriq reason' commands to generate reasoning outputs.")
        return
    
    # Display reasoning results
    if 'reasoning_results.json' in reasoning_data:
        results = reasoning_data['reasoning_results.json']
        
        if type:
            results = [r for r in results if r.get('reasoning_type') == type]
        
        recent_results = results[-recent:] if len(results) > recent else results
        
        print(f"🔍 Recent Reasoning Results ({len(recent_results)}):")
        for result in recent_results:
            reasoning_type = result.get('reasoning_type', 'unknown')
            confidence = result.get('confidence', 0)
            conclusion = result.get('conclusion', 'No conclusion')[:80]
            print(f"  • [{reasoning_type}] {conclusion}... (confidence: {confidence:.2f})")
    
    # Show strategic recommendations
    if 'strategic_recommendations.json' in reasoning_data:
        print("\n🎯 Strategic Recommendations:")
        recommendations = reasoning_data['strategic_recommendations.json']
        for rec in recommendations[:5]:
            priority = rec.get('priority', 'medium')
            recommendation = rec.get('recommendation', '')
            print(f"  • [{priority.upper()}] {recommendation}")


@list_group.command('actions')
@click.option('--component', help='Filter actions by component')
@click.option('--recent', '-r', type=int, default=15, help='Number of recent actions to show')
def list_actions(component: Optional[str], recent: int):
    """🎬 List all recent actions across framework components."""
    
    print_header("🎬 Framework Actions Log")
    
    actions_data = get_component_outputs('actions', [
        'action_history.json',
        'component_actions.json',
        'system_events.json'
    ])
    
    if not actions_data:
        print_info("No actions logged yet. Framework actions will be recorded automatically.")
        return
    
    # Display action history
    if 'action_history.json' in actions_data:
        actions = actions_data['action_history.json']
        
        if component:
            actions = [a for a in actions if a.get('component') == component]
        
        recent_actions = actions[-recent:] if len(actions) > recent else actions
        
        print(f"📋 Recent Actions ({len(recent_actions)}):")
        for action in recent_actions:
            timestamp = action.get('timestamp', 'Unknown')
            component_name = action.get('component', 'Unknown')
            action_type = action.get('action', 'Unknown')
            status = action.get('status', 'unknown')
            status_emoji = "✅" if status == "success" else "❌" if status == "error" else "⏳"
            
            print(f"  {status_emoji} [{timestamp}] {component_name}: {action_type}")
    
    # Show component action summary
    if 'component_actions.json' in actions_data:
        print("\n📊 Component Action Summary:")
        summary = actions_data['component_actions.json']
        for component_name, count in summary.items():
            print(f"  • {component_name}: {count} actions")


@list_group.command('generate-samples')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def generate_sample_outputs(confirm: bool):
    """🎭 Generate sample outputs for all framework components (for demonstration)."""
    
    if not confirm:
        click.echo("This will generate sample outputs for all Feriq framework components.")
        click.echo("This is useful for demonstration and testing the listing capabilities.")
        if not click.confirm("Do you want to continue?"):
            return
    
    try:
        from feriq.demos.sample_output_generator import SampleOutputGenerator
        
        print_info("Generating sample outputs for all framework components...")
        generator = SampleOutputGenerator()
        generator.generate_all_samples()
        
        print_success("Sample outputs generated successfully!")
        print_info("Use 'feriq list components' to see all available listings")
        print_info("Use 'feriq list <component>' to view specific component outputs")
        
    except ImportError as e:
        print_error(f"Sample generator not available: {e}")
    except Exception as e:
        print_error(f"Error generating sample outputs: {e}")


# Helper functions

def get_component_outputs(component_type: str, expected_files: List[str]) -> Dict[str, Any]:
    """Get outputs from a specific component type."""
    outputs = {}
    
    # Check in project outputs directory
    outputs_dir = Path.cwd() / 'outputs' / component_type
    if outputs_dir.exists():
        for filename in expected_files:
            file_path = outputs_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if filename.endswith('.json'):
                            outputs[filename] = json.load(f)
                        elif filename.endswith(('.yaml', '.yml')):
                            outputs[filename] = yaml.safe_load(f)
                        else:
                            outputs[filename] = f.read()
                except Exception as e:
                    print_error(f"Error reading {filename}: {e}")
    
    # Also check in project directories
    project_dirs = ['agents', 'goals', 'workflows', 'plans', 'logs']
    for dir_name in project_dirs:
        dir_path = Path.cwd() / dir_name
        if dir_path.exists():
            for filename in expected_files:
                file_path = dir_path / filename
                if file_path.exists() and filename not in outputs:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if filename.endswith('.json'):
                                outputs[filename] = json.load(f)
                            elif filename.endswith(('.yaml', '.yml')):
                                outputs[filename] = yaml.safe_load(f)
                            else:
                                outputs[filename] = f.read()
                    except Exception as e:
                        print_error(f"Error reading {filename}: {e}")
    
    return outputs


def display_roles_table(roles: Dict, filter_term: Optional[str]):
    """Display roles in table format."""
    if not roles:
        return
    
    print("Role Name | Type | Status | Description")
    print("-" * 60)
    
    for role_name, role_data in roles.items():
        if filter_term and filter_term.lower() not in role_name.lower():
            continue
        
        role_type = role_data.get('type', 'Unknown')
        status = role_data.get('status', 'Unknown') 
        description = role_data.get('description', 'No description')[:30]
        print(f"{role_name} | {role_type} | {status} | {description}")


def display_tasks_table(tasks: Dict):
    """Display tasks in table format."""
    if not tasks:
        return
    
    print("Task ID | Name | Status | Agent | Priority")
    print("-" * 50)
    
    for task_id, task_data in tasks.items():
        name = task_data.get('name', 'Unnamed')[:20]
        status = task_data.get('status', 'Unknown')
        agent = task_data.get('assigned_agent', 'Unassigned')
        priority = task_data.get('priority', 'Medium')
        print(f"{task_id} | {name} | {status} | {agent} | {priority}")


def display_plans_table(plans: Dict):
    """Display plans in table format."""
    if not plans:
        return
    
    print("Plan ID | Name | Status | Tasks | Progress")
    print("-" * 45)
    
    for plan_id, plan_data in plans.items():
        name = plan_data.get('name', 'Unnamed')[:20]
        status = plan_data.get('status', 'Unknown')
        task_count = len(plan_data.get('tasks', []))
        progress = plan_data.get('progress', 0)
        print(f"{plan_id} | {name} | {status} | {task_count} | {progress}%")


def display_agents_table(agents: Dict):
    """Display agents in table format."""
    if not agents:
        return
    
    print("Agent Name | Role | Status | Goals | Last Active")
    print("-" * 50)
    
    for agent_name, agent_data in agents.items():
        role = agent_data.get('role', 'Unknown')
        status = agent_data.get('status', 'Unknown')
        goal_count = len(agent_data.get('goals', []))
        last_active = agent_data.get('last_active', 'Never')
        print(f"{agent_name} | {role} | {status} | {goal_count} | {last_active}")


def display_workflows_table(workflows: Dict):
    """Display workflows in table format."""
    if not workflows:
        return
    
    print("Workflow ID | Name | Status | Agents | Progress")
    print("-" * 45)
    
    for workflow_id, workflow_data in workflows.items():
        name = workflow_data.get('name', 'Unnamed')[:20]
        status = workflow_data.get('status', 'Unknown')
        agent_count = len(workflow_data.get('agents', []))
        progress = workflow_data.get('progress', 0)
        print(f"{workflow_id} | {name} | {status} | {agent_count} | {progress}%")


def filter_tasks(tasks: Dict, status: Optional[str], agent: Optional[str]) -> Dict:
    """Filter tasks by status and agent."""
    if not status and not agent:
        return tasks
    
    filtered = {}
    for task_id, task_data in tasks.items():
        if status and task_data.get('status') != status:
            continue
        if agent and task_data.get('assigned_agent') != agent:
            continue
        filtered[task_id] = task_data
    
    return filtered


def filter_agents(agents: Dict, status: Optional[str], role: Optional[str]) -> Dict:
    """Filter agents by status and role."""
    if not status and not role:
        return agents
    
    filtered = {}
    for agent_name, agent_data in agents.items():
        if status and agent_data.get('status') != status:
            continue
        if role and agent_data.get('role') != role:
            continue
        filtered[agent_name] = agent_data
    
    return filtered


def filter_logs(logs: List, recent: int, level: Optional[str]) -> List:
    """Filter logs by level and return recent entries."""
    if level:
        logs = [log for log in logs if log.get('level') == level]
    
    return logs[-recent:] if len(logs) > recent else logs


def get_level_emoji(level: str) -> str:
    """Get emoji for log level."""
    emojis = {
        'info': 'ℹ️',
        'warning': '⚠️',
        'error': '❌',
        'critical': '🚨',
        'success': '✅'
    }
    return emojis.get(level.lower(), 'ℹ️')


# Integration function
def add_list_commands(main_cli):
    """Add comprehensive listing commands to the main Feriq CLI."""
    main_cli.add_command(list_group)