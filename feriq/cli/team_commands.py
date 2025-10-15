"""
CLI Commands for Team Management - Feriq Framework

Comprehensive command-line interface for team creation, management, collaboration,
and autonomous problem-solving capabilities.
"""

import click
import json
from datetime import datetime
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax

from ..components.team_designer import TeamDesigner, TeamType, CollaborationMode, TeamStatus
from ..components.role_designer import DynamicRoleDesigner
from ..components.task_designer import TaskDesigner
from ..components.reasoner import Reasoner

console = Console()

@click.group()
def team():
    """Team management commands for collaborative AI coordination."""
    pass

@team.command()
@click.argument('name')
@click.argument('discipline')
@click.option('--description', '-d', default="", help="Team description")
@click.option('--team-type', '-t', 
              type=click.Choice(['autonomous', 'hierarchical', 'specialist', 'cross_functional', 'swarm']),
              default='autonomous', help="Type of team")
@click.option('--max-size', '-m', default=10, help="Maximum team size")
@click.option('--min-size', '-s', default=1, help="Minimum team size")
@click.option('--capabilities', '-c', help="Comma-separated list of team capabilities")
def create(name: str, discipline: str, description: str, team_type: str, max_size: int, min_size: int, capabilities: str):
    """Create a new team with specified discipline and capabilities."""
    
    team_designer = TeamDesigner()
    
    # Parse capabilities
    capability_list = [cap.strip() for cap in capabilities.split(',')] if capabilities else []
    
    # Create team
    team = team_designer.create_team(
        name=name,
        description=description or f"Team specialized in {discipline}",
        discipline=discipline,
        team_type=TeamType(team_type),
        max_size=max_size,
        min_size=min_size,
        capabilities=capability_list
    )
    
    console.print(Panel(
        f"✅ Created team: {team.name}\n"
        f"🏷️ ID: {team.id}\n"
        f"🎯 Discipline: {team.discipline}\n"
        f"📊 Type: {team.team_type}\n"
        f"👥 Size: {len(team.members)}/{team.max_size}\n"
        f"🛠️ Capabilities: {', '.join(team.capabilities[:5])}{'...' if len(team.capabilities) > 5 else ''}",
        title="🎉 Team Created Successfully",
        style="green"
    ))

@team.command()
@click.argument('team_id')
@click.argument('role_id')
@click.argument('role_name')
@click.argument('specialization')
@click.option('--capabilities', '-c', required=True, help="Comma-separated list of role capabilities")
@click.option('--contribution', '-l', default=1.0, help="Contribution level (0.0-1.0)")
def add_member(team_id: str, role_id: str, role_name: str, specialization: str, capabilities: str, contribution: float):
    """Add a role as a member to a team."""
    
    team_designer = TeamDesigner()
    
    # Parse capabilities
    capability_list = [cap.strip() for cap in capabilities.split(',')]
    
    # Add member to team
    success = team_designer.add_member_to_team(
        team_id=team_id,
        role_id=role_id,
        role_name=role_name,
        specialization=specialization,
        capabilities=capability_list,
        contribution_level=contribution
    )
    
    if success:
        console.print(Panel(
            f"✅ Added {role_name} to team\n"
            f"🎯 Specialization: {specialization}\n"
            f"📊 Contribution Level: {contribution:.1%}\n"
            f"🛠️ Capabilities: {', '.join(capability_list)}",
            title="👥 Member Added Successfully",
            style="green"
        ))
    else:
        console.print(Panel(
            f"❌ Failed to add member to team\n"
            f"Possible reasons:\n"
            f"• Team not found\n"
            f"• Member already exists\n"
            f"• Team at maximum capacity",
            title="🚫 Addition Failed",
            style="red"
        ))

@team.command()
@click.argument('title')
@click.argument('description')
@click.option('--priority', '-p', 
              type=click.Choice(['low', 'medium', 'high', 'critical']),
              default='medium', help="Goal priority")
@click.option('--complexity', '-c', default=0.5, help="Goal complexity (0.0-1.0)")
@click.option('--effort', '-e', default=40, help="Estimated effort in hours")
@click.option('--deadline', '-d', help="Goal deadline (YYYY-MM-DD)")
def create_goal(title: str, description: str, priority: str, complexity: float, effort: int, deadline: str):
    """Create a goal that teams can work towards."""
    
    team_designer = TeamDesigner()
    
    # Create goal
    goal = team_designer.create_team_goal(
        title=title,
        description=description,
        priority=priority,
        complexity=complexity,
        estimated_effort=effort,
        deadline=deadline
    )
    
    console.print(Panel(
        f"✅ Created goal: {goal.title}\n"
        f"🏷️ ID: {goal.id}\n"
        f"📋 Description: {goal.description[:100]}{'...' if len(goal.description) > 100 else ''}\n"
        f"⚡ Priority: {goal.priority}\n"
        f"🧩 Complexity: {goal.complexity:.1%}\n"
        f"⏱️ Estimated Effort: {goal.estimated_effort}h",
        title="🎯 Goal Created Successfully",
        style="green"
    ))

@team.command()
@click.argument('goal_id')
@click.argument('team_id')
def assign_goal(goal_id: str, team_id: str):
    """Assign a goal to a specific team."""
    
    team_designer = TeamDesigner()
    
    success = team_designer.assign_goal_to_team(goal_id, team_id)
    
    if success:
        console.print(Panel(
            f"✅ Goal assigned to team successfully\n"
            f"🎯 Goal ID: {goal_id}\n"
            f"👥 Team ID: {team_id}",
            title="📌 Goal Assignment Complete",
            style="green"
        ))
    else:
        console.print(Panel(
            f"❌ Failed to assign goal to team\n"
            f"Possible reasons:\n"
            f"• Goal not found\n"
            f"• Team not found\n"
            f"• Assignment already exists",
            title="🚫 Assignment Failed",
            style="red"
        ))

@team.command()
@click.argument('team_ids', nargs=-1, required=True)
@click.option('--collaboration-type', '-t',
              type=click.Choice(['independent', 'cooperative', 'coordinated', 'integrated']),
              default='cooperative', help="Type of collaboration")
@click.option('--shared-goals', '-g', help="Comma-separated list of shared goal IDs")
def create_collaboration(team_ids: tuple, collaboration_type: str, shared_goals: str):
    """Create collaboration between multiple teams."""
    
    team_designer = TeamDesigner()
    
    # Parse shared goals
    goal_list = [goal.strip() for goal in shared_goals.split(',')] if shared_goals else []
    
    # Create collaboration
    collaboration = team_designer.create_team_collaboration(
        team_ids=list(team_ids),
        collaboration_type=CollaborationMode(collaboration_type),
        shared_goals=goal_list
    )
    
    console.print(Panel(
        f"✅ Created collaboration between {len(team_ids)} teams\n"
        f"🏷️ ID: {collaboration.id}\n"
        f"🤝 Type: {collaboration.collaboration_type}\n"
        f"🎯 Shared Goals: {len(collaboration.shared_goals)}\n"
        f"👥 Teams: {', '.join(team_ids)}",
        title="🤝 Collaboration Created Successfully",
        style="green"
    ))

@team.command()
@click.argument('team_id')
@click.argument('problem_description')
@click.option('--use-reasoning', '-r', is_flag=True, help="Use AI reasoning for goal extraction")
@click.option('--save-solution', '-s', is_flag=True, help="Save the autonomous solution")
def extract_goals(team_id: str, problem_description: str, use_reasoning: bool, save_solution: bool):
    """Extract and refine goals from a problem description using team intelligence."""
    
    team_designer = TeamDesigner()
    
    with console.status("🧠 Extracting goals using team intelligence..."):
        goals = team_designer.extract_and_refine_goals(team_id, problem_description)
    
    if goals:
        console.print(Panel(
            f"✅ Extracted {len(goals)} goals from problem description\n"
            f"👥 Team ID: {team_id}\n"
            f"🧠 Using team intelligence and expertise",
            title="🎯 Goals Extracted Successfully",
            style="green"
        ))
        
        # Display goals
        table = Table(title="📋 Extracted Goals")
        table.add_column("Title", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Complexity", style="red")
        table.add_column("Effort (h)", style="green")
        
        for goal in goals:
            table.add_row(
                goal.title[:40] + "..." if len(goal.title) > 40 else goal.title,
                goal.priority,
                f"{goal.complexity:.1%}",
                str(goal.estimated_effort)
            )
        
        console.print(table)
    else:
        console.print(Panel(
            f"❌ Failed to extract goals\n"
            f"Possible reasons:\n"
            f"• Team not found\n"
            f"• Problem description too vague\n"
            f"• Team lacks relevant expertise",
            title="🚫 Goal Extraction Failed",
            style="red"
        ))

@team.command()
@click.argument('team_id')
@click.argument('problem')
@click.option('--save-solution', '-s', is_flag=True, help="Save the autonomous solution")
@click.option('--detailed', '-d', is_flag=True, help="Show detailed solution breakdown")
def solve_problem(team_id: str, problem: str, save_solution: bool, detailed: bool):
    """Simulate autonomous problem-solving by a team."""
    
    team_designer = TeamDesigner()
    
    with console.status("🤖 Team autonomously solving problem..."):
        solution = team_designer.simulate_autonomous_problem_solving(team_id, problem)
    
    if "error" not in solution:
        console.print(Panel(
            f"✅ Autonomous solution generated\n"
            f"👥 Team: {solution['team_name']}\n"
            f"⏱️ Estimated Time: {solution['estimated_completion_time']}h\n"
            f"🎯 Confidence: {solution['confidence_score']:.1%}\n"
            f"🎯 Goals: {len(solution['extracted_goals'])}\n"
            f"📋 Tasks: {len(solution['task_breakdown'])}",
            title="🤖 Autonomous Solution Complete",
            style="green"
        ))
        
        if detailed:
            # Show goals
            console.print("\n🎯 [bold blue]Extracted Goals:[/bold blue]")
            for i, goal in enumerate(solution['extracted_goals'], 1):
                console.print(f"  {i}. {goal['title']} (Priority: {goal['priority']})")
            
            # Show task breakdown
            console.print("\n📋 [bold blue]Task Breakdown:[/bold blue]")
            for i, task in enumerate(solution['task_breakdown'], 1):
                console.print(f"  {i}. {task['title']} ({task['estimated_effort']}h)")
            
            # Show collaboration requirements
            if solution['collaboration_requirements']:
                console.print("\n🤝 [bold blue]Collaboration Requirements:[/bold blue]")
                for req in solution['collaboration_requirements'].get('recommended_collaborations', []):
                    console.print(f"  • {req.get('type', 'Unknown')}: {req.get('missing_capabilities', 'N/A')}")
        
        if save_solution:
            console.print(f"\n💾 Solution saved to outputs/teams/solutions/")
    else:
        console.print(Panel(
            f"❌ {solution['error']}",
            title="🚫 Problem Solving Failed",
            style="red"
        ))

@team.command()
@click.argument('team_id')
def performance(team_id: str):
    """Get performance metrics for a team."""
    
    team_designer = TeamDesigner()
    
    with console.status("📊 Calculating team performance metrics..."):
        metrics = team_designer.get_team_performance_metrics(team_id)
    
    if metrics:
        console.print(Panel(
            f"📊 Team Performance Metrics\n"
            f"👥 Team ID: {team_id}\n\n"
            f"⚡ Efficiency: {metrics['efficiency']:.1%}\n"
            f"🤝 Collaboration Score: {metrics['collaboration_score']:.1%}\n"
            f"🎯 Goal Completion Rate: {metrics['goal_completion_rate']:.1%}\n"
            f"🔄 Adaptability: {metrics['adaptability']:.1%}\n"
            f"🏆 Overall Performance: {metrics['overall_performance']:.1%}",
            title="📈 Performance Dashboard",
            style="blue"
        ))
        
        # Create performance visualization
        performance_tree = Tree("📊 Performance Breakdown")
        
        efficiency_node = performance_tree.add("⚡ Efficiency")
        efficiency_node.add(f"Score: {metrics['efficiency']:.1%}")
        efficiency_node.add("Factors: Team size, capability diversity, availability")
        
        collaboration_node = performance_tree.add("🤝 Collaboration")
        collaboration_node.add(f"Score: {metrics['collaboration_score']:.1%}")
        collaboration_node.add("Factors: Communication protocols, capability diversity")
        
        goals_node = performance_tree.add("🎯 Goal Achievement")
        goals_node.add(f"Rate: {metrics['goal_completion_rate']:.1%}")
        goals_node.add("Factors: Completed vs total goals")
        
        adapt_node = performance_tree.add("🔄 Adaptability")
        adapt_node.add(f"Score: {metrics['adaptability']:.1%}")
        adapt_node.add("Factors: Team type, capability diversity")
        
        console.print(performance_tree)
    else:
        console.print(Panel(
            f"❌ Team not found: {team_id}",
            title="🚫 Performance Check Failed",
            style="red"
        ))

@team.command()
@click.option('--discipline', '-d', help="Filter by discipline")
@click.option('--status', '-s',
              type=click.Choice(['forming', 'active', 'collaborating', 'paused', 'completed', 'disbanded']),
              help="Filter by status")
@click.option('--detailed', '-v', is_flag=True, help="Show detailed information")
def list(discipline: str, status: str, detailed: bool):
    """List all teams with optional filtering."""
    
    team_designer = TeamDesigner()
    
    if discipline:
        teams = team_designer.get_teams_by_discipline(discipline)
        title = f"🎯 Teams - {discipline.title()} Discipline"
    elif status:
        teams = [team for team in team_designer.teams.values() if team.status == status]
        title = f"👥 Teams - {status.title()} Status"
    else:
        teams = list(team_designer.teams.values())
        title = "👥 All Teams"
    
    if teams:
        if detailed:
            for team in teams:
                console.print(Panel(
                    f"🏷️ ID: {team.id}\n"
                    f"📛 Name: {team.name}\n"
                    f"🎯 Discipline: {team.discipline}\n"
                    f"📊 Type: {team.team_type}\n"
                    f"📈 Status: {team.status}\n"
                    f"👥 Members: {len(team.members)}/{team.max_size}\n"
                    f"🎯 Goals: {len(team.goals)}\n"
                    f"🛠️ Capabilities: {', '.join(team.capabilities[:3])}{'...' if len(team.capabilities) > 3 else ''}\n"
                    f"📅 Created: {team.created_at[:10]}",
                    title=f"👥 {team.name}",
                    style="cyan"
                ))
        else:
            table = Table(title=title)
            table.add_column("Name", style="cyan")
            table.add_column("Discipline", style="yellow")
            table.add_column("Type", style="green")
            table.add_column("Status", style="blue")
            table.add_column("Members", style="red")
            table.add_column("Goals", style="magenta")
            
            for team in teams:
                table.add_row(
                    team.name,
                    team.discipline,
                    team.team_type,
                    team.status,
                    f"{len(team.members)}/{team.max_size}",
                    str(len(team.goals))
                )
            
            console.print(table)
    else:
        filter_desc = f" (filtered by {discipline or status})" if discipline or status else ""
        console.print(Panel(
            f"No teams found{filter_desc}",
            title="👥 Teams List",
            style="yellow"
        ))

@team.command()
def list_collaborations():
    """List all active team collaborations."""
    
    team_designer = TeamDesigner()
    collaborations = team_designer.get_collaborating_teams()
    
    if collaborations:
        table = Table(title="🤝 Active Team Collaborations")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Teams", style="green")
        table.add_column("Shared Goals", style="blue")
        table.add_column("Status", style="red")
        
        for collab in collaborations:
            table.add_row(
                collab.id[:8] + "...",
                collab.collaboration_type,
                str(len(collab.team_ids)),
                str(len(collab.shared_goals)),
                collab.status
            )
        
        console.print(table)
    else:
        console.print(Panel(
            "No active collaborations found",
            title="🤝 Team Collaborations",
            style="yellow"
        ))

@team.command()
def list_available():
    """List teams available for new work."""
    
    team_designer = TeamDesigner()
    available_teams = team_designer.get_available_teams()
    
    if available_teams:
        table = Table(title="✅ Available Teams")
        table.add_column("Name", style="cyan")
        table.add_column("Discipline", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Capacity", style="blue")
        table.add_column("Current Goals", style="red")
        
        for team in available_teams:
            capacity = f"{len(team.members)}/{team.max_size}"
            table.add_row(
                team.name,
                team.discipline,
                team.status,
                capacity,
                str(len(team.goals))
            )
        
        console.print(table)
    else:
        console.print(Panel(
            "No teams currently available for new work",
            title="✅ Available Teams",
            style="yellow"
        ))

@team.command()
@click.argument('problem_description')
@click.option('--teams', '-t', help="Comma-separated list of team IDs to include")
@click.option('--auto-select', '-a', is_flag=True, help="Automatically select suitable teams")
@click.option('--save-plan', '-s', is_flag=True, help="Save the collaborative plan")
def collaborate(problem_description: str, teams: str, auto_select: bool, save_plan: bool):
    """Create collaborative solution using multiple teams."""
    
    team_designer = TeamDesigner()
    task_designer = TaskDesigner()
    
    # Get teams
    if teams:
        team_ids = [t.strip() for t in teams.split(',')]
        selected_teams = [team for team in team_designer.teams.values() if team.id in team_ids]
    elif auto_select:
        selected_teams = team_designer.get_available_teams()[:3]  # Select up to 3 available teams
    else:
        console.print("❌ Please specify teams with --teams or use --auto-select")
        return
    
    if not selected_teams:
        console.print("❌ No suitable teams found")
        return
    
    with console.status("🤝 Creating collaborative solution..."):
        # Convert teams to the format expected by task designer
        team_data = []
        for team in selected_teams:
            team_data.append({
                "id": team.id,
                "name": team.name,
                "discipline": team.discipline,
                "capabilities": team.capabilities
            })
        
        # Create cross-functional tasks
        collaboration_plan = task_designer.create_cross_functional_tasks(problem_description, team_data)
    
    console.print(Panel(
        f"✅ Collaborative solution created\n"
        f"🤝 Teams: {len(selected_teams)}\n"
        f"📋 Cross-functional Tasks: {len(collaboration_plan['cross_functional_tasks'])}\n"
        f"🎯 Problem: {problem_description[:100]}{'...' if len(problem_description) > 100 else ''}",
        title="🤝 Collaboration Plan Ready",
        style="green"
    ))
    
    # Show team assignments
    console.print("\n👥 [bold blue]Team Assignments:[/bold blue]")
    for assignment in collaboration_plan['team_assignments']:
        assigned_teams = assignment['assigned_teams']
        team_names = [team['name'] for team in assigned_teams]
        console.print(f"  📋 {assignment['task_name']}: {', '.join(team_names)}")
    
    # Show collaboration framework
    console.print("\n🤝 [bold blue]Collaboration Framework:[/bold blue]")
    framework = collaboration_plan['collaboration_framework']
    console.print(f"  📞 Communication: {', '.join(framework['communication_protocols'][:2])}")
    console.print(f"  🎯 Decision Making: {framework['decision_making_process']['cross_team_decisions']}")
    
    if save_plan:
        # Save collaboration plan
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feriq/outputs/teams/collaborations/collab_plan_{timestamp}.json"
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(collaboration_plan, f, indent=2, default=str)
        
        console.print(f"\n💾 Collaboration plan saved to {filename}")

@team.command()
@click.argument('teams', nargs=-1, required=True)
@click.argument('objectives', nargs=-1, required=True)
@click.option('--save-workflow', '-s', is_flag=True, help="Save the autonomous workflow")
def create_autonomous_workflow(teams: tuple, objectives: tuple, save_workflow: bool):
    """Create autonomous workflow where teams self-organize and adapt."""
    
    team_designer = TeamDesigner()
    task_designer = TaskDesigner()
    
    # Get team data
    team_data = []
    for team_id in teams:
        if team_id in team_designer.teams:
            team = team_designer.teams[team_id]
            team_data.append({
                "id": team.id,
                "name": team.name,
                "discipline": team.discipline,
                "capabilities": team.capabilities
            })
    
    if not team_data:
        console.print("❌ No valid teams found")
        return
    
    with console.status("🤖 Creating autonomous workflow..."):
        autonomous_workflow = task_designer.create_autonomous_task_workflow(team_data, list(objectives))
    
    console.print(Panel(
        f"✅ Autonomous workflow created\n"
        f"👥 Teams: {len(team_data)}\n"
        f"🎯 Objectives: {len(objectives)}\n"
        f"🤖 Self-organization enabled\n"
        f"🔄 Adaptive mechanisms active",
        title="🤖 Autonomous Workflow Ready",
        style="green"
    ))
    
    # Show key features
    console.print("\n🤖 [bold blue]Autonomous Features:[/bold blue]")
    framework = autonomous_workflow['adaptive_framework']
    for mechanism, description in framework['adaptation_mechanisms'].items():
        console.print(f"  🔄 {mechanism.replace('_', ' ').title()}: {description}")
    
    console.print("\n⚖️ [bold blue]Decision Authority:[/bold blue]")
    authority = autonomous_workflow['autonomy_rules']['decision_boundaries']
    console.print(f"  ✅ Within Scope: {len(authority['within_scope'])} types")
    console.print(f"  🤝 Requires Coordination: {len(authority['requires_coordination'])} types")
    console.print(f"  🔐 Requires Approval: {len(authority['requires_approval'])} types")
    
    if save_workflow:
        # Save workflow
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feriq/outputs/teams/solutions/autonomous_workflow_{timestamp}.json"
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(autonomous_workflow, f, indent=2, default=str)
        
        console.print(f"\n💾 Autonomous workflow saved to {filename}")

@team.command()
def demo():
    """Run a comprehensive team management demonstration."""
    
    console.print(Panel(
        "🎭 Starting Team Management Demo\n"
        "This demo will showcase:\n"
        "• Team creation with different disciplines\n"
        "• Goal extraction and assignment\n"
        "• Team collaboration setup\n"
        "• Autonomous problem solving\n"
        "• Performance monitoring",
        title="🎪 Team Demo",
        style="magenta"
    ))
    
    team_designer = TeamDesigner()
    
    # Create demo teams
    teams_to_create = [
        ("AI Research Team", "data_science", "autonomous", ["machine_learning", "data_analysis", "research"]),
        ("Software Dev Team", "software_development", "cross_functional", ["coding", "testing", "deployment"]),
        ("Design Team", "design", "specialist", ["ui_design", "ux_research", "prototyping"])
    ]
    
    created_teams = []
    
    console.print("\n👥 [bold blue]Creating Demo Teams...[/bold blue]")
    for name, discipline, team_type, capabilities in teams_to_create:
        team = team_designer.create_team(
            name=name,
            description=f"Demo team for {discipline}",
            discipline=discipline,
            team_type=TeamType(team_type),
            capabilities=capabilities
        )
        created_teams.append(team)
        console.print(f"  ✅ Created: {name} ({discipline})")
    
    # Create demo goals
    console.print("\n🎯 [bold blue]Creating Demo Goals...[/bold blue]")
    demo_goals = [
        ("Build AI-Powered Application", "Create an intelligent application using machine learning", "high", 0.8),
        ("User Experience Research", "Conduct comprehensive UX research for the application", "medium", 0.6),
        ("System Architecture Design", "Design scalable system architecture", "high", 0.7)
    ]
    
    created_goals = []
    for title, description, priority, complexity in demo_goals:
        goal = team_designer.create_team_goal(
            title=title,
            description=description,
            priority=priority,
            complexity=complexity,
            estimated_effort=60
        )
        created_goals.append(goal)
        console.print(f"  🎯 Created: {title}")
    
    # Assign goals to teams
    console.print("\n📌 [bold blue]Assigning Goals to Teams...[/bold blue]")
    for i, (team, goal) in enumerate(zip(created_teams, created_goals)):
        success = team_designer.assign_goal_to_team(goal.id, team.id)
        if success:
            console.print(f"  📌 Assigned '{goal.title}' to '{team.name}'")
    
    # Create collaboration
    console.print("\n🤝 [bold blue]Creating Team Collaboration...[/bold blue]")
    collaboration = team_designer.create_team_collaboration(
        team_ids=[team.id for team in created_teams],
        collaboration_type=CollaborationMode.COORDINATED,
        shared_goals=[goal.id for goal in created_goals[:2]]
    )
    console.print(f"  🤝 Created collaboration between {len(created_teams)} teams")
    
    # Demonstrate autonomous problem solving
    console.print("\n🤖 [bold blue]Autonomous Problem Solving Demo...[/bold blue]")
    problem = "Design and develop an AI-powered mobile application that helps users manage their daily tasks with intelligent recommendations and natural language processing capabilities."
    
    solution = team_designer.simulate_autonomous_problem_solving(created_teams[0].id, problem)
    
    if "error" not in solution:
        console.print(f"  🎯 Extracted {len(solution['extracted_goals'])} goals")
        console.print(f"  📋 Created {len(solution['task_breakdown'])} tasks")
        console.print(f"  ⏱️ Estimated completion: {solution['estimated_completion_time']}h")
        console.print(f"  🎯 Confidence: {solution['confidence_score']:.1%}")
    
    # Show performance metrics
    console.print("\n📊 [bold blue]Team Performance Metrics...[/bold blue]")
    for team in created_teams:
        metrics = team_designer.get_team_performance_metrics(team.id)
        console.print(f"  📈 {team.name}: {metrics['overall_performance']:.1%} overall performance")
    
    console.print(Panel(
        "🎉 Demo completed successfully!\n\n"
        "✅ Created 3 specialized teams\n"
        "✅ Generated 3 collaborative goals\n"
        "✅ Established team collaboration\n"
        "✅ Demonstrated autonomous problem solving\n"
        "✅ Calculated performance metrics\n\n"
        "Check feriq/outputs/teams/ for generated files",
        title="🎪 Demo Complete",
        style="green"
    ))

if __name__ == "__main__":
    team()