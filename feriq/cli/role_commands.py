"""
CLI Commands for Role Management - Feriq Framework

Comprehensive command-line interface for creating, managing, and assigning roles
to teams and agents within the Feriq collaborative AI framework.
"""

import click
import json
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from pathlib import Path

from ..components.role_designer import DynamicRoleDesigner, RoleTemplate
from ..components.team_designer import TeamDesigner
from ..core.role import Role, RoleType, RoleCapability
from .utils import print_success, print_error, print_info, print_warning

console = Console()

@click.group()
def role():
    """Role management commands for creating and assigning roles."""
    pass

@role.command()
@click.argument('name')
@click.argument('role_type', type=click.Choice([
    'researcher', 'analyst', 'planner', 'executor', 
    'coordinator', 'reviewer', 'specialist', 'generalist'
]))
@click.option('--description', '-d', help="Role description")
@click.option('--capabilities', '-c', help="Comma-separated list of capabilities with optional levels (e.g., 'analysis:0.8,research:0.9')")
@click.option('--responsibilities', '-r', help="Comma-separated list of responsibilities")
@click.option('--constraints', '--cons', help="Comma-separated list of constraints")
@click.option('--tags', '-t', help="Comma-separated list of tags")
@click.option('--template', help="Base role on existing template")
@click.option('--save-template', is_flag=True, help="Save this role as a template")
def create(name: str, role_type: str, description: str, capabilities: str, 
          responsibilities: str, constraints: str, tags: str, template: str, save_template: bool):
    """Create a new role with specified capabilities and responsibilities."""
    
    role_designer = DynamicRoleDesigner()
    
    try:
        # If using template, start from template
        if template:
            role = role_designer.create_role_from_template(template, {
                "name": name,
                "description": description
            })
            if not role:
                print_error(f"Template '{template}' not found. Available templates: {list(role_designer.role_templates.keys())}")
                return
        else:
            # Create role from scratch
            role_capabilities = []
            
            # Parse capabilities
            if capabilities:
                for cap_def in capabilities.split(','):
                    if ':' in cap_def:
                        cap_name, level = cap_def.strip().split(':')
                        level = float(level)
                    else:
                        cap_name = cap_def.strip()
                        level = 0.7  # Default level
                    
                    capability = RoleCapability(
                        name=cap_name,
                        description=f"Capability: {cap_name}",
                        proficiency_level=level
                    )
                    role_capabilities.append(capability)
            
            # Parse other attributes
            role_responsibilities = [r.strip() for r in responsibilities.split(',')] if responsibilities else []
            role_constraints = [c.strip() for c in constraints.split(',')] if constraints else []
            role_tags = [t.strip() for t in tags.split(',')] if tags else []
            
            # Create the role
            role = Role(
                name=name,
                role_type=RoleType(role_type.lower()),
                description=description or f"Custom {role_type} role",
                capabilities=role_capabilities,
                responsibilities=role_responsibilities,
                constraints=role_constraints,
                created_at=datetime.now().isoformat(),
                version="1.0",
                tags=role_tags
            )
        
        # Save role to outputs
        output_dir = Path("outputs/roles")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        role_file = output_dir / f"role_{role.name.lower().replace(' ', '_')}.json"
        
        role_data = {
            "name": role.name,
            "role_type": role.role_type.value if hasattr(role.role_type, 'value') else str(role.role_type),
            "description": role.description,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "proficiency_level": cap.proficiency_level
                } for cap in role.capabilities
            ],
            "responsibilities": role.responsibilities,
            "constraints": role.constraints,
            "created_at": role.created_at,
            "version": role.version,
            "tags": role.tags
        }
        
        with open(role_file, 'w') as f:
            json.dump(role_data, f, indent=2)
        
        # Save as template if requested
        if save_template:
            template_data = {
                "name": role.name,
                "role_type": role.role_type.value if hasattr(role.role_type, 'value') else str(role.role_type),
                "base_capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "proficiency_level": cap.proficiency_level
                    } for cap in role.capabilities
                ],
                "responsibilities_template": role.responsibilities,
                "constraints_template": role.constraints,
                "adaptable_attributes": ["description", "capabilities", "responsibilities"]
            }
            
            template_file = output_dir / f"template_{role.name.lower().replace(' ', '_')}.yaml"
            with open(template_file, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False)
            
            print_info(f"Saved as template: {template_file}")
        
        # Display success message
        console.print(Panel(
            f"✅ Created role: {role.name}\n"
            f"🎭 Type: {role.role_type.value if hasattr(role.role_type, 'value') else str(role.role_type)}\n"
            f"📋 Description: {role.description}\n"
            f"🛠️ Capabilities: {len(role.capabilities)}\n"
            f"📝 Responsibilities: {len(role.responsibilities)}\n"
            f"🔒 Constraints: {len(role.constraints)}\n"
            f"📁 Saved to: {role_file}",
            title="🎉 Role Created Successfully",
            style="green"
        ))
        
    except Exception as e:
        print_error(f"Failed to create role: {str(e)}")

def _load_existing_teams(team_designer):
    """Load existing teams from files into the team designer."""
    import os
    from dataclasses import asdict
    
    teams_dir = Path("feriq/outputs/teams")
    if not teams_dir.exists():
        return
    
    for team_file in teams_dir.glob("team_*.json"):
        try:
            with open(team_file, 'r') as f:
                team_data = json.load(f)
            
            # Convert to Team object and add to designer
            from ..components.team_designer import Team, TeamMember
            
            # Convert members
            members = []
            for member_data in team_data.get('members', []):
                member = TeamMember(
                    role_id=member_data['role_id'],
                    role_name=member_data['role_name'],
                    specialization=member_data['specialization'],
                    capabilities=member_data['capabilities'],
                    contribution_level=member_data['contribution_level']
                )
                members.append(member)
            
            team = Team(
                id=team_data['id'],
                name=team_data['name'],
                description=team_data['description'],
                team_type=team_data['team_type'],
                discipline=team_data['discipline'],
                max_size=team_data['max_size'],
                min_size=team_data['min_size'],
                capabilities=team_data['capabilities'],
                status=team_data['status'],
                members=members,
                goals=team_data.get('goals', []),
                created_at=team_data['created_at'],
                updated_at=team_data.get('updated_at', team_data['created_at'])
            )
            
            team_designer.teams[team.id] = team
            
        except Exception as e:
            print(f"Warning: Could not load team from {team_file}: {e}")

@role.command()
@click.argument('role_file')
@click.argument('team_id')
@click.option('--specialization', '-s', help="Role specialization within the team")
@click.option('--contribution', '-c', default=1.0, help="Contribution level (0.0-1.0)")
def assign(role_file: str, team_id: str, specialization: str, contribution: float):
    """Assign a role to a team."""
    
    team_designer = TeamDesigner()
    
    # Load existing teams
    _load_existing_teams(team_designer)
    
    try:
        # Load role from file
        role_path = Path(role_file)
        if not role_path.exists():
            # Try looking in outputs/roles directory
            role_path = Path("outputs/roles") / role_file
            if not role_path.exists():
                role_path = Path("outputs/roles") / f"role_{role_file.lower().replace(' ', '_')}.json"
        
        if not role_path.exists():
            print_error(f"Role file not found: {role_file}")
            print_info("Use 'feriq list roles' to see available roles")
            return
        
        with open(role_path, 'r') as f:
            role_data = json.load(f)
        
        # Extract role information
        role_name = role_data['name']
        capabilities = [cap['name'] for cap in role_data.get('capabilities', [])]
        
        # Add member to team
        success = team_designer.add_member_to_team(
            team_id=team_id,
            role_id=role_data.get('name', '').lower().replace(' ', '_'),
            role_name=role_name,
            specialization=specialization or role_data.get('description', ''),
            capabilities=capabilities,
            contribution_level=contribution
        )
        
        if success:
            console.print(Panel(
                f"✅ Assigned role to team successfully\n"
                f"🎭 Role: {role_name}\n"
                f"👥 Team ID: {team_id}\n"
                f"🎯 Specialization: {specialization or 'General'}\n"
                f"📊 Contribution: {contribution:.1%}\n"
                f"🛠️ Capabilities: {', '.join(capabilities[:3])}{'...' if len(capabilities) > 3 else ''}",
                title="👥 Role Assignment Successful",
                style="green"
            ))
        else:
            print_error("Failed to assign role to team. Check team ID and try again.")
            
    except Exception as e:
        print_error(f"Failed to assign role: {str(e)}")

@role.command('list')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table')
@click.option('--filter', help='Filter roles by type, capability, or tag')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
def list_roles(format: str, filter: str, detailed: bool):
    """List all available roles."""
    
    try:
        roles_dir = Path("outputs/roles")
        
        if not roles_dir.exists():
            print_info("No roles directory found. Create roles with 'feriq role create'")
            return
        
        # Find all role files
        role_files = list(roles_dir.glob("role_*.json"))
        
        if not role_files:
            print_info("No roles found. Create roles with 'feriq role create'")
            return
        
        roles_data = []
        
        for role_file in role_files:
            try:
                with open(role_file, 'r') as f:
                    role_data = json.load(f)
                    role_data['file_path'] = str(role_file)
                    roles_data.append(role_data)
            except Exception as e:
                print_warning(f"Could not load {role_file}: {e}")
        
        # Apply filter if specified
        if filter:
            filtered_roles = []
            filter_lower = filter.lower()
            
            for role in roles_data:
                # Check role type
                if filter_lower in role.get('role_type', '').lower():
                    filtered_roles.append(role)
                    continue
                    
                # Check capabilities
                capabilities = [cap.get('name', '') for cap in role.get('capabilities', [])]
                if any(filter_lower in cap.lower() for cap in capabilities):
                    filtered_roles.append(role)
                    continue
                    
                # Check tags
                tags = role.get('tags', [])
                if any(filter_lower in tag.lower() for tag in tags):
                    filtered_roles.append(role)
                    continue
            
            roles_data = filtered_roles
        
        if not roles_data:
            print_info(f"No roles found matching filter: {filter}")
            return
        
        # Display results
        if format == 'json':
            console.print(json.dumps(roles_data, indent=2))
        elif format == 'yaml':
            console.print(yaml.dump(roles_data, default_flow_style=False))
        else:
            # Table format
            if detailed:
                _display_detailed_roles_table(roles_data)
            else:
                _display_roles_table(roles_data)
                
    except Exception as e:
        print_error(f"Failed to list roles: {str(e)}")
        import traceback
        if detailed:  # Show full traceback in detailed mode
            print_error(traceback.format_exc())

def _display_roles_table(roles_data: List[Dict]):
    """Display roles in a simple table format."""
    table = Table(title="🎭 Available Roles")
    
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Capabilities", style="green")
    table.add_column("Responsibilities", style="yellow")
    table.add_column("Created", style="blue")
    
    for role in roles_data:
        capabilities = [cap.get('name', '') for cap in role.get('capabilities', [])]
        cap_display = ', '.join(capabilities[:2])
        if len(capabilities) > 2:
            cap_display += f" (+{len(capabilities)-2} more)"
        
        responsibilities = role.get('responsibilities', [])
        resp_display = ', '.join(responsibilities[:1])
        if len(responsibilities) > 1:
            resp_display += f" (+{len(responsibilities)-1} more)"
        
        created_date = role.get('created_at', '')[:10] if role.get('created_at') else 'Unknown'
        
        table.add_row(
            role.get('name', 'Unknown'),
            role.get('role_type', 'Unknown'),
            cap_display,
            resp_display,
            created_date
        )
    
    console.print(table)

def _display_detailed_roles_table(roles_data: List[Dict]):
    """Display roles with detailed information."""
    for i, role in enumerate(roles_data, 1):
        console.print(f"\n[bold cyan]Role {i}: {role.get('name', 'Unknown')}[/bold cyan]")
        
        # Basic info
        basic_table = Table(show_header=False, box=None)
        basic_table.add_column("Field", style="yellow", width=15)
        basic_table.add_column("Value", style="white")
        
        basic_table.add_row("Type", role.get('role_type', 'Unknown'))
        basic_table.add_row("Version", role.get('version', 'Unknown'))
        basic_table.add_row("Created", role.get('created_at', 'Unknown')[:19] if role.get('created_at') else 'Unknown')
        basic_table.add_row("File", Path(role.get('file_path', '')).name)
        
        console.print(basic_table)
        
        # Description
        if role.get('description'):
            console.print(f"[bold]Description:[/bold] {role['description']}")
        
        # Capabilities
        capabilities = role.get('capabilities', [])
        if capabilities:
            console.print("\n[bold]Capabilities:[/bold]")
            for cap in capabilities:
                level = cap.get('proficiency_level', 0.0)
                level_bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
                console.print(f"  • {cap.get('name', 'Unknown')} [{level_bar}] {level:.1%}")
        
        # Responsibilities
        responsibilities = role.get('responsibilities', [])
        if responsibilities:
            console.print("\n[bold]Responsibilities:[/bold]")
            for resp in responsibilities:
                console.print(f"  • {resp}")
        
        # Constraints
        constraints = role.get('constraints', [])
        if constraints:
            console.print("\n[bold]Constraints:[/bold]")
            for constraint in constraints:
                console.print(f"  • {constraint}")
        
        # Tags
        tags = role.get('tags', [])
        if tags:
            console.print(f"\n[bold]Tags:[/bold] {', '.join(tags)}")
        
        if i < len(roles_data):
            console.print("\n" + "─" * 50)

@role.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table')
def templates(format: str):
    """List available role templates."""
    
    role_designer = DynamicRoleDesigner()
    
    templates = role_designer.role_templates
    
    if not templates:
        print_info("No role templates available.")
        return
    
    # Also check for saved templates
    templates_dir = Path("outputs/roles")
    if templates_dir.exists():
        template_files = list(templates_dir.glob("template_*.yaml"))
        for template_file in template_files:
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                    template_name = template_data.get('name', template_file.stem)
                    if template_name not in templates:
                        # Add to templates list for display
                        pass
            except Exception:
                pass
    
    if format == 'json':
        templates_data = {}
        for name, template in templates.items():
            templates_data[name] = {
                "name": template.name,
                "role_type": template.role_type.value,
                "base_capabilities": [
                    {
                        "name": cap.name,
                        "proficiency_level": cap.proficiency_level
                    } for cap in template.base_capabilities
                ],
                "responsibilities": template.responsibilities_template,
                "constraints": template.constraints_template
            }
        console.print(json.dumps(templates_data, indent=2))
    
    elif format == 'yaml':
        templates_data = {}
        for name, template in templates.items():
            templates_data[name] = {
                "name": template.name,
                "role_type": template.role_type.value,
                "base_capabilities": [cap.name for cap in template.base_capabilities],
                "responsibilities": template.responsibilities_template,
                "constraints": template.constraints_template
            }
        console.print(yaml.dump(templates_data, default_flow_style=False))
    
    else:
        # Table format
        table = Table(title="🎨 Available Role Templates")
        
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Capabilities", style="green")
        table.add_column("Responsibilities", style="yellow")
        
        for name, template in templates.items():
            capabilities = [cap.name for cap in template.base_capabilities]
            cap_display = ', '.join(capabilities[:2])
            if len(capabilities) > 2:
                cap_display += f" (+{len(capabilities)-2} more)"
            
            responsibilities = template.responsibilities_template
            resp_display = ', '.join(responsibilities[:1])
            if len(responsibilities) > 1:
                resp_display += f" (+{len(responsibilities)-1} more)"
            
            table.add_row(
                template.name,
                template.role_type.value,
                cap_display,
                resp_display
            )
        
        console.print(table)

@role.command()
@click.argument('role_name')
def show(role_name: str):
    """Show detailed information about a specific role."""
    
    # Try to find the role file
    role_path = Path("outputs/roles") / f"role_{role_name.lower().replace(' ', '_')}.json"
    
    if not role_path.exists():
        # Try exact filename
        role_path = Path("outputs/roles") / role_name
        if not role_path.exists():
            print_error(f"Role not found: {role_name}")
            print_info("Use 'feriq role list' to see available roles")
            return
    
    try:
        with open(role_path, 'r') as f:
            role_data = json.load(f)
        
        # Display detailed role information
        console.print(Panel(
            f"[bold cyan]{role_data.get('name', 'Unknown')}[/bold cyan]\n\n"
            f"[bold]Type:[/bold] {role_data.get('role_type', 'Unknown')}\n"
            f"[bold]Version:[/bold] {role_data.get('version', 'Unknown')}\n"
            f"[bold]Created:[/bold] {role_data.get('created_at', 'Unknown')}\n\n"
            f"[bold]Description:[/bold]\n{role_data.get('description', 'No description available')}",
            title="🎭 Role Details",
            style="blue"
        ))
        
        # Capabilities
        capabilities = role_data.get('capabilities', [])
        if capabilities:
            cap_table = Table(title="🛠️ Capabilities")
            cap_table.add_column("Name", style="cyan")
            cap_table.add_column("Proficiency", style="green")
            cap_table.add_column("Description", style="white")
            
            for cap in capabilities:
                level = cap.get('proficiency_level', 0.0)
                level_bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
                cap_table.add_row(
                    cap.get('name', 'Unknown'),
                    f"{level_bar} {level:.1%}",
                    cap.get('description', 'No description')
                )
            
            console.print(cap_table)
        
        # Responsibilities
        responsibilities = role_data.get('responsibilities', [])
        if responsibilities:
            console.print("\n[bold yellow]📝 Responsibilities:[/bold yellow]")
            for i, resp in enumerate(responsibilities, 1):
                console.print(f"  {i}. {resp}")
        
        # Constraints
        constraints = role_data.get('constraints', [])
        if constraints:
            console.print("\n[bold red]🔒 Constraints:[/bold red]")
            for i, constraint in enumerate(constraints, 1):
                console.print(f"  {i}. {constraint}")
        
        # Tags
        tags = role_data.get('tags', [])
        if tags:
            console.print(f"\n[bold magenta]🏷️ Tags:[/bold magenta] {', '.join(tags)}")
        
    except Exception as e:
        print_error(f"Failed to load role details: {str(e)}")

@role.command()
@click.argument('role_name')
@click.argument('team_id')
def unassign(role_name: str, team_id: str):
    """Remove a role assignment from a team."""
    
    team_designer = TeamDesigner()
    
    # Load existing teams
    _load_existing_teams(team_designer)
    
    try:
        if team_id not in team_designer.teams:
            print_error(f"Team not found: {team_id}")
            return
        
        team = team_designer.teams[team_id]
        
        # Find and remove the member
        role_id = role_name.lower().replace(' ', '_')
        member_to_remove = None
        
        for member in team.members:
            if member.role_name == role_name or member.role_id == role_id:
                member_to_remove = member
                break
        
        if member_to_remove:
            team.members.remove(member_to_remove)
            team.updated_at = datetime.now().isoformat()
            
            # Update team status if needed
            if len(team.members) < team.min_size and team.status == "active":
                team.status = "forming"
            
            # Save updated team
            team_designer._save_team(team)
            
            console.print(Panel(
                f"✅ Removed role from team\n"
                f"🎭 Role: {role_name}\n"
                f"👥 Team ID: {team_id}\n"
                f"📊 Remaining members: {len(team.members)}/{team.max_size}",
                title="👥 Role Unassignment Successful",
                style="green"
            ))
        else:
            print_error(f"Role '{role_name}' not found in team. Use 'feriq role list' to see assigned roles.")
            
    except Exception as e:
        print_error(f"Failed to unassign role: {str(e)}")

# Add to main CLI
def add_role_commands(cli):
    """Add role commands to the main CLI."""
    cli.add_command(role)