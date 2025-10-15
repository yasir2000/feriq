# 🎭 Role Management System Documentation

The Feriq Role Management System provides comprehensive CLI commands for creating, managing, and assigning roles to teams within the collaborative AI framework.

## 🌟 Overview

The role management system enables:
- **Dynamic Role Creation**: Create custom roles with specific capabilities and responsibilities
- **Team Role Assignment**: Assign roles to teams with specializations and contribution levels
- **Role Lifecycle Management**: Complete CRUD operations for role management
- **Integration with Teams**: Seamless integration with the team management system
- **Professional CLI Interface**: Rich console output with tables, panels, and progress indicators

## 📋 Available Commands

### Role Creation
```bash
# Create a basic role
feriq role create "Software Developer" executor \
  --description "Full-stack software development specialist"

# Create role with capabilities and proficiency levels
feriq role create "Data Scientist" analyst \
  --description "Machine learning and data analysis specialist" \
  --capabilities "python:0.9,machine_learning:0.8,statistics:0.7,sql:0.8" \
  --responsibilities "Data analysis,Model development,Statistical analysis"

# Create role with constraints and tags
feriq role create "Project Manager" coordinator \
  --description "Agile project management specialist" \
  --capabilities "project_management:0.9,agile:0.8,leadership:0.7" \
  --responsibilities "Project planning,Team coordination,Risk management" \
  --constraints "Budget limitations,Timeline constraints" \
  --tags "leadership,management,agile"
```

### Role Listing and Information
```bash
# List all roles in table format
feriq role list

# List roles with detailed information
feriq role list --detailed

# List roles in JSON format
feriq role list --format json

# Filter roles by capabilities or type
feriq role list --filter "python"

# Show detailed role information
feriq role show "Software Developer"
```

### Role Assignment to Teams
```bash
# Assign role to team with specialization
feriq role assign "role_software_developer.json" <team_id> \
  --specialization "Lead Developer" \
  --contribution 1.0

# Assign multiple roles to the same team
feriq role assign "role_qa_engineer.json" <team_id> \
  --specialization "Senior QA Engineer"

feriq role assign "role_devops_engineer.json" <team_id> \
  --specialization "Infrastructure Lead"
```

### Role Unassignment
```bash
# Remove role from team
feriq role unassign "Software Developer" <team_id>

# Remove specific role assignments
feriq role unassign "QA Engineer" <team_id>
```

### Role Templates
```bash
# List available role templates
feriq role templates

# Create role from template
feriq role create "Custom Analyst" analyst --template "data_analyst"
```

## 🏗️ Role Types

The system supports 8 different role types:

| Role Type | Description | Example Use Cases |
|-----------|-------------|-------------------|
| `researcher` | Investigation and analysis focused | Market research, Technical research |
| `analyst` | Data analysis and interpretation | Data scientist, Business analyst |
| `planner` | Strategic planning and coordination | Project manager, Product manager |
| `executor` | Implementation and development | Software developer, Designer |
| `coordinator` | Team coordination and leadership | Team lead, Scrum master |
| `reviewer` | Quality assurance and validation | QA engineer, Code reviewer |
| `specialist` | Domain-specific expertise | Security expert, AI specialist |
| `generalist` | Cross-functional capabilities | Full-stack developer, Consultant |

## 🛠️ Role Capabilities

Capabilities define what a role can do and at what proficiency level (0.0 - 1.0):

```bash
# Examples of capability definitions
--capabilities "python:0.9,javascript:0.8,sql:0.7,git:0.8"
--capabilities "machine_learning:0.9,tensorflow:0.8,data_analysis:0.9"
--capabilities "project_management:0.9,agile:0.8,leadership:0.7,risk_management:0.6"
```

Common capability categories:
- **Technical Skills**: `python`, `javascript`, `sql`, `git`, `docker`, `kubernetes`
- **Frameworks**: `react`, `django`, `tensorflow`, `pytorch`, `spring_boot`
- **Methodologies**: `agile`, `scrum`, `devops`, `tdd`, `ci_cd`
- **Domains**: `machine_learning`, `data_analysis`, `security`, `ui_ux`, `databases`
- **Soft Skills**: `leadership`, `communication`, `problem_solving`, `teamwork`

## 📊 Team Integration

The role management system integrates seamlessly with team management:

### Team Status Updates
- When a role is assigned to a team, the member count increases
- Teams with minimum members automatically change status from "forming" to "active"
- When roles are unassigned, teams may revert to "forming" status if below minimum size

### Role Assignment Workflow
1. **Create Team**: `feriq team create "Development Team" software_development`
2. **Create Roles**: Define specialized roles for the project
3. **Assign Roles**: Assign roles to teams with specific specializations
4. **Monitor Teams**: Use `feriq list teams` to see member counts and status
5. **Manage Assignments**: Add or remove role assignments as needed

### Example Complete Workflow
```bash
# 1. Create a development team
feriq team create "Frontend Team" software_development \
  --description "React and UI development specialists"

# 2. Create specialized roles
feriq role create "Frontend Developer" executor \
  --description "React and modern frontend specialist" \
  --capabilities "react:0.9,javascript:0.8,css:0.8,html:0.9" \
  --responsibilities "UI development,Component creation,Frontend testing"

feriq role create "UI Designer" specialist \
  --description "User interface and experience designer" \
  --capabilities "ui_design:0.9,ux_design:0.8,figma:0.9,prototyping:0.7" \
  --responsibilities "UI design,Prototyping,User research"

# 3. Get team ID from listing
feriq list teams

# 4. Assign roles to team
feriq role assign "role_frontend_developer.json" <team_id> \
  --specialization "Senior Frontend Developer"

feriq role assign "role_ui_designer.json" <team_id> \
  --specialization "Lead UI Designer"

# 5. Verify assignments
feriq list teams  # Should show 2/10 members for Frontend Team
feriq role list   # Should show created roles
```

## 📁 File Structure

Role files are saved in the `outputs/roles/` directory:

```
outputs/
├── roles/
│   ├── role_software_developer.json
│   ├── role_qa_engineer.json
│   ├── role_data_scientist.json
│   └── role_ui_designer.json
└── teams/
    ├── team_<team_id>.json
    └── ...
```

### Role File Format
```json
{
  "name": "Software Developer",
  "role_type": "executor",
  "description": "Full-stack software development specialist",
  "capabilities": [
    {
      "name": "python",
      "description": "Capability: python",
      "proficiency_level": 0.9
    },
    {
      "name": "javascript", 
      "description": "Capability: javascript",
      "proficiency_level": 0.8
    }
  ],
  "responsibilities": [
    "Write code",
    "Debug issues",
    "Code review",
    "Documentation"
  ],
  "constraints": [],
  "created_at": "2025-10-15T18:20:25.403161",
  "version": "1.0",
  "tags": []
}
```

## 🎯 Best Practices

### Role Design
1. **Be Specific**: Create roles with clear, specific capabilities and responsibilities
2. **Use Realistic Proficiency Levels**: Set capability levels based on actual expertise (0.7-0.9 for strong skills)
3. **Include Relevant Constraints**: Define limitations that affect role performance
4. **Tag Appropriately**: Use tags for easy filtering and categorization

### Team Assignment
1. **Match Skills to Needs**: Assign roles that match team discipline and project requirements
2. **Define Specializations**: Use specialization field to clarify the specific role within the team
3. **Balance Contribution Levels**: Set appropriate contribution levels based on role importance
4. **Monitor Team Composition**: Regularly check team member counts and role distribution

### Workflow Integration
1. **Create Roles Before Teams**: Define roles first, then create teams and assign roles
2. **Use Consistent Naming**: Follow consistent naming conventions for roles and specializations
3. **Document Role Purposes**: Use clear descriptions that explain the role's purpose and scope
4. **Regular Maintenance**: Review and update role assignments as project needs evolve

## 🔧 Advanced Usage

### Bulk Role Creation
```bash
# Create multiple roles for a software project
for role in "Backend Developer" "Frontend Developer" "DevOps Engineer" "QA Engineer"; do
  feriq role create "$role" executor --description "Project specialist role"
done
```

### Role Assignment Automation
```bash
# Script to assign all project roles to a team
TEAM_ID="your-team-id-here"

feriq role assign "role_backend_developer.json" $TEAM_ID --specialization "Senior Backend"
feriq role assign "role_frontend_developer.json" $TEAM_ID --specialization "Senior Frontend" 
feriq role assign "role_devops_engineer.json" $TEAM_ID --specialization "Infrastructure Lead"
feriq role assign "role_qa_engineer.json" $TEAM_ID --specialization "QA Lead"
```

### Role Capability Analysis
```bash
# Analyze roles with specific capabilities
feriq role list --filter "python" --format json | jq '.[] | .name'
feriq role list --filter "machine_learning" --detailed
```

## 🐛 Troubleshooting

### Common Issues

**Role Assignment Fails**
- Verify team ID exists: `feriq list teams`
- Check role file exists: `ls outputs/roles/`
- Ensure team is not at maximum capacity

**Role Not Found**
- Check exact role name: `feriq role list`
- Verify file path: role files should be in `outputs/roles/`
- Use proper file naming: `role_<name>.json`

**Team Status Not Updating**
- Verify minimum team size requirements
- Check if role assignment was successful
- Review team configuration

### Debug Commands
```bash
# List all teams with member counts
feriq list teams

# Show detailed role information
feriq role show "<role_name>"

# List role files
ls outputs/roles/

# Check team files
ls outputs/teams/
```

## 🚀 Future Enhancements

Planned improvements for the role management system:
- **Role Dependencies**: Define prerequisite roles for complex assignments
- **Skill Recommendations**: AI-powered role suggestions based on project analysis
- **Performance Tracking**: Monitor role effectiveness and team contribution
- **Role Evolution**: Automatic capability updates based on performance
- **Certification Integration**: Link roles to external certifications and training