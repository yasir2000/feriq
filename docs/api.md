# Feriq Framework API Documentation

**Version:** 1.1.0  
**Last Updated:** October 15, 2025

## Overview

The Feriq Framework provides both a Python API for programmatic access and a comprehensive CLI for command-line operations. This document covers all public APIs, methods, classes, and CLI commands available to developers and users.

## Table of Contents

1. [Python API](#python-api)
   - [Core Framework](#core-framework)
   - [Core Classes](#core-classes)
   - [Components](#components)
   - [Utilities](#utilities)
2. [CLI API](#cli-api)
   - [Command Groups](#command-groups)
   - [Usage Examples](#usage-examples)
3. [Configuration](#configuration)
4. [Error Handling](#error-handling)
5. [Examples](#examples)

---

## Python API

### Core Framework

#### FeriqFramework

The main orchestrator class that coordinates all framework components.

```python
from feriq import FeriqFramework

# Initialize framework
framework = FeriqFramework(name="My Framework", version="1.0.0")
```

**Constructor Parameters:**
- `name` (str): Framework instance name (default: "Feriq Framework")
- `version` (str): Framework version (default: "0.1.0")
- `config` (Config): Configuration object
- Additional component instances can be provided

**Key Methods:**

##### Agent Management
```python
# Create a new agent
agent = framework.create_agent(
    name="DataAnalyst",
    description="Analyzes data and generates insights",
    base_capabilities={"analysis": 0.9, "visualization": 0.7}
)

# Register existing agent
framework.register_agent(agent)

# Get agent by ID
agent = framework.get_agent(agent_id)

# Get available agents
available_agents = framework.get_available_agents()

# Get agents by state
busy_agents = framework.get_agents_by_state(AgentState.BUSY)
```

##### Role Management
```python
# Create a role
role = framework.create_role(
    name="Data Scientist",
    role_type=RoleType.SPECIALIST,
    description="Expert in data analysis and machine learning"
)

# Assign role to agent
success = framework.assign_role_to_agent(agent_id, "Data Scientist")

# Get dynamic role based on requirements
role = framework.get_dynamic_role({
    "domain": "data_science",
    "skills": ["python", "statistics"],
    "experience_level": "senior"
})
```

##### Task Management
```python
# Create a task
task = framework.create_task(
    name="Analyze Sales Data",
    description="Analyze Q3 sales data and generate report",
    goal_id=goal.id,
    task_type=TaskType.ANALYSIS,
    priority=TaskPriority.HIGH
)

# Assign task to specific agent
success = framework.assign_task(task.id, agent.id)

# Auto-assign task to best available agent
success = framework.assign_task(task.id)

# Complete task
framework.complete_task(task.id, result={"status": "completed"}, success=True)
```

##### Goal Management
```python
# Create a goal
goal = framework.create_goal(
    name="Q3 Business Analysis",
    description="Complete analysis of Q3 business performance",
    priority=GoalPriority.HIGH,
    deadline=datetime(2025, 12, 31)
)

# Execute goal with automatic planning
result = framework.execute_goal(goal, auto_plan=True)

# Execute goal from description
result = framework.execute_goal("Analyze customer churn patterns")
```

##### Framework Control
```python
# Start framework
await framework.start()

# Stop framework
await framework.stop()

# Check if running
if framework.is_running:
    print("Framework is active")

# Get metrics
metrics = framework.get_metrics()
health = framework.get_health_status()
```

### Core Classes

#### FeriqAgent

Represents an AI agent within the framework.

```python
from feriq import FeriqAgent, AgentState

agent = FeriqAgent(
    name="Assistant",
    description="General purpose assistant",
    model_name="gpt-4",
    base_capabilities={"reasoning": 0.8, "creativity": 0.7}
)
```

**Key Properties:**
- `id` (str): Unique agent identifier
- `name` (str): Agent name
- `state` (AgentState): Current state (IDLE, BUSY, ERROR)
- `current_role` (Role): Currently assigned role
- `assigned_tasks` (List[str]): List of assigned task IDs
- `capabilities` (Dict[str, float]): Agent capabilities (0.0-1.0)

**Key Methods:**
```python
# Role management
agent.assign_role(role)
agent.update_capabilities({"new_skill": 0.9})

# Task management
agent.assign_task(task_id)
agent.complete_task(task_id, success=True)

# Communication
response = await agent.communicate(message, context)

# State management
agent.set_state(AgentState.BUSY)
```

#### FeriqTask

Represents a task within the system.

```python
from feriq import FeriqTask, TaskType, TaskPriority, TaskStatus

task = FeriqTask(
    name="Data Processing",
    description="Process incoming data files",
    task_type=TaskType.PROCESSING,
    priority=TaskPriority.MEDIUM,
    requirements={"python": True, "pandas": True}
)
```

**Key Properties:**
- `id` (str): Unique task identifier
- `name` (str): Task name
- `status` (TaskStatus): Current status
- `assigned_agent_id` (str): ID of assigned agent
- `priority` (TaskPriority): Task priority
- `requirements` (Dict): Task requirements

**Key Methods:**
```python
# Task lifecycle
task.assign_to_agent(agent_id)
task.start_task()
task.complete_task(result)
task.fail_task(error_message)

# Progress tracking
task.update_progress(0.5)  # 50% complete
```

#### Goal

Represents a high-level objective.

```python
from feriq import Goal, GoalPriority, GoalStatus

goal = Goal(
    name="Improve Efficiency",
    description="Increase operational efficiency by 20%",
    priority=GoalPriority.HIGH,
    success_criteria={"efficiency_increase": 0.2}
)
```

#### Role

Represents a role that can be assigned to agents.

```python
from feriq import Role, RoleType

role = Role(
    name="Project Manager",
    role_type=RoleType.COORDINATOR,
    description="Coordinates project activities",
    required_capabilities={"leadership": 0.8, "communication": 0.9}
)
```

#### Plan

Represents an execution plan for achieving goals.

```python
from feriq import Plan, PlanStatus

plan = Plan(
    name="Q4 Strategy",
    description="Plan for Q4 objectives",
    goal_id=goal.id,
    steps=[
        {"action": "analyze", "duration": "2h"},
        {"action": "plan", "duration": "1h"},
        {"action": "execute", "duration": "5h"}
    ]
)
```

### Components

#### DynamicRoleDesigner

Designs roles dynamically based on requirements.

```python
role_designer = framework.role_designer

# Design role based on requirements
role = role_designer.design_role({
    "domain": "marketing",
    "skills": ["content_creation", "analytics"],
    "experience_level": "intermediate"
})

# Get role templates
templates = role_designer.get_role_templates()
```

#### TaskDesigner

Creates and designs tasks based on goals and context.

```python
task_designer = framework.task_designer

# Design tasks for a goal
tasks = task_designer.design_tasks_for_goal(goal_id)

# Create task from requirements
task = task_designer.create_task_from_requirements({
    "type": "analysis",
    "data_source": "sales_data.csv",
    "output_format": "report"
})
```

#### TaskAllocator

Allocates tasks to the most suitable agents.

```python
task_allocator = framework.task_allocator

# Allocate task automatically
success = task_allocator.allocate_task(task_id)

# Find best agent for task
agent_id = task_allocator.find_best_agent_for_task(task_id)

# Get allocation metrics
metrics = task_allocator.get_allocation_metrics()
```

#### PlanDesigner

Creates execution plans for goals.

```python
plan_designer = framework.plan_designer

# Create plan for goal
plan = plan_designer.create_plan_for_goal(goal_id)

# Create plan from description
plan = plan_designer.create_plan_from_description(
    "Complete market research and analysis"
)
```

#### WorkflowOrchestrator

Orchestrates workflow execution across agents.

```python
orchestrator = framework.orchestrator

# Execute workflow
result = await orchestrator.execute_workflow(plan_id)

# Monitor workflow
status = orchestrator.get_workflow_status(plan_id)

# Pause/resume workflow
orchestrator.pause_workflow(plan_id)
orchestrator.resume_workflow(plan_id)
```

#### Reasoner

Provides reasoning capabilities for decision-making.

```python
reasoner = framework.reasoner

# Perform deductive reasoning
conclusion = reasoner.deduce(premises, rules)

# Perform inductive reasoning
pattern = reasoner.induce(examples)

# Perform abductive reasoning
explanation = reasoner.abduce(observation, knowledge_base)

# Hybrid reasoning
result = reasoner.hybrid_reason(problem, context)
```

### Utilities

#### Logger

```python
from feriq.utils.logger import get_logger

logger = get_logger("MyComponent")
logger.info("Operation completed")
logger.error("An error occurred")
```

#### Config

```python
from feriq.utils.config import Config

config = Config()
config.load_from_file("config.json")
value = config.get("setting_name", default_value)
```

---

## CLI API

### Command Groups

#### Framework Management

```bash
# Initialize new project
feriq init my-project

# Check status
feriq status

# Get framework information
feriq status info
```

#### Model Management

```bash
# List available models
feriq model list

# Configure model
feriq model configure --provider ollama --model llama3

# Test model connection
feriq model test
```

#### Agent Management

```bash
# Create new agent
feriq agent create "DataAnalyst" --description "Analyzes data" --model gpt-4

# List agents
feriq agent list

# Show agent details
feriq agent show <agent-id>

# Update agent
feriq agent update <agent-id> --name "NewName"

# Delete agent
feriq agent delete <agent-id>
```

#### Goal Management

```bash
# Create goal
feriq goal create "Improve Sales" --description "Increase sales by 15%" --priority high

# List goals
feriq goal list

# Execute goal
feriq goal execute <goal-id>

# Show goal progress
feriq goal show <goal-id>
```

#### Workflow Management

```bash
# Create workflow
feriq workflow create "Data Pipeline" --description "ETL workflow"

# Run workflow
feriq workflow run <workflow-id>

# Monitor workflow
feriq workflow monitor <workflow-id>

# List workflows
feriq workflow list
```

#### Team Management

```bash
# Create team
feriq team create "DataTeam" --description "Data science team"

# Add agent to team
feriq team add-agent <team-id> <agent-id>

# List teams
feriq team list

# Show team details
feriq team show <team-id>

# Remove agent from team
feriq team remove-agent <team-id> <agent-id>

# Delete team
feriq team delete <team-id>
```

#### Role Management

```bash
# Create role
feriq role create "DataScientist" --type specialist --description "Data science expert"

# List roles
feriq role list

# Show role details
feriq role show "DataScientist"

# Assign role to agent
feriq role assign "DataScientist" <agent-id>

# Unassign role from agent
feriq role unassign "DataScientist" <agent-id>

# List role templates
feriq role templates
```

#### Reasoning Operations

```bash
# Perform deductive reasoning
feriq reason deduce --premises "premises.json" --rules "rules.json"

# Perform inductive reasoning
feriq reason induce --examples "examples.json"

# Perform abductive reasoning
feriq reason abduce --observation "observation.json" --kb "knowledge.json"

# Hybrid reasoning
feriq reason hybrid --problem "problem.json" --context "context.json"
```

#### Interactive Mode

```bash
# Start interactive session
feriq interactive

# In interactive mode:
> create agent "Assistant" --model gpt-4
> assign role "Helper" to agent-123
> execute goal "Complete analysis"
> exit
```

### Global Options

All commands support these global options:

- `--config, -c`: Specify configuration file
- `--verbose, -v`: Enable verbose output
- `--debug, -d`: Enable debug mode
- `--help`: Show help information

### Usage Examples

#### Basic Workflow

```bash
# 1. Initialize project
feriq init data-analysis-project

# 2. Create agents
feriq agent create "Analyst" --description "Data analyst" --model gpt-4
feriq agent create "Visualizer" --description "Creates charts" --model claude-3

# 3. Create team
feriq team create "DataTeam" --description "Data analysis team"

# 4. Add agents to team
feriq team add-agent team-123 agent-456
feriq team add-agent team-123 agent-789

# 5. Create and execute goal
feriq goal create "Analyze Q3 Data" --description "Complete Q3 analysis" --priority high
feriq goal execute goal-123
```

#### Advanced Usage

```bash
# Create specialized roles
feriq role create "SeniorAnalyst" --type specialist --description "Senior data analyst"

# Assign roles
feriq role assign "SeniorAnalyst" agent-456

# Monitor execution
feriq status info
feriq workflow monitor workflow-123

# Check agent status
feriq agent show agent-456
```

---

## Configuration

### Configuration File Structure

```json
{
  "framework": {
    "name": "My Feriq Instance",
    "version": "1.0.0",
    "log_level": "INFO"
  },
  "models": {
    "default_provider": "ollama",
    "providers": {
      "ollama": {
        "base_url": "http://localhost:11434",
        "models": ["llama3", "codellama"]
      },
      "openai": {
        "api_key": "your-api-key",
        "models": ["gpt-4", "gpt-3.5-turbo"]
      }
    }
  },
  "agents": {
    "default_model": "llama3",
    "max_concurrent": 5
  },
  "storage": {
    "output_dir": "./outputs",
    "backup_enabled": true
  }
}
```

### Environment Variables

- `FERIQ_CONFIG`: Path to configuration file
- `FERIQ_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `FERIQ_OUTPUT_DIR`: Output directory path
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key

---

## Error Handling

### Common Exceptions

```python
from feriq.exceptions import (
    FeriqError,
    AgentNotFoundError,
    TaskAllocationError,
    ModelConnectionError,
    ConfigurationError
)

try:
    agent = framework.get_agent("invalid-id")
except AgentNotFoundError as e:
    print(f"Agent not found: {e}")

try:
    framework.assign_task(task_id)
except TaskAllocationError as e:
    print(f"Task allocation failed: {e}")
```

### Error Codes

- `ERR_001`: Agent not found
- `ERR_002`: Task allocation failed
- `ERR_003`: Model connection error
- `ERR_004`: Configuration error
- `ERR_005`: Workflow execution error

---

## Examples

### Complete Python Example

```python
import asyncio
from feriq import FeriqFramework, RoleType, TaskType, GoalPriority

async def main():
    # Initialize framework
    framework = FeriqFramework(name="Data Analysis Framework")
    
    # Create agents
    analyst = framework.create_agent(
        name="DataAnalyst",
        description="Analyzes data and generates insights",
        model_name="gpt-4",
        base_capabilities={
            "data_analysis": 0.9,
            "visualization": 0.7,
            "reporting": 0.8
        }
    )
    
    # Create role
    analyst_role = framework.create_role(
        name="Senior Data Analyst",
        role_type=RoleType.SPECIALIST,
        description="Expert in data analysis and reporting"
    )
    
    # Assign role
    framework.assign_role_to_agent(analyst.id, analyst_role.name)
    
    # Create goal
    goal = framework.create_goal(
        name="Q3 Sales Analysis",
        description="Analyze Q3 sales data and generate comprehensive report",
        priority=GoalPriority.HIGH
    )
    
    # Execute goal
    await framework.start()
    result = framework.execute_goal(goal, auto_plan=True)
    
    print(f"Goal execution started: {result}")
    
    # Monitor progress
    while framework.is_running:
        status = framework.get_health_status()
        print(f"Framework status: {status}")
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
```

### CLI Workflow Example

```bash
#!/bin/bash

# Complete data analysis workflow using CLI

# 1. Setup
feriq init data-analysis --template analytics

# 2. Configure model
feriq model configure --provider ollama --model llama3

# 3. Create team structure
feriq team create "Analytics" --description "Data analytics team"

# 4. Create specialized agents
feriq agent create "DataProcessor" --model llama3 --team Analytics
feriq agent create "Visualizer" --model llama3 --team Analytics
feriq agent create "Reporter" --model llama3 --team Analytics

# 5. Create and assign roles
feriq role create "DataEngineer" --type specialist
feriq role create "Analyst" --type specialist  
feriq role create "ReportWriter" --type specialist

feriq role assign "DataEngineer" $(feriq agent list --filter name=DataProcessor --format id)
feriq role assign "Analyst" $(feriq agent list --filter name=Visualizer --format id)
feriq role assign "ReportWriter" $(feriq agent list --filter name=Reporter --format id)

# 6. Execute analysis
feriq goal create "Monthly Analysis" --description "Complete monthly data analysis" --priority high
feriq goal execute $(feriq goal list --latest --format id)

# 7. Monitor progress
feriq status info
feriq workflow monitor $(feriq workflow list --active --format id)
```

---

## API Reference Summary

### Core Classes
- `FeriqFramework`: Main orchestrator
- `FeriqAgent`: AI agent representation  
- `FeriqTask`: Task representation
- `Goal`: High-level objective
- `Role`: Agent role definition
- `Plan`: Execution plan

### Components
- `DynamicRoleDesigner`: Dynamic role creation
- `TaskDesigner`: Task design and creation
- `TaskAllocator`: Task-agent allocation
- `PlanDesigner`: Execution planning
- `WorkflowOrchestrator`: Workflow execution
- `Reasoner`: Reasoning engine

### CLI Commands
- `feriq init`: Initialize projects
- `feriq agent`: Agent management
- `feriq team`: Team management  
- `feriq role`: Role management
- `feriq goal`: Goal management
- `feriq workflow`: Workflow operations
- `feriq model`: Model configuration
- `feriq status`: System status
- `feriq interactive`: Interactive mode

This API documentation provides comprehensive coverage of all public interfaces in the Feriq Framework. For additional examples and tutorials, see the other documentation files in this directory.