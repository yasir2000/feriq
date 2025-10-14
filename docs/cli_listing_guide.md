# Feriq CLI Listing Commands Guide

## Overview

Feriq provides comprehensive CLI commands to list outputs from all framework components. This guide covers all available listing capabilities for the 8 core components plus general actions tracking.

## Available Components

### 🎭 Dynamic Role Designer
Automatically creates and assigns roles based on task requirements and context

### 📋 Task Designer & Allocator  
Intelligently breaks down goals into tasks and optimally assigns them to agents

### 📊 Plan Designer
Creates comprehensive execution plans with resource allocation and timeline management

### 👁️ Plan Observer
Real-time monitoring of plan execution with alerts and performance metrics

### 🎯 Goal-Oriented Agents
Intelligent agents that work towards specific goals with learning and adaptation

### 🎼 Workflow Orchestrator
Central coordinator for workflow execution and resource management

### 💃 Choreographer
Manages agent interactions, coordination patterns, and communication protocols

### 🧠 Reasoner
Advanced reasoning engine for decision-making, problem-solving, and strategic planning

## CLI Commands

### General Listing Commands

#### List All Components
```bash
# Show all available framework components
feriq list components

# Show detailed component information
feriq list components --detailed
```

#### Generate Sample Data
```bash
# Generate sample outputs for demonstration
feriq list generate-samples

# Generate without confirmation
feriq list generate-samples --confirm
```

### Component-Specific Listing Commands

#### 🎭 List Role Designer Outputs
```bash
# List all role definitions and assignments
feriq list roles

# List roles in JSON format
feriq list roles --format json

# List roles in YAML format  
feriq list roles --format yaml

# Filter roles by type
feriq list roles --filter research
```

**Outputs:**
- Role definitions with types and capabilities
- Role assignments to agents
- Available role templates
- Dynamic role generation results

#### 📋 List Task Designer Outputs
```bash
# List all tasks
feriq list tasks

# Filter by task status
feriq list tasks --status pending
feriq list tasks --status in_progress
feriq list tasks --status completed

# Filter by assigned agent
feriq list tasks --agent agent_001

# Output in different formats
feriq list tasks --format json
feriq list tasks --format yaml
```

**Outputs:**
- Task breakdowns with dependencies
- Task assignments to agents
- Allocation optimization reports
- Task dependency graphs

#### 📊 List Plan Designer Outputs
```bash
# List all execution plans
feriq list plans

# Show only active plans
feriq list plans --active-only

# Output in different formats
feriq list plans --format json
feriq list plans --format yaml
```

**Outputs:**
- Execution plans with timelines
- Resource allocation details
- Timeline schedules and milestones
- Plan templates

#### 👁️ List Plan Observer Outputs
```bash
# List recent observations and logs
feriq list observations

# Show more recent entries
feriq list observations --recent 20

# Filter by log level
feriq list observations --level warning
feriq list observations --level error
```

**Outputs:**
- Execution logs with timestamps
- Performance metrics
- Status reports
- Active alerts and notifications

#### 🎯 List Agent Outputs
```bash
# List all agents
feriq list agents

# Filter by agent status
feriq list agents --status active
feriq list agents --status busy
feriq list agents --status idle

# Filter by role
feriq list agents --role research_specialist

# Output in different formats
feriq list agents --format json
feriq list agents --format yaml
```

**Outputs:**
- Agent configurations and capabilities
- Goal progress tracking
- Learning logs and adaptations
- Performance metrics

#### 🎼 List Workflow Orchestrator Outputs
```bash
# List all workflows
feriq list workflows

# Filter by workflow status
feriq list workflows --status running
feriq list workflows --status completed

# Output in different formats
feriq list workflows --format json
feriq list workflows --format yaml
```

**Outputs:**
- Workflow definitions and stages
- Execution results and metrics
- Resource usage statistics
- Coordination logs

#### 💃 List Choreographer Outputs
```bash
# List agent interactions
feriq list interactions

# Filter by interaction pattern
feriq list interactions --pattern collaboration

# Show more recent interactions
feriq list interactions --recent 30
```

**Outputs:**
- Interaction patterns and protocols
- Communication logs between agents
- Coordination matrices and efficiency scores

#### 🧠 List Reasoner Outputs
```bash
# List reasoning results
feriq list reasoning

# Filter by reasoning type
feriq list reasoning --type causal
feriq list reasoning --type probabilistic
feriq list reasoning --type inductive

# Show more recent results
feriq list reasoning --recent 20
```

**Outputs:**
- Reasoning results with confidence scores
- Decision trees and logic flows
- Strategic recommendations
- Problem solutions and analyses

#### 🎬 List Actions Across All Components
```bash
# List recent actions from all components
feriq list actions

# Filter by specific component
feriq list actions --component role_designer
feriq list actions --component reasoner

# Show more recent actions
feriq list actions --recent 25
```

**Outputs:**
- Action history across all components
- Component action summaries
- System events and integration logs

## Output Formats

All listing commands support multiple output formats:

- **Table** (default): Human-readable table format
- **JSON**: Machine-readable JSON format
- **YAML**: YAML format for configuration files

```bash
# Examples of different formats
feriq list roles --format table    # Default
feriq list roles --format json     # JSON output
feriq list roles --format yaml     # YAML output
```

## Filtering and Options

### Common Filters

- **Status filters**: `--status active|pending|completed|failed`
- **Agent filters**: `--agent agent_name`
- **Role filters**: `--role role_name`
- **Component filters**: `--component component_name`
- **Type filters**: `--type type_name`
- **Recent limits**: `--recent N` (show N most recent items)

### Examples

```bash
# Show only active tasks assigned to agent_001
feriq list tasks --status active --agent agent_001

# Show last 10 warning-level observations
feriq list observations --level warning --recent 10

# Show recent causal reasoning results
feriq list reasoning --type causal --recent 5

# Show actions from reasoning component only
feriq list actions --component reasoner
```

## Directory Structure

Listing commands look for outputs in the following directories:

```
project_root/
├── outputs/
│   ├── roles/           # Role designer outputs
│   ├── tasks/           # Task designer outputs
│   ├── plans/           # Plan designer outputs
│   ├── observations/    # Plan observer outputs
│   ├── agents/          # Agent outputs
│   ├── workflows/       # Workflow orchestrator outputs
│   ├── interactions/    # Choreographer outputs
│   ├── reasoning/       # Reasoner outputs
│   └── actions/         # Actions log
├── agents/              # Agent configurations
├── goals/               # Goal definitions
├── workflows/           # Workflow definitions
└── logs/               # General logs
```

## Sample Output Examples

### Component Overview
```bash
$ feriq list components

🏗️  Feriq Framework Components
====================================
🎭 Dynamic Role Designer - ✅ Available
📋 Task Designer & Allocator - ✅ Available
📊 Plan Designer - ✅ Available
👁️ Plan Observer - ✅ Available
🎯 Goal-Oriented Agents - ✅ Available
🎼 Workflow Orchestrator - ✅ Available
💃 Choreographer - ✅ Available
🧠 Reasoner - ✅ Available

💡 Use 'feriq list <component>' to see specific outputs
💡 Use 'feriq list actions' to see available actions
```

### Task Listing Example
```bash
$ feriq list tasks --status in_progress

📋 Task Designer & Allocator Outputs
=====================================
Task ID | Name | Status | Agent | Priority
--------------------------------------------------
task_001 | Literature Review | in_progress | agent_001 | high
task_002 | Data Collection | in_progress | agent_004 | medium

📊 Task Allocation Summary:
  • Optimal allocation achieved with 95% efficiency
  • Agent workload balanced across skill sets
```

### Reasoning Results Example
```bash
$ feriq list reasoning --type causal --recent 3

🧠 Reasoner Outputs
====================
🔍 Recent Reasoning Results (3):
  • [causal] Task delay caused by dependency wait on task_001 completion... (confidence: 0.89)
  • [causal] Resource bottleneck identified in GPU allocation for training... (confidence: 0.85)
  • [causal] Communication gaps leading to coordination delays... (confidence: 0.78)

🎯 Strategic Recommendations:
  • [HIGH] Increase parallel task execution to accelerate project timeline
  • [HIGH] Establish automated quality gates to prevent defect propagation
```

## Integration with Other Commands

The listing commands integrate seamlessly with other Feriq CLI commands:

```bash
# Create a goal and then list resulting tasks
feriq goal create --name "Build AI system"
feriq list tasks

# Run reasoning and then view results
feriq reason analyze --query "Project timeline risks"
feriq list reasoning --type probabilistic

# Execute workflow and monitor progress
feriq workflow run workflow_001
feriq list observations --recent 10
```

## Troubleshooting

### No Outputs Found
If listing commands show "No outputs found", try:

1. Generate sample outputs: `feriq list generate-samples`
2. Create some framework activities first
3. Check directory permissions
4. Verify project structure

### Performance Tips

1. Use `--recent N` to limit output for large datasets
2. Apply filters to reduce result sets
3. Use `--format json` for machine processing
4. Check specific component listings rather than general actions

## Advanced Usage

### Scripting and Automation
```bash
# Get JSON output for processing
feriq list agents --format json > agents.json

# Filter and count active tasks
feriq list tasks --status active --format json | jq 'length'

# Monitor recent errors
feriq list observations --level error --recent 5
```

### Continuous Monitoring
```bash
# Set up monitoring script
while true; do
    echo "=== $(date) ==="
    feriq list observations --level warning --recent 5
    sleep 60
done
```

This comprehensive listing system provides complete visibility into all Feriq framework component activities and outputs, enabling effective monitoring, debugging, and analysis of multi-agent workflows.