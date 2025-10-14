# Quick Start Guide - Feriq Framework

Get up and running with the Feriq Collaborative AI Agents Framework in 5 minutes.

## 🚀 30-Second Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test CLI**:
   ```bash
   python -m feriq.cli.main --help
   ```

3. **List Framework Components**:
   ```bash
   python -m feriq.cli.main list components
   ```

4. **Generate Sample Outputs**:
   ```bash
   python -m feriq.cli.main list generate-samples --confirm
   ```

5. **View Generated Outputs**:
   ```bash
   python -m feriq.cli.main list outputs
   ```

**🎉 You're ready to use Feriq!**

---

## 📋 5-Minute Setup

### Step 1: Environment Setup

```bash
# Clone or ensure you're in the feriq directory
cd /path/to/feriq

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -m feriq.cli.main --version
```

### Step 2: Verify Framework Components

```bash
# List all 8 framework components
python -m feriq.cli.main list components --detailed

# Expected output:
# 🎭 Role Designer: Dynamic role creation and assignment
# 📋 Task Designer: Task breakdown and allocation
# 📊 Plan Designer: Execution planning and optimization
# 👁️ Plan Observer: Real-time monitoring and alerts
# 🎯 Agent System: Goal-oriented intelligent agents
# 🎼 Workflow Orchestrator: Workflow coordination
# 💃 Choreographer: Agent interaction management
# 🧠 Reasoner: Advanced reasoning engine
```

### Step 3: Generate Sample Data

```bash
# Generate professional sample outputs for all components
python -m feriq.cli.main list generate-samples --confirm

# This creates sample data in feriq/outputs/ directory
```

### Step 4: Explore CLI Capabilities

```bash
# List all available commands
python -m feriq.cli.main --help

# List component outputs
python -m feriq.cli.main list outputs

# View specific component outputs
python -m feriq.cli.main list roles
python -m feriq.cli.main list tasks
python -m feriq.cli.main list plans

# Test reasoning capabilities
python -m feriq.cli.main plan demo --type all
```

### Step 5: Configure Your First Project

```bash
# Create a new project
python -m feriq.cli.main init-project "My First Project"

# View project structure
python -m feriq.cli.main list outputs --filter-type plans
```

---

## 🧠 Quick Reasoning Demo

### Test All Reasoning Types

```bash
# Demonstrate all 10+ reasoning types
python -m feriq.cli.main plan demo --type all

# Specific reasoning types
python -m feriq.cli.main plan demo --type analytical
python -m feriq.cli.main plan demo --type creative
python -m feriq.cli.main plan demo --type strategic
```

### Intelligent Planning

```bash
# Generate reasoning-enhanced plan
python -m feriq.cli.main plan create "Software Development Project" \
  --use-reasoning \
  --reasoning-type strategic

# View created plan
python -m feriq.cli.main list plans --detailed
```

---

## 💻 Quick Programming Example

### Basic Framework Usage

```python
# demo_usage.py
from feriq.components.role_designer import RoleDesigner
from feriq.components.task_designer import TaskDesigner
from feriq.components.plan_designer import PlanDesigner

# Create components
role_designer = RoleDesigner()
task_designer = TaskDesigner()
plan_designer = PlanDesigner()

# Create a role
role = role_designer.create_role(
    name="Senior Developer",
    responsibilities=["Code architecture", "Code review", "Mentoring"],
    skills=["Python", "System Design", "Leadership"]
)

# Create tasks
tasks = task_designer.create_tasks_for_project(
    project_name="E-commerce Platform",
    requirements=["User authentication", "Product catalog", "Payment system"]
)

# Create execution plan
plan = plan_designer.create_execution_plan(
    project_name="E-commerce Platform",
    roles=[role],
    tasks=tasks
)

print(f"Created plan with {len(plan.phases)} phases")
```

### Run the Demo

```bash
python demo_usage.py
```

---

## 🔧 Common Commands Reference

### Component Listing Commands

```bash
# List all component outputs
python -m feriq.cli.main list outputs

# Filter by component type
python -m feriq.cli.main list roles
python -m feriq.cli.main list tasks
python -m feriq.cli.main list plans
python -m feriq.cli.main list observations
python -m feriq.cli.main list agents
python -m feriq.cli.main list workflows
python -m feriq.cli.main list choreographies
python -m feriq.cli.main list reasoning

# Detailed output
python -m feriq.cli.main list roles --detailed
python -m feriq.cli.main list tasks --summary
```

### Planning Commands

```bash
# Create plans with reasoning
python -m feriq.cli.main plan create "Project Name" --use-reasoning
python -m feriq.cli.main plan analyze "Project Name" --reasoning-type analytical
python -m feriq.cli.main plan optimize "Project Name" --strategy performance

# Planning demos
python -m feriq.cli.main plan demo --type strategic
python -m feriq.cli.main plan demo --type adaptive
```

### Project Management

```bash
# Initialize new projects
python -m feriq.cli.main init-project "Project Name"
python -m feriq.cli.main create-role "Role Name" --skills "skill1,skill2"
python -m feriq.cli.main create-task "Task Name" --priority high

# Monitor outputs
python -m feriq.cli.main list outputs --recent
python -m feriq.cli.main list outputs --filter-name "pattern"
```

---

## 📊 Understanding the Framework

### 8 Core Components

1. **🎭 Role Designer**: Creates and manages agent roles
2. **📋 Task Designer**: Breaks down and allocates tasks
3. **📊 Plan Designer**: Creates execution plans with reasoning
4. **👁️ Plan Observer**: Monitors execution and provides alerts
5. **🎯 Agent System**: Manages goal-oriented intelligent agents
6. **🎼 Workflow Orchestrator**: Coordinates complex workflows
7. **💃 Choreographer**: Manages agent interactions
8. **🧠 Reasoner**: Provides advanced reasoning capabilities

### CLI Architecture

- **List Commands**: Comprehensive component output listing
- **Planning Commands**: Reasoning-enhanced intelligent planning
- **Project Commands**: Full project lifecycle management
- **Component Commands**: Direct component interaction

### Output Organization

```
feriq/outputs/
├── roles/           # Role definitions and assignments
├── tasks/           # Task breakdowns and allocations
├── plans/           # Execution plans and strategies
├── observations/    # Monitoring data and alerts
├── agents/          # Agent configurations and states
├── workflows/       # Workflow definitions and executions
├── choreographies/  # Agent interaction patterns
└── reasoning/       # Reasoning analyses and decisions
```

---

## 🎯 Next Steps

### After Quick Start

1. **Explore Documentation**:
   - Read [CLI Listing Guide](cli_listing_guide.md) for comprehensive CLI usage
   - Review [Architecture Overview](architecture.md) for system understanding
   - Study [Reasoning Usage Guide](reasoning_usage.md) for advanced capabilities

2. **Try Advanced Features**:
   - Generate larger sample datasets
   - Create complex multi-agent plans
   - Test different reasoning strategies

3. **Build Your First Project**:
   - Use the framework for a real project
   - Integrate with your existing workflows
   - Customize components for your needs

### Common Next Actions

```bash
# Generate comprehensive samples
python -m feriq.cli.main list generate-samples --confirm --count 10

# Create a complex project plan
python -m feriq.cli.main plan create "Enterprise Software Project" \
  --use-reasoning \
  --reasoning-type strategic \
  --include-monitoring

# Explore all CLI capabilities
python -m feriq.cli.main --help
python -m feriq.cli.main list --help
python -m feriq.cli.main plan --help
```

---

## 🆘 Getting Help

### CLI Help

```bash
# General help
python -m feriq.cli.main --help

# Command-specific help
python -m feriq.cli.main list --help
python -m feriq.cli.main plan --help

# Component information
python -m feriq.cli.main list components --help
```

### Documentation

- **[Main README](../README.md)**: Comprehensive overview
- **[CLI Guide](cli_listing_guide.md)**: Complete CLI reference
- **[Architecture](architecture.md)**: System design documentation
- **[Examples](examples/)**: Practical usage examples

### Troubleshooting

- **Import Errors**: Ensure all dependencies are installed
- **CLI Not Found**: Run from feriq root directory
- **No Outputs**: Run `generate-samples` first
- **Model Integration**: Check model configuration

---

**🚀 You're ready to build amazing collaborative AI systems with Feriq!**

Need more help? Check the [documentation index](README.md) or explore the CLI help system.