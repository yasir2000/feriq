

<img width="526" height="290" alt="image" src="https://github.com/user-attachments/assets/3514c1f4-7570-44fe-aef9-64b214715d41" />


# 🎉 Feriq - Complete Collaborative AI Agents Framework v1.0.0

**The Future of Autonomous Team Collaboration is Here! 🚀**

<img width="526" height="290" alt="image" src="https://github.com/user-attachments/assets/3514c1f4-7570-44fe-aef9-64b214715d41" />

Feriq is a comprehensive collaborative AI agents framework that enables intelligent multi-agent coordination, autonomous team collaboration, advanced reasoning, and sophisticated workflow orchestration. This is the first complete release featuring all **9 core components**, a powerful CLI system with **60+ commands**, **real LLM integration** with DeepSeek/Ollama, and **production-ready autonomous team management**.

## 🌟 Major Features

### 🏗️ Complete 9-Component Framework
- **🎭 Role Designer**: Dynamic role creation and assignment system
- **📋 Task Designer**: Intelligent task breakdown and allocation with team collaboration
- **📊 Plan Designer**: Execution planning with AI reasoning integration
- **👁️ Plan Observer**: Real-time monitoring and alerting system
- **🎯 Agent System**: Goal-oriented intelligent agent management
- **🎼 Workflow Orchestrator**: Complex workflow coordination
- **💃 Choreographer**: Agent interaction management
- **🧠 Reasoner**: Advanced reasoning engine with 10+ reasoning types
- **👥 Team Designer**: **NEW** - Autonomous team collaboration and coordination system

### 🤖 LLM-Powered Intelligence
- **Real AI Integration**: **DeepSeek, Ollama, OpenAI, Azure OpenAI** support
- **Intelligent Problem Analysis**: LLMs analyze complex problems and recommend optimal team structures
- **Autonomous Task Assignment**: AI agents automatically assign tasks based on team capabilities
- **Inter-Agent Communication**: Intelligent agent-to-agent communication using natural language
- **Goal Extraction & Refinement**: AI-powered goal decomposition from problem descriptions
- **Performance Optimization**: Continuous learning and adaptation based on team outcomes

### 👥 Autonomous Team Collaboration
- **Multi-Team Coordination**: Concurrent and cooperative team execution
- **Specialized Disciplines**: Software development, data science, research, marketing, finance, design, operations
- **Autonomous Problem-Solving**: Teams extract goals, refine objectives, and solve problems independently
- **Collaborative Workflows**: Cross-functional collaboration between teams with different specializations
- **Intelligent Task Distribution**: AI-powered task allocation based on team capabilities and availability

### 🖥️ Comprehensive CLI Interface
- **� Complete Listing System**: List outputs from all 8 framework components with filtering and formatting
- **🤖 Ollama Integration**: Seamless integration with Ollama models (llama3.1, deepseek-r1, qwen2.5, etc.)
- **📁 Project Management**: Initialize, configure, and manage multi-agent projects from terminal
- **📊 Real-time Monitoring**: Monitor execution logs, performance metrics, alerts, and component activities
- **🔧 Model Management**: Test, configure, and switch between different LLM models interactively

### 🧠 Advanced Reasoning & Planning
- **Reasoning-Enhanced Planning**: Intelligent planning using causal, probabilistic, temporal, spatial, and collaborative reasoning
- **Multi-Strategy Planning**: 7 intelligent planning strategies for different project types and challenges
- **Comprehensive Reasoning Types**: Inductive, deductive, probabilistic, causal, abductive, analogical, temporal, spatial, hybrid, and collaborative reasoning
- **Strategic Recommendations**: AI-powered strategic recommendations with confidence scoring
- **Decision Trees**: Automated decision tree generation for complex scenarios

### 🎯 Production-Ready Features
- **Comprehensive Output Tracking**: Track and list outputs from all framework components
- **Flexible Filtering**: Filter by status, agent, role, component, type, and time
- **Multiple Output Formats**: Table, JSON, and YAML formats for different use cases
- **Sample Data Generation**: Generate demonstration data to explore framework capabilities
- **Real-time Performance Metrics**: Monitor task completion rates, agent utilization, and system efficiency
- **Cross-Component Integration**: Seamless integration between all framework components

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yasir2000/feriq.git
cd feriq

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🖥️ CLI Quick Start

Feriq provides a comprehensive command-line interface for managing multi-agent workflows with advanced listing and monitoring capabilities:

### Initialize a New Project
```bash
# Create a new Feriq project with model setup
python -m feriq.cli.main init project --name my-ai-project --model-setup

# Navigate to project directory
cd my-ai-project
```

### Model Management
```bash
# List available models (supports Ollama, OpenAI, Anthropic)
python -m feriq.cli.main model list

# Test a specific model
python -m feriq.cli.main model test ollama llama3.1:8b

# Interactive model setup
python -m feriq.cli.main model setup
```

### 📋 Comprehensive Component Listing
```bash
# Overview of all framework components
python -m feriq.cli.main list components --detailed

# List role designer outputs
python -m feriq.cli.main list roles --format table

# List tasks with filtering
python -m feriq.cli.main list tasks --status active --agent agent_001

# Monitor plan execution
python -m feriq.cli.main list observations --recent 10 --level warning

# View reasoning results
python -m feriq.cli.main list reasoning --type causal --recent 5

# Track actions across all components
python -m feriq.cli.main list actions --component reasoner --recent 15

# List agent configurations and progress
python -m feriq.cli.main list agents --status active --format json

# Monitor workflow orchestration
python -m feriq.cli.main list workflows --status running

# View agent interactions
python -m feriq.cli.main list interactions --recent 20
```

### 🧠 Reasoning-Enhanced Planning
```bash
# List available reasoning strategies
python -m feriq.cli.main plan strategies

# Create intelligent plan with reasoning
python -m feriq.cli.main plan create \
  --goal "Develop AI-powered recommendation system" \
  --type development \
  --strategy hybrid_intelligent \
  --priority high \
  --deadline 90

# Analyze planning requirements
python -m feriq.cli.main plan analyze \
  --goal "Launch fintech startup" \
  --context "Market competition" \
  --context "Regulatory compliance"

# Run planning demonstrations
python -m feriq.cli.main plan demo --type software
```

### 🎭 Sample Data Generation
```bash
# Generate sample outputs for demonstration
python -m feriq.cli.main list generate-samples --confirm

# Run comprehensive demos
python -m feriq.demos.sample_output_generator
python -m feriq.demos.intelligent_planning_demo
```

### Traditional Project Management
```bash
# Show project status
python -m feriq.cli.main status show

# Create agents, goals, workflows
python -m feriq.cli.main agent create research-agent
python -m feriq.cli.main goal create market-research
python -m feriq.cli.main workflow create research-workflow

# Get comprehensive help
python -m feriq.cli.main --help
python -m feriq.cli.main list --help
python -m feriq.cli.main plan --help
```

## 🏃‍♂️ Programming Quick Start

### Simple Multi-Agent Example
```python
from feriq.core.framework import FeriqFramework
from feriq.core.agent import FeriqAgent
from feriq.core.goal import Goal, GoalType, GoalPriority
from feriq.core.role import Role, RoleCapability
from feriq.components.reasoning_plan_designer import ReasoningEnhancedPlanDesigner, ReasoningPlanningStrategy
from datetime import timedelta, datetime

# Initialize framework with reasoning-enhanced planning
framework = FeriqFramework()
planner = ReasoningEnhancedPlanDesigner()

# Create a goal
goal = Goal(
    description="Develop AI-powered customer service system",
    goal_type=GoalType.DEVELOPMENT,
    priority=GoalPriority.HIGH,
    deadline=datetime.now() + timedelta(days=90)
)

# Create specialized roles
researcher_role = Role(
    name="AI Researcher",
    description="Specialist in AI research and analysis",
    capabilities=[
        RoleCapability("research", 0.9),
        RoleCapability("analysis", 0.8),
        RoleCapability("technical_writing", 0.7)
    ]
)

developer_role = Role(
    name="AI Developer", 
    description="Expert in AI system development",
    capabilities=[
        RoleCapability("coding", 0.9),
        RoleCapability("system_design", 0.8),
        RoleCapability("ai_integration", 0.9)
    ]
)

# Create intelligent agents
research_agent = FeriqAgent(
    name="ResearchBot",
    role=researcher_role,
    capabilities=["research", "analysis", "technical_writing"]
)

dev_agent = FeriqAgent(
    name="DevBot",
    role=developer_role,
    capabilities=["coding", "system_design", "ai_integration"]
)

# Add agents to framework
framework.add_agent(research_agent)
framework.add_agent(dev_agent)

# Generate intelligent plan using reasoning
async def create_intelligent_plan():
    plan = await planner.design_intelligent_plan(
        goal=goal,
        reasoning_strategy=ReasoningPlanningStrategy.HYBRID_INTELLIGENT,
        planning_context={
            'resource_constraints': {'team_size': '2 agents', 'timeline': '90 days'},
            'risk_factors': ['integration_complexity', 'performance_requirements'],
            'stakeholder_preferences': {'CTO': 'scalability', 'Product': 'user_experience'}
        }
    )
    return plan

# Execute the workflow
plan = asyncio.run(create_intelligent_plan())
framework.execute_plan(plan)
```

### Reasoning Integration Example
```python
from feriq.core.reasoning_integration import ReasoningAgent
from feriq.reasoning.reasoning_types import ReasoningType
import asyncio

# Create reasoning-enhanced agent
reasoning_agent = ReasoningAgent(
    name="AnalyticsBot",
    reasoning_types=[
        ReasoningType.CAUSAL,
        ReasoningType.PROBABILISTIC,
        ReasoningType.INDUCTIVE
    ]
)

# Apply multiple reasoning types
async def analyze_with_reasoning():
    results = await reasoning_agent.apply_multiple_reasoning(
        query="Why did our user engagement drop by 15% last month?",
        reasoning_types=[ReasoningType.CAUSAL, ReasoningType.INDUCTIVE],
        context={
            'user_metrics': 'engagement_data.json',
            'feature_releases': 'recent_changes.json'
        }
    )
    
    for reasoning_type, result in results.items():
        print(f"{reasoning_type}: {result.conclusions[0].statement}")
        print(f"Confidence: {result.conclusions[0].confidence}")

asyncio.run(analyze_with_reasoning())
```

## 📖 Examples & Demonstrations

### CLI Examples
```bash
# Complete workflow example
### CLI Demonstrations
```bash
# Generate sample data and explore framework capabilities
python -m feriq.cli.main list generate-samples --confirm

# View all component outputs
python -m feriq.cli.main list components --detailed

# Monitor system performance
python -m feriq.cli.main list observations --recent 10
python -m feriq.cli.main list actions --recent 15

# Test reasoning capabilities
python -m feriq.cli.main list reasoning --type causal
python -m feriq.cli.main plan demo --type software

# Initialize and test project
cd examples
python -m feriq.cli.main init project --name demo-project --model-setup
cd demo-project
python -m feriq.cli.main model test ollama llama3.1:8b
python -m feriq.cli.main status show
```

### Programming Examples
```bash
# Framework examples
cd examples
python comprehensive_example.py
python reasoning_integration_example.py

# Run intelligent planning demo
python -m feriq.demos.intelligent_planning_demo

# Test reasoning capabilities
python -m feriq.examples.reasoning_integration
python -m feriq.examples.reasoning_planning_examples
```

### Example Scenarios
1. **🔬 Research Project**: Multi-agent research coordination with reasoning-enhanced planning
2. **💻 Software Development**: End-to-end development workflow with causal dependency optimization
3. **📊 Data Analysis**: Collaborative data processing with probabilistic risk assessment
4. **🎯 Strategic Planning**: Complex problem solving using collaborative consensus reasoning
5. **🏥 Medical Research**: Clinical trial planning with probabilistic reasoning and risk mitigation
6. **🏭 Supply Chain**: Global logistics optimization using spatial reasoning and resource distribution

## 🛠️ CLI Commands Reference

### 📋 Component Listing Commands
- `feriq list components [--detailed]` - Overview of all framework components
- `feriq list roles [--format json|yaml|table] [--filter <type>]` - List role designer outputs
- `feriq list tasks [--status active|pending|completed] [--agent <name>]` - List task designer outputs
- `feriq list plans [--active-only] [--format json]` - List plan designer outputs
- `feriq list observations [--recent N] [--level info|warning|error]` - List plan observer outputs
- `feriq list agents [--status active|idle] [--role <role>]` - List agent configurations and progress
- `feriq list workflows [--status running|completed]` - List workflow orchestrator outputs
- `feriq list interactions [--pattern <name>] [--recent N]` - List choreographer outputs
- `feriq list reasoning [--type causal|probabilistic] [--recent N]` - List reasoner outputs
- `feriq list actions [--component <name>] [--recent N]` - List cross-component actions

### 🧠 Reasoning & Planning Commands
- `feriq plan strategies` - List available reasoning-enhanced planning strategies
- `feriq plan create --goal <desc> --strategy <strategy> --type <type>` - Create intelligent plan
- `feriq plan analyze --goal <desc> --context <factor>` - Analyze planning requirements
- `feriq plan demo [--type software|medical|supply-chain|all]` - Run planning demonstrations

### 🎭 Sample Data & Demos
- `feriq list generate-samples [--confirm]` - Generate demonstration data for all components
- `feriq plan demo --type all` - Run all reasoning-enhanced planning demos

### Traditional Project Commands
- `feriq init project --name <name> [--model-setup]` - Initialize new project with optional model setup
- `feriq status show` - Display project status and resources
- `feriq agent list` - List project agents (traditional)
- `feriq agent create <name>` - Create new agent interactively
- `feriq goal list` - List project goals (traditional)
- `feriq goal create <name>` - Create new goal interactively
- `feriq workflow list` - List project workflows (traditional)
- `feriq workflow run <name>` - Execute specific workflow

### Model Operations
- `feriq model list` - Show available models (Ollama, OpenAI, Anthropic)
- `feriq model test <provider> <model>` - Test model functionality with interactive prompts
- `feriq model setup` - Interactive model configuration wizard
- `feriq model pull <model>` - Pull Ollama model locally

### Utility Commands
- `feriq --help` - Show help and available commands
- `feriq <command> --help` - Get detailed help for specific commands
- `feriq interactive start` - Start interactive terminal mode

## 🏗️ Architecture

The Feriq framework features a sophisticated architecture with 8 core components working in harmony:

### 🏛️ Core Framework Components
```
┌─────────────────────────────────────────────────────────────┐
│                    🏗️ Feriq Framework                      │
├─────────────────────────────────────────────────────────────┤
│ 🎭 Role Designer    │ 📋 Task Designer    │ 📊 Plan Designer  │
│ 👁️ Plan Observer    │ 🎯 Agent System     │ 🎼 Orchestrator   │
│ 💃 Choreographer    │ 🧠 Reasoner         │                   │
├─────────────────────────────────────────────────────────────┤
│                 📋 Comprehensive CLI System                  │
│          🖥️ List Commands  │  🧠 Planning  │  🔧 Management   │
└─────────────────────────────────────────────────────────────┘
```

### Component Integration
- **🎭 Dynamic Role Designer**: Creates roles → assigns to agents → tracks in outputs
- **📋 Task Designer & Allocator**: Breaks down goals → assigns tasks → monitors allocation efficiency
- **📊 Plan Designer**: Creates execution plans → uses reasoning enhancement → optimizes resources
- **👁️ Plan Observer**: Monitors execution → generates alerts → tracks performance metrics
- **🎯 Goal-Oriented Agents**: Execute tasks → learn and adapt → report progress
- **🎼 Workflow Orchestrator**: Coordinates execution → manages resources → logs coordination
- **💃 Choreographer**: Manages interactions → tracks communication → optimizes patterns
- **🧠 Reasoner**: Provides intelligence → enhances planning → generates recommendations

### CLI Architecture
- **📋 Listing System**: Comprehensive output tracking for all 8 components with filtering
- **🧠 Reasoning Integration**: CLI commands for reasoning-enhanced planning and analysis
- **🔧 Model Management**: Native Ollama support with OpenAI/Anthropic compatibility
- **📁 Project Management**: YAML-based configuration with intelligent templates
- **📊 Real-time Monitoring**: Performance metrics, alerts, and cross-component activity tracking

```
┌─────────────────────────────────────────────────────────────┐
│                     Feriq CLI Interface                     │
├─────────────────────────────────────────────────────────────┤
│  Project Mgmt    │  Agent Mgmt     │  Model Integration     │
│  Status Monitor  │  Goal Tracking  │  Workflow Execution    │
├─────────────────────────────────────────────────────────────┤
│                    Feriq Framework Core                     │
├─────────────────────────────────────────────────────────────┤
│  Dynamic Role Designer │ Task Designer & Allocator         │
│  Plan Designer        │ Plan Observer                      │
│  Workflow Orchestrator │ Choreographer                     │
│  Goal-Oriented Agents  │ Reasoner                          │
├─────────────────────────────────────────────────────────────┤
│                    CrewAI Foundation                        │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

Feriq can be configured through YAML files or environment variables:

### Project Configuration (feriq.yaml)
```yaml
# Project settings
name: my-ai-project
version: 0.1.0
template: basic

# Model configuration
models:
  default:
    provider: ollama
    model: llama3.1:8b
    
# Agent settings
agents:
  default_capabilities: ["research", "analysis", "writing"]
  max_concurrent: 5
```

### Framework Configuration (config.yaml)
```yaml
### 🎯 Key Features Summary
- **8 Core Components**: Complete framework with role design, task allocation, planning, observation, agents, orchestration, choreography, and reasoning
- **Comprehensive CLI**: 50+ commands for listing, monitoring, planning, and managing multi-agent workflows
- **Reasoning-Enhanced Planning**: 7 intelligent planning strategies using advanced reasoning engines
- **Real-time Monitoring**: Track execution logs, performance metrics, alerts, and cross-component activities
- **Multiple Output Formats**: Table, JSON, and YAML formats with flexible filtering capabilities
- **Professional Integration**: Ollama model support with OpenAI/Anthropic compatibility

## ⚙️ Configuration

Create a `feriq.yaml` configuration file in your project:

```yaml
# Project configuration
project:
  name: "my-feriq-project"
  version: "1.0.0"

# Framework settings
framework:
  components:
    role_designer: enabled
    task_designer: enabled
    plan_designer: enabled
    plan_observer: enabled
    workflow_orchestrator: enabled
    choreographer: enabled
    reasoner: enabled

# Model configuration
models:
  default_provider: "ollama"
  ollama:
    host: "http://localhost:11434"
    model: "llama3.1:8b"
  openai:
    api_key: "${OPENAI_API_KEY}"
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"

# CLI settings
cli:
  auto_save: true
  verbose_output: false
  default_format: "table"
  
# Reasoning configuration
reasoning:
  default_strategy: "hybrid_intelligent"
  confidence_threshold: 0.7
  max_reasoning_depth: 5

# Output settings
outputs:
  directory: "outputs"
  retention_days: 30
  compression: true

# Monitoring settings
monitoring:
  real_time_alerts: true
  performance_tracking: true
  log_level: "INFO"
```

## 🧪 Testing & Validation

### CLI Testing
```bash
# Test all CLI commands
python -m feriq.cli.main --help

# Test component listing
python -m feriq.cli.main list components --detailed

# Test sample data generation
python -m feriq.cli.main list generate-samples --confirm

# Test reasoning planning
python -m feriq.cli.main plan demo --type all

# Test model integration
python -m feriq.cli.main model list
python -m feriq.cli.main model test ollama llama3.1:8b
```

### Framework Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific component tests
python -m pytest tests/cli/ -v
python -m pytest tests/reasoning/ -v
python -m pytest tests/components/ -v

# Test with coverage
python -m pytest --cov=feriq tests/

# Run integration tests
python -m pytest tests/integration/ -v
```

### Demo Testing
```bash
# Run comprehensive demos
python -m feriq.demos.sample_output_generator
python -m feriq.demos.intelligent_planning_demo

# Test reasoning integration
python -m feriq.examples.reasoning_integration
python -m feriq.examples.reasoning_planning_examples
```

## 📚 Documentation

### Quick Access
- **[📋 Documentation Index](docs/README.md)** - Complete documentation navigation
- **[🚀 Quick Start Guide](docs/quick_start.md)** - Get started in 5 minutes
- **[💾 Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[💻 CLI Listing Guide](docs/cli_listing_guide.md)** - Comprehensive CLI capabilities

### Essential Guides
- **[🏗️ Architecture Overview](docs/architecture.md)** - System design and component integration
- **[🧠 Reasoning Usage Guide](docs/reasoning_usage.md)** - Advanced reasoning capabilities
- **[📊 Reasoning Planning Guide](docs/reasoning_planning_guide.md)** - Intelligent planning with reasoning
- **[🎯 Programming Guide](docs/programming_guide.md)** - Framework development reference

### Complete Documentation Suite
- **[CLI User Guide](docs/cli_guide.md)** - Complete CLI command reference
- **[Model Integration](docs/models.md)** - LLM model setup and configuration
- **[Examples Directory](examples/)** - Working examples and tutorials

### Getting Started (30 seconds)
1. **Install**: `pip install -r requirements.txt`
2. **Test CLI**: `python -m feriq.cli.main --help`
3. **Generate Samples**: `python -m feriq.cli.main list generate-samples --confirm`
4. **Explore Components**: `python -m feriq.cli.main list components --detailed`
5. **Try Reasoning**: `python -m feriq.cli.main plan demo --type all`

## 🚀 What's New

### Recent Updates (October 2025)
- ✅ **Comprehensive CLI Listing System**: Complete output tracking for all 8 framework components
- ✅ **Reasoning-Enhanced Planning**: Intelligent planning with 7 reasoning strategies
- ✅ **Real-time Monitoring**: Performance metrics, alerts, and cross-component activity tracking
- ✅ **Advanced Reasoning Integration**: 10+ reasoning types with multi-agent collaboration
- ✅ **Sample Data Generation**: Professional demonstration capabilities
- ✅ **Multi-format Output**: Table, JSON, and YAML with flexible filtering
- ✅ **Production-Ready CLI**: 50+ commands for complete workflow management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the excellent [CrewAI](https://github.com/joaomdmoura/crewAI) framework
- CLI interface powered by [Click](https://click.palletsprojects.com/)
- Ollama integration for local LLM support
- Inspired by multi-agent systems research and collaborative AI principles

---

**Feriq** - Empowering collaborative AI agents to solve complex problems together! 🚀

## 🎯 Current Status

- ✅ **Core Framework**: Complete 8-component collaborative AI system
- ✅ **CLI Interface**: Full terminal interface with project management
- ✅ **Ollama Integration**: Native support for local LLM models
- ✅ **Project Management**: Initialize, configure, and track projects
- ✅ **Resource Management**: Agents, goals, and workflows management
- ✅ **Model Testing**: Comprehensive model testing and configuration
- 🔄 **Testing Suite**: Unit and integration tests (in progress)
- 📝 **Documentation**: Comprehensive guides and API docs (in progress)

## 🚀 Getting Started Today

1. **Install Ollama** (if not already installed):
   ```bash
   # Visit: https://ollama.ai/download
   # Pull a model: ollama pull llama3.1:8b
   ```

2. **Set up Feriq**:
   ```bash
   git clone https://github.com/yasir2000/feriq.git
   cd feriq
   pip install -r requirements.txt
   ```

3. **Create your first project**:
   ```bash
   python -m feriq.cli.main init project --name my-agents
   cd my-agents
   python -m feriq.cli.main model setup
   python -m feriq.cli.main status show
   ```

4. **Start building**:
   ```bash
   python -m feriq.cli.main agent create researcher
   python -m feriq.cli.main goal create analysis
   python -m feriq.cli.main agent list
   ```

Ready to revolutionize your AI workflows? Let's build something amazing together! 🤖✨
