


# Feriq - Collaborative AI Agents Framework

Feriq is a comprehensive collaborative AI agents framework built on CrewAI that provides advanced multi-agent coordination, dynamic role assignment, intelligent task orchestration, and sophisticated reasoning capabilities. Now featuring a complete CLI interface with Ollama model integration for terminal-based workflow management.

## 🌟 Features

### Core Capabilities
- **🎭 Dynamic Role Designer**: Automatically creates and assigns roles based on task requirements and context
- **📋 Task Designer & Allocator**: Intelligently breaks down goals into tasks and optimally assigns them to agents
- **📊 Plan Designer**: Creates comprehensive execution plans with resource allocation and timeline management
- **👁️ Plan Observer**: Real-time monitoring of plan execution with alerts and performance metrics
- **🎯 Goal-Oriented Agents**: Intelligent agents that work towards specific goals with learning and adaptation
- **🎼 Workflow Orchestrator**: Central coordinator for workflow execution and resource management
- **💃 Choreographer**: Manages agent interactions, coordination patterns, and communication protocols
- **🧠 Reasoner**: Advanced reasoning engine for decision-making, problem-solving, and strategic planning

### CLI Interface
- **🖥️ Terminal Interface**: Complete command-line interface for all framework operations
- **🤖 Ollama Integration**: Seamless integration with Ollama models (llama3.1, deepseek-r1, etc.)
- **📁 Project Management**: Initialize, configure, and manage multi-agent projects
- **📊 Status Monitoring**: Real-time project status and resource tracking
- **🔧 Model Management**: Test, configure, and switch between different LLM models

### Advanced Features
- **Multi-Strategy Reasoning**: Deductive, inductive, probabilistic, and causal reasoning
- **Coordination Patterns**: Pipeline, scatter-gather, consensus, broadcast, and hierarchical coordination
- **Real-time Monitoring**: Performance metrics, bottleneck detection, and adaptive optimization
- **Knowledge Management**: Persistent knowledge base with learning from experience
- **Conflict Resolution**: Intelligent conflict detection and resolution mechanisms
- **Resource Management**: Dynamic resource allocation and constraint handling

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

Feriq provides a comprehensive command-line interface for managing multi-agent workflows:

### Initialize a New Project
```bash
# Create a new Feriq project
python -m feriq.cli.main init project --name my-ai-project

# Navigate to project directory
cd my-ai-project
```

### Model Management
```bash
# List available models (requires Ollama)
python -m feriq.cli.main model list

# Test a specific model
python -m feriq.cli.main model test ollama llama3.1:8b

# Setup model configuration
python -m feriq.cli.main model setup
```

### Project Management
```bash
# Show project status
python -m feriq.cli.main status show

# List agents
python -m feriq.cli.main agent list

# List goals
python -m feriq.cli.main goal list

# List workflows
python -m feriq.cli.main workflow list
```

### Creating Resources
```bash
# Create a new agent
python -m feriq.cli.main agent create research-agent

# Create a new goal
python -m feriq.cli.main goal create market-research

# Get help for any command
python -m feriq.cli.main --help
```

## 🏃‍♂️ Programming Quick Start

### Simple Example
```python
from feriq import FeriqFramework, FeriqAgent, Goal, GoalType, Role, RoleCapability
from datetime import timedelta

# Initialize framework
framework = FeriqFramework()

# Create a goal
goal = Goal(
    name="Research AI Trends",
    description="Conduct comprehensive research on latest AI trends",
    goal_type=GoalType.RESEARCH,
    required_capabilities=["research", "analysis", "writing"],
    estimated_duration=timedelta(days=3)
)

# Create a role
researcher_role = Role(
    name="AI Researcher",
    description="Specialist in AI research and analysis",
    capabilities=[
        RoleCapability("research", 0.9),
        RoleCapability("analysis", 0.8),
        RoleCapability("writing", 0.7)
    ]
)

# Create an agent
agent = FeriqAgent(
    name="ResearchBot",
    role=researcher_role,
    capabilities=["research", "analysis", "writing"]
)

## 📖 Examples

### CLI Examples
```bash
# Complete workflow example
cd examples
python -m feriq.cli.main init project --name research-project
cd research-project
python -m feriq.cli.main model setup  # Configure Ollama model
python -m feriq.cli.main status show  # Check project status
```

### Programming Examples
```bash
# Simple example
cd examples
python simple_example.py

# Comprehensive example
python comprehensive_example.py
```

### Example Scenarios
1. **Research Project**: Multi-agent research coordination with CLI management
2. **Software Development**: End-to-end development workflow with terminal interface
3. **Data Analysis**: Collaborative data processing pipeline with model testing
4. **Problem Solving**: Complex problem decomposition using CLI tools

## 🛠️ CLI Commands Reference

### Project Commands
- `feriq init project --name <name>` - Initialize new project
- `feriq status show` - Display project status and resources

### Agent Management
- `feriq agent list` - List all project agents
- `feriq agent create <name>` - Create new agent interactively
- `feriq agent run <name> --goal <goal>` - Execute agent with specific goal

### Goal Management  
- `feriq goal list` - List all project goals
- `feriq goal create <name>` - Create new goal interactively

### Workflow Management
- `feriq workflow list` - List all project workflows  
- `feriq workflow run <name>` - Execute specific workflow

### Model Operations
- `feriq model list` - Show available models (Ollama, OpenAI, Anthropic)
- `feriq model test <provider> <model>` - Test model functionality
- `feriq model setup` - Interactive model configuration
- `feriq model pull <model>` - Pull Ollama model

### Utility Commands
- `feriq --help` - Show help and available commands
- `feriq <command> --help` - Get help for specific commands
- `feriq interactive start` - Start interactive mode

## 🏗️ Architecture

The Feriq framework consists of interconnected components that work together to enable sophisticated multi-agent workflows:

### Core Components
- **Framework Core**: Central coordination and management
- **Agent System**: Enhanced AI agents with learning and collaboration
- **Task System**: Intelligent task decomposition and execution
- **Planning System**: Advanced planning and optimization
- **Monitoring System**: Real-time observation and adaptation
- **Reasoning System**: Intelligent decision-making and problem-solving

### CLI Architecture
- **Command Interface**: Click-based terminal interface with intuitive commands
- **Model Integration**: Native Ollama support with OpenAI/Anthropic compatibility
- **Project Management**: YAML-based configuration with template system
- **Resource Tracking**: Real-time monitoring of agents, goals, and workflows

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
# Logging configuration
logging:
  level: INFO
  structured: true

# Orchestrator settings
orchestrator:
  max_concurrent_workflows: 5
  default_strategy: "dynamic"

# Reasoner configuration
reasoner:
  default_reasoning_type: "probabilistic"
  confidence_threshold: 0.7

# CLI settings
cli:
  auto_save: true
  verbose_output: false
```

## 🧪 Testing

Run the test suite to ensure everything is working:

```bash
# Run all tests
python -m pytest tests/

# Run CLI tests specifically
python -m pytest tests/cli/

# Test with coverage
python -m pytest --cov=feriq tests/
```

## 📚 Documentation

- [CLI User Guide](docs/cli_guide.md) - Complete CLI documentation
- [Programming Guide](docs/programming_guide.md) - Framework programming guide
- [Architecture Overview](docs/architecture.md) - Detailed architecture documentation
- [Model Integration](docs/models.md) - LLM model setup and configuration
- [Examples](examples/) - Working examples and tutorials

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