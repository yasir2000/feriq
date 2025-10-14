# Feriq CLI Documentation

The Feriq CLI provides a comprehensive command-line interface for interacting with the Feriq collaborative AI agents framework. It enables terminal-based management of agents, goals, workflows, and model configurations.

## Installation

### From Source
```bash
# Clone and install
git clone https://github.com/yasir2000/feriq.git
cd feriq
pip install -e .

# Verify installation
feriq --help
```

### Development Installation
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with all model providers
pip install -e ".[all]"
```

## Quick Start

### 1. Initialize a New Project
```bash
# Create a new Feriq project
feriq init project my-ai-project

# Initialize with model setup
feriq init project my-ai-project --model-setup
```

### 2. Configure Models
```bash
# Setup model configuration interactively
feriq model setup

# List available models
feriq model list

# Pull an Ollama model
feriq model pull llama2

# Test a specific model
feriq model test ollama llama2
```

### 3. Create Agents
```bash
# Create a new agent interactively
feriq agent create research_agent

# List all agents
feriq agent list

# Run an agent
feriq agent run research_agent --goal "research market trends"
```

### 4. Manage Goals
```bash
# Create a new goal
feriq goal create market_research

# List all goals
feriq goal list
```

### 5. Workflow Management
```bash
# List workflows
feriq workflow list

# Run a workflow
feriq workflow run research_workflow
```

## Commands Reference

### Global Options
```bash
feriq [GLOBAL_OPTIONS] COMMAND [ARGS]

Global Options:
  -c, --config PATH    Configuration file path
  -v, --verbose        Enable verbose output
  -d, --debug         Enable debug mode
  --help              Show help message
  --version           Show version
```

### Core Commands

#### `feriq version`
Display version and component information.

```bash
# Table format (default)
feriq version

# JSON format
feriq version --output json

# YAML format
feriq version --output yaml
```

#### `feriq config`
Generate or manage configuration files.

```bash
# Generate config template
feriq config

# Save to file
feriq config --output feriq.yaml --format yaml

# JSON format
feriq config --format json
```

#### `feriq doctor`
Run diagnostic checks on the installation.

```bash
# Run all checks
feriq doctor

# Check specific components
feriq doctor --check-models
feriq doctor --check-components
feriq doctor --check-dependencies
```

### Project Management

#### `feriq init`
Initialize new projects and components.

```bash
# Create new project
feriq init project <name> [OPTIONS]

Options:
  --name TEXT                 Project name
  --template [basic|advanced|research]  Project template
  --model-setup              Setup model configuration during init
```

### Model Management

#### `feriq model`
Manage LLM models and providers.

```bash
# Interactive model setup
feriq model setup

# List all available models
feriq model list

# Pull Ollama model
feriq model pull <model_name>

# Test specific model
feriq model test <provider> <model_name>
```

**Supported Providers:**
- **Ollama**: Local models (`ollama`)
- **OpenAI**: GPT models (`openai`)
- **Anthropic**: Claude models (`anthropic`)

### Agent Management

#### `feriq agent`
Create and manage AI agents.

```bash
# List all agents
feriq agent list

# Create new agent
feriq agent create <name>

# Run agent with goal
feriq agent run <name> --goal "<goal_description>"
```

### Goal Management

#### `feriq goal`
Define and track project goals.

```bash
# List all goals
feriq goal list

# Create new goal
feriq goal create <name>
```

### Workflow Management

#### `feriq workflow`
Orchestrate multi-agent workflows.

```bash
# List workflows
feriq workflow list

# Run workflow
feriq workflow run <name>
```

### Status and Monitoring

#### `feriq status`
Show project and system status.

```bash
# Show project status
feriq status show
```

### Interactive Mode

#### `feriq interactive`
Start interactive terminal mode.

```bash
# Start interactive mode
feriq interactive start
```

**Interactive Commands:**
- `help` - Show help
- `status` - Show project status
- `agents list` - List agents
- `goals list` - List goals
- `workflows list` - List workflows
- `models list` - List models
- `run agent <name>` - Run agent
- `run workflow <name>` - Run workflow
- `chat` - Start chat mode
- `exit` - Exit interactive mode

## Configuration

### Project Configuration (`feriq.yaml`)
```yaml
name: my-project
version: 0.1.0
template: basic

framework:
  version: 1.0.0
  features:
    - role_designer
    - task_designer
    - plan_designer
    - plan_observer
    - workflow_executor
    - reasoner

models:
  default:
    provider: ollama
    model: llama2
    config:
      temperature: 0.7
      max_tokens: 2048

agents: {}
goals: {}
workflows: {}
```

### Environment Variables
```bash
# Model Provider APIs
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Ollama Configuration
export OLLAMA_HOST="http://localhost:11434"

# Framework Settings
export FERIQ_LOG_LEVEL="INFO"
export FERIQ_DEBUG="false"
```

## Project Structure

After initializing a project, you'll have:

```
my-project/
├── feriq.yaml              # Project configuration
├── agents/                 # Agent definitions
│   ├── .gitkeep
│   └── example_agent.yaml
├── goals/                  # Goal definitions
│   ├── .gitkeep
│   └── example_goal.yaml
├── workflows/              # Workflow definitions
│   ├── .gitkeep
│   └── example_workflow.yaml
├── plans/                  # Generated plans
│   └── .gitkeep
├── logs/                   # Application logs
│   └── .gitkeep
├── data/                   # Data files
│   └── .gitkeep
└── configs/                # Additional configurations
    └── .gitkeep
```

## Examples

### Basic Usage Flow
```bash
# 1. Create project
feriq init project my-research --model-setup

# 2. Create agent
cd my-research
feriq agent create researcher

# 3. Create goal
feriq goal create market_analysis

# 4. Check status
feriq status show

# 5. Start interactive mode
feriq interactive start
```

### Model Configuration
```bash
# Setup Ollama
feriq model pull llama2
feriq model test ollama llama2

# Setup with OpenAI (requires API key)
export OPENAI_API_KEY="sk-..."
feriq model test openai gpt-3.5-turbo

# List all available models
feriq model list
```

### Workflow Execution
```bash
# Create workflow
feriq workflow create research_pipeline

# Run workflow
feriq workflow run research_pipeline

# Monitor in real-time
feriq status show --watch
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Check installation
   feriq doctor

   # Reinstall
   pip uninstall feriq
   pip install -e .
   ```

2. **Model Connection Issues**
   ```bash
   # Check Ollama
   feriq doctor --check-models
   
   # Test specific model
   feriq model test ollama llama2
   ```

3. **Project Not Found**
   ```bash
   # Ensure you're in a project directory
   feriq status show
   
   # Or initialize new project
   feriq init project my-project
   ```

### Debug Mode
```bash
# Enable debug output
feriq --debug command

# Verbose output
feriq --verbose command
```

### Getting Help
```bash
# General help
feriq --help

# Command-specific help
feriq model --help
feriq agent create --help

# Interactive help
feriq interactive start
> help
```

## Integration

### Python API Integration
```python
from feriq.cli.utils import CliContext
from feriq.cli.models import ModelManager

# Use CLI utilities in Python
context = CliContext()
framework = context.get_framework()

# Manage models programmatically
model_manager = ModelManager()
models = model_manager.list_all_models()
```

### Scripting
```bash
#!/bin/bash
# Automated project setup script

# Create project
feriq init project auto-research --template advanced

# Setup models
feriq model pull llama2
feriq model setup

# Create components
feriq agent create researcher
feriq goal create analysis
feriq workflow create pipeline

# Run
feriq workflow run pipeline
```

This CLI provides a powerful interface for managing your Feriq collaborative AI agents framework, from initial setup to complex workflow execution.