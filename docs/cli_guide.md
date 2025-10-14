# Feriq CLI User Guide

Complete guide to using the Feriq command-line interface for managing collaborative AI agent workflows.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Getting Started](#getting-started)
3. [Project Management](#project-management)
4. [Agent Management](#agent-management)
5. [Goal Management](#goal-management)
6. [Workflow Management](#workflow-management)
7. [Model Management](#model-management)
8. [Configuration](#configuration)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Prerequisites

1. **Python 3.8+** - Ensure Python is installed and accessible
2. **Ollama** (optional but recommended) - For local LLM support
3. **Git** - For cloning the repository

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yasir2000/feriq.git
cd feriq

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
python -m feriq.cli.main --help
```

### Ollama Setup (Recommended)

```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
# Pull recommended models
ollama pull llama3.1:8b
ollama pull deepseek-r1:1.5b

# Verify Ollama is running
ollama list
```

## Getting Started

### Your First Project

```bash
# Create a new project
python -m feriq.cli.main init project --name my-first-project

# Navigate to project directory
cd my-first-project

# Check project status
python -m feriq.cli.main status show

# Configure a model (if Ollama is available)
python -m feriq.cli.main model setup
```

### Understanding Project Structure

When you create a project, Feriq creates the following structure:

```
my-first-project/
├── feriq.yaml              # Project configuration
├── agents/                 # Agent definitions
│   └── example_agent.yaml  # Example agent
├── goals/                  # Goal definitions
│   └── example_goal.yaml   # Example goal
├── workflows/              # Workflow definitions
│   └── example_workflow.yaml # Example workflow
└── outputs/                # Execution outputs (created on first run)
```

## Project Management

### Creating Projects

```bash
# Basic project creation
python -m feriq.cli.main init project --name research-project

# Create with specific template
python -m feriq.cli.main init project --name advanced-project --template advanced

# Create with model setup
python -m feriq.cli.main init project --name ai-project --model-setup
```

### Project Status

```bash
# Show comprehensive project status
python -m feriq.cli.main status show

# Example output:
# 🔸 Project Status
# Name: research-project
# Version: 0.1.0
# Template: basic
# 
# 🔸 Resources
# Agents: 3
# Goals: 2
# Workflows: 1
# 
# 🔸 Models
# default: ollama:llama3.1:8b
```

### Project Configuration

The `feriq.yaml` file contains your project configuration:

```yaml
# Example feriq.yaml
name: research-project
version: 0.1.0
template: basic
description: "AI-powered research automation"

models:
  default:
    provider: ollama
    model: llama3.1:8b
  
agents:
  max_concurrent: 3
  default_capabilities:
    - research
    - analysis
    - writing
```

## Agent Management

### Listing Agents

```bash
# List all agents in the project
python -m feriq.cli.main agent list

# Example output:
# 🔸 Project Agents
#   📋 research_agent (Research Specialist) - example_agent.yaml
#   📋 writer_agent (Content Writer) - writer.yaml
#   📋 analyst_agent (Data Analyst) - analyst.yaml
```

### Creating Agents

```bash
# Create a new agent interactively
python -m feriq.cli.main agent create researcher

# Follow the prompts:
# Agent role: Research Specialist
# Agent goal: Conduct comprehensive research on specified topics
# Agent backstory: Expert researcher with 10+ years of experience...
```

### Agent Configuration Example

```yaml
# agents/researcher.yaml
name: researcher
role: Research Specialist
goal: |
  Conduct comprehensive research on specified topics, gather relevant
  information from multiple sources, and provide detailed analysis.

backstory: |
  You are an experienced research specialist with expertise in gathering,
  analyzing, and synthesizing information from various sources. You have
  a keen eye for detail and can identify credible sources.

capabilities:
  - information_gathering
  - source_verification
  - data_analysis
  - report_generation

tools:
  - web_search
  - document_analysis
  - citation_management

model:
  provider: ollama
  name: llama3.1:8b
```

### Running Agents

```bash
# Run an agent with a specific goal
python -m feriq.cli.main agent run researcher --goal "research_market_trends"

# Run with custom parameters
python -m feriq.cli.main agent run researcher --goal custom_research --verbose
```

## Goal Management

### Listing Goals

```bash
# List all project goals
python -m feriq.cli.main goal list

# Example output:
# 🔸 Project Goals
# ┌─────────────────┬─────────────────────────┬──────────┬─────────────────┐
# │ Name            │ Title                   │ Priority │ File            │
# ├─────────────────┼─────────────────────────┼──────────┼─────────────────┤
# │ market_research │ Market Analysis Study   │ high     │ market.yaml     │
# │ content_creation│ Content Generation      │ medium   │ content.yaml    │
# └─────────────────┴─────────────────────────┴──────────┴─────────────────┘
```

### Creating Goals

```bash
# Create a new goal interactively
python -m feriq.cli.main goal create market_analysis

# Follow the prompts:
# Goal title: Comprehensive Market Analysis
# Goal description: Analyze market trends, competitors, and opportunities
# Priority [medium]: high
```

### Goal Configuration Example

```yaml
# goals/market_analysis.yaml
name: market_analysis
title: Comprehensive Market Analysis
description: |
  Conduct a thorough analysis of the target market, including
  competitor analysis, trend identification, and opportunity assessment.

objectives:
  - Identify key market players and their strategies
  - Analyze market trends and growth patterns
  - Assess market opportunities and threats
  - Generate actionable insights and recommendations

success_criteria:
  - Complete competitor profile for top 10 players
  - Market trend analysis with supporting data
  - Opportunity assessment with risk analysis
  - Executive summary with strategic recommendations

priority: high
estimated_duration: "4 hours"
required_capabilities:
  - research
  - analysis
  - strategic_thinking
```

## Workflow Management

### Listing Workflows

```bash
# List all workflows
python -m feriq.cli.main workflow list

# Example output:
# 🔸 Project Workflows
# ┌─────────────────┬─────────────────────────┬────────┬─────────────────┐
# │ Name            │ Title                   │ Stages │ File            │
# ├─────────────────┼─────────────────────────┼────────┼─────────────────┤
# │ research_flow   │ Research Workflow       │ 4      │ research.yaml   │
# │ content_flow    │ Content Creation Flow   │ 3      │ content.yaml    │
# └─────────────────┴─────────────────────────┴────────┴─────────────────┘
```

### Running Workflows

```bash
# Execute a workflow
python -m feriq.cli.main workflow run research_flow

# Run with monitoring
python -m feriq.cli.main workflow run research_flow --monitor

# Run with custom parameters
python -m feriq.cli.main workflow run research_flow --params config.json
```

### Workflow Configuration Example

```yaml
# workflows/research_flow.yaml
name: research_flow
title: Research Workflow
description: Automated research workflow with analysis and reporting

stages:
  - name: planning
    description: Plan research approach and identify sources
    agents: [researcher]
    tasks:
      - define_research_scope
      - identify_sources
      - create_research_plan
    
  - name: data_collection
    description: Gather information from identified sources
    agents: [researcher, data_collector]
    tasks:
      - collect_primary_data
      - gather_secondary_sources
      - verify_information
    
  - name: analysis
    description: Analyze collected data and identify patterns
    agents: [analyst, researcher]
    tasks:
      - data_analysis
      - pattern_identification
      - insight_generation
    
  - name: reporting
    description: Generate comprehensive research report
    agents: [writer, analyst]
    tasks:
      - draft_report
      - review_findings
      - finalize_document

dependencies:
  - planning -> data_collection
  - data_collection -> analysis
  - analysis -> reporting

timeout: "2 hours"
max_retries: 3
```

## Model Management

### Listing Available Models

```bash
# List all available models
python -m feriq.cli.main model list

# Example output:
# ℹ️  
# OLLAMA Models:
#   ✅ deepseek-r1:1.5b
#   ✅ llama3.1:8b
#   ✅ codellama:7b
# ⚠️  
# OPENAI: Not available or no models
# ⚠️  
# ANTHROPIC: Not available or no models
```

### Testing Models

```bash
# Test a specific model
python -m feriq.cli.main model test ollama llama3.1:8b

# Test with custom prompt
python -m feriq.cli.main model test ollama deepseek-r1:1.5b --prompt "Explain quantum computing"

# Benchmark multiple models
python -m feriq.cli.main model test ollama llama3.1:8b --benchmark
```

### Model Setup and Configuration

```bash
# Interactive model setup
python -m feriq.cli.main model setup

# Set default model for project
python -m feriq.cli.main model set-default ollama llama3.1:8b

# Configure model parameters
python -m feriq.cli.main model configure ollama llama3.1:8b --temperature 0.7 --max-tokens 2048
```

### Pulling New Models

```bash
# Pull a new Ollama model
python -m feriq.cli.main model pull llama3.2:3b

# Pull and set as default
python -m feriq.cli.main model pull llama3.2:3b --set-default

# List available models to pull
python -m feriq.cli.main model available --provider ollama
```

## Configuration

### Global Configuration

Create a global configuration file at `~/.feriq/config.yaml`:

```yaml
# Global Feriq configuration
default_provider: ollama
default_model: llama3.1:8b

cli:
  verbose: false
  auto_save: true
  confirm_destructive: true

logging:
  level: INFO
  file: ~/.feriq/logs/feriq.log

models:
  ollama:
    base_url: http://localhost:11434
    timeout: 30
  
  openai:
    api_key: ${OPENAI_API_KEY}
    organization: ${OPENAI_ORG_ID}
  
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

### Environment Variables

```bash
# Model API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Ollama configuration
export OLLAMA_HOST="http://localhost:11434"

# Feriq settings
export FERIQ_LOG_LEVEL="DEBUG"
export FERIQ_CONFIG_PATH="/path/to/config.yaml"
```

### Project-Level Configuration

Each project can override global settings in its `feriq.yaml`:

```yaml
name: my-project
version: 1.0.0

# Override global model settings
models:
  default:
    provider: ollama
    model: deepseek-r1:1.5b
    temperature: 0.8
    max_tokens: 4096

# Project-specific settings
settings:
  max_concurrent_agents: 5
  auto_retry: true
  save_intermediate_results: true

# Custom capabilities
capabilities:
  research:
    tools: [web_search, document_analysis]
    max_sources: 10
  
  analysis:
    tools: [data_analysis, visualization]
    output_format: markdown
```

## Advanced Usage

### Interactive Mode

```bash
# Start interactive mode
python -m feriq.cli.main interactive start

# Available in interactive mode:
# feriq> status
# feriq> agent list
# feriq> goal create research_task
# feriq> workflow run analysis_flow
# feriq> help
# feriq> exit
```

### Batch Operations

```bash
# Run multiple commands from file
python -m feriq.cli.main batch commands.txt

# Example commands.txt:
# agent create researcher1
# agent create analyst1  
# goal create market_study
# workflow run research_flow
```

### Custom Templates

```bash
# Create project with custom template
python -m feriq.cli.main init project --name custom-project --template /path/to/template

# List available templates
python -m feriq.cli.main templates list

# Create new template
python -m feriq.cli.main templates create my-template
```

### Debugging and Monitoring

```bash
# Enable verbose output
python -m feriq.cli.main --verbose status show

# Enable debug mode
python -m feriq.cli.main --debug workflow run research_flow

# Monitor workflow execution
python -m feriq.cli.main workflow monitor research_flow --real-time

# View execution logs
python -m feriq.cli.main logs show --workflow research_flow --last 100
```

## Troubleshooting

### Common Issues

#### 1. Model Not Found

```bash
# Error: Model 'llama3.1:8b' not found
# Solution: Pull the model first
ollama pull llama3.1:8b
python -m feriq.cli.main model list  # Verify it's available
```

#### 2. Project Not Found

```bash
# Error: Not in a Feriq project directory
# Solution: Navigate to project or create new one
cd my-project
# OR
python -m feriq.cli.main init project --name new-project
```

#### 3. Connection Issues

```bash
# Error: Connection refused to Ollama
# Solution: Ensure Ollama is running
ollama serve  # Start Ollama server
# OR check if running on different port
python -m feriq.cli.main model test ollama llama3.1:8b --host http://localhost:11434
```

#### 4. Permission Errors

```bash
# Error: Permission denied writing to directory
# Solution: Check permissions or use different directory
chmod 755 /path/to/project
# OR
python -m feriq.cli.main init project --name project --path ~/projects/
```

### Diagnostic Commands

```bash
# Check system status
python -m feriq.cli.main doctor

# Verify installation
python -m feriq.cli.main version --verbose

# Test model connections
python -m feriq.cli.main model test-all

# Validate project configuration
python -m feriq.cli.main config validate
```

### Getting Help

```bash
# General help
python -m feriq.cli.main --help

# Command-specific help
python -m feriq.cli.main agent --help
python -m feriq.cli.main workflow run --help

# List all commands
python -m feriq.cli.main commands list

# Show examples for a command
python -m feriq.cli.main examples agent create
```

### Log Files

Default log locations:
- **Linux/Mac**: `~/.feriq/logs/feriq.log`
- **Windows**: `%USERPROFILE%\.feriq\logs\feriq.log`
- **Project logs**: `<project>/logs/`

```bash
# View recent logs
tail -f ~/.feriq/logs/feriq.log

# View project-specific logs
python -m feriq.cli.main logs show --project --last 50
```

## Best Practices

1. **Always test models** before using in workflows
2. **Use version control** for your Feriq projects
3. **Configure appropriate timeouts** for long-running workflows
4. **Monitor resource usage** when running multiple agents
5. **Keep configurations modular** using separate files for different environments
6. **Use descriptive names** for agents, goals, and workflows
7. **Document custom capabilities** and tools
8. **Regular backups** of project configurations
9. **Test workflows** with small datasets first
10. **Use monitoring** for production workflows

## Additional Resources

- [Programming Guide](programming_guide.md) - Framework programming guide
- [Architecture Overview](architecture.md) - Detailed architecture documentation
- [Model Integration](models.md) - LLM model setup and configuration
- [Examples](../examples/) - Working examples and tutorials
- [API Reference](api_reference.md) - Complete API documentation

---

*This guide covers the core CLI functionality. For advanced programming with the framework, see the [Programming Guide](programming_guide.md).*