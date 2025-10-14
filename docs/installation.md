# Installation Guide - Feriq Framework

Complete installation and setup guide for the Feriq Collaborative AI Agents Framework.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8 or higher (3.9+ recommended)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 1GB free space
- **Network**: Internet connection for model downloads

### Optional Requirements

- **GPU**: NVIDIA GPU with CUDA for accelerated model inference
- **Docker**: For containerized deployment
- **Git**: For development and version control

---

## 🚀 Quick Installation

### Method 1: Basic Setup (Recommended)

```bash
# 1. Navigate to feriq directory
cd /path/to/feriq

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Verify installation
python -m feriq.cli.main --help

# 4. Generate sample data
python -m feriq.cli.main list generate-samples --confirm

# 5. Test framework
python -m feriq.cli.main list components
```

### Method 2: Development Setup

```bash
# 1. Clone repository (if needed)
git clone <repository-url>
cd feriq

# 2. Create virtual environment
python -m venv feriq_env
source feriq_env/bin/activate  # On Windows: feriq_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# 4. Install in development mode
pip install -e .

# 5. Verify installation
python -m feriq.cli.main --version
```

---

## 📦 Detailed Installation Steps

### Step 1: Python Environment Setup

#### Windows

```bash
# Check Python version
python --version

# Create virtual environment
python -m venv feriq_env

# Activate virtual environment
feriq_env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### macOS/Linux

```bash
# Check Python version
python3 --version

# Create virtual environment
python3 -m venv feriq_env

# Activate virtual environment
source feriq_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2: Core Dependencies

```bash
# Install core framework dependencies
pip install click>=8.0.0
pip install pydantic>=2.0.0
pip install requests>=2.28.0
pip install python-dateutil>=2.8.0
pip install typing-extensions>=4.0.0

# Install CLI dependencies
pip install rich>=12.0.0
pip install tabulate>=0.9.0

# Install optional dependencies
pip install numpy>=1.21.0
pip install pandas>=1.3.0
```

### Step 3: Verify Installation

```bash
# Test framework import
python -c "from feriq.components.role_designer import RoleDesigner; print('✅ Framework imported successfully')"

# Test CLI
python -m feriq.cli.main --help

# Test component listing
python -m feriq.cli.main list components --detailed
```

---

## 🧠 Model Integration Setup

### Ollama Setup (Recommended)

#### Install Ollama

**Windows/macOS:**
1. Download from [ollama.ai](https://ollama.ai)
2. Run installer
3. Verify installation: `ollama --version`

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Install Models

```bash
# Install recommended models
ollama pull llama2
ollama pull codellama
ollama pull mistral

# Verify model installation
ollama list
```

#### Test Integration

```bash
# Test reasoning with Ollama
python -m feriq.cli.main plan demo --type analytical --model ollama

# Create reasoning-enhanced plan
python -m feriq.cli.main plan create "Test Project" \
  --use-reasoning \
  --model ollama \
  --reasoning-type strategic
```

### OpenAI Setup (Optional)

```bash
# Install OpenAI package
pip install openai>=1.0.0

# Set API key (choose one method)
export OPENAI_API_KEY="your-api-key-here"
# OR create .env file with OPENAI_API_KEY=your-api-key-here

# Test integration
python -m feriq.cli.main plan demo --type creative --model openai
```

### Azure OpenAI Setup (Optional)

```bash
# Install Azure OpenAI package
pip install azure-openai>=1.0.0

# Set environment variables
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_VERSION="2023-12-01-preview"

# Test integration
python -m feriq.cli.main plan demo --type analytical --model azure
```

---

## 🔧 Configuration Setup

### Basic Configuration

Create `config.yaml` in the feriq root directory:

```yaml
# config.yaml
framework:
  version: "1.0.0"
  debug: false
  output_directory: "outputs"

models:
  default: "ollama"
  ollama:
    base_url: "http://localhost:11434"
    model: "llama2"
    timeout: 30
  
  openai:
    model: "gpt-3.5-turbo"
    max_tokens: 1000
    temperature: 0.7
  
  azure:
    deployment_name: "gpt-35-turbo"
    max_tokens: 1000
    temperature: 0.7

components:
  role_designer:
    max_roles: 100
    output_format: "json"
  
  task_designer:
    max_tasks: 500
    priority_levels: ["low", "medium", "high", "critical"]
  
  plan_designer:
    max_phases: 20
    enable_reasoning: true
    default_reasoning_type: "strategic"

cli:
  max_list_items: 50
  default_format: "table"
  enable_colors: true
```

### Environment Variables

Create `.env` file (optional):

```bash
# .env file
FERIQ_DEBUG=false
FERIQ_OUTPUT_DIR=outputs
FERIQ_DEFAULT_MODEL=ollama

# Model API Keys (if using)
OPENAI_API_KEY=your-openai-key
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=your-azure-endpoint

# Logging
FERIQ_LOG_LEVEL=INFO
FERIQ_LOG_FILE=feriq.log
```

### Directory Structure Setup

```bash
# Create required directories
mkdir -p feriq/outputs/{roles,tasks,plans,observations,agents,workflows,choreographies,reasoning}
mkdir -p feriq/config
mkdir -p feriq/logs
mkdir -p feriq/cache

# Verify structure
python -m feriq.cli.main list outputs --show-structure
```

---

## 🧪 Verification & Testing

### Complete Installation Test

```bash
# Run comprehensive test suite
python -m feriq.cli.main test installation

# Generate sample data for all components
python -m feriq.cli.main list generate-samples --confirm --count 5

# Test all component types
python -m feriq.cli.main list roles
python -m feriq.cli.main list tasks
python -m feriq.cli.main list plans
python -m feriq.cli.main list observations
python -m feriq.cli.main list agents
python -m feriq.cli.main list workflows
python -m feriq.cli.main list choreographies
python -m feriq.cli.main list reasoning

# Test reasoning capabilities
python -m feriq.cli.main plan demo --type all

# Test planning with reasoning
python -m feriq.cli.main plan create "Installation Test Project" \
  --use-reasoning \
  --reasoning-type analytical
```

### Performance Validation

```bash
# Test framework performance
python -c "
import time
from feriq.components.role_designer import RoleDesigner

start = time.time()
designer = RoleDesigner()
role = designer.create_role('Test Role', ['responsibility1'], ['skill1'])
end = time.time()

print(f'✅ Role creation time: {end-start:.3f}s')
print(f'✅ Framework performance: OK')
"

# Test CLI performance
time python -m feriq.cli.main list components
```

### Integration Validation

```bash
# Test model integration (if Ollama installed)
python -m feriq.cli.main plan demo --type analytical --model ollama

# Test reasoning integration
python -m feriq.cli.main plan analyze "Test Analysis" --reasoning-type strategic

# Test output management
python -m feriq.cli.main list outputs --detailed --recent
```

---

## 🔄 Post-Installation Setup

### 1. Generate Initial Data

```bash
# Generate comprehensive sample data
python -m feriq.cli.main list generate-samples --confirm --count 10

# Create demo project
python -m feriq.cli.main init-project "Demo Project"

# Generate sample plans
python -m feriq.cli.main plan demo --type strategic
python -m feriq.cli.main plan demo --type analytical
python -m feriq.cli.main plan demo --type creative
```

### 2. Explore CLI Capabilities

```bash
# Explore all commands
python -m feriq.cli.main --help

# Test component listing
python -m feriq.cli.main list --help
python -m feriq.cli.main list components --detailed

# Test planning commands
python -m feriq.cli.main plan --help
python -m feriq.cli.main plan create "Test Plan" --use-reasoning
```

### 3. Review Documentation

```bash
# View documentation index
cat docs/README.md

# Quick start guide
cat docs/quick_start.md

# CLI listing guide
cat docs/cli_listing_guide.md
```

---

## 🐛 Troubleshooting

### Common Installation Issues

#### Python Import Errors

```bash
# Issue: ModuleNotFoundError
# Solution: Ensure you're in the correct directory and have dependencies installed

# Check current directory
pwd
ls -la  # Should see feriq/ directory

# Reinstall dependencies
pip install -r requirements.txt

# Test import
python -c "import feriq; print('✅ Import successful')"
```

#### CLI Command Not Found

```bash
# Issue: python -m feriq.cli.main not working
# Solution: Verify Python path and module structure

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check feriq module
python -c "import feriq.cli.main; print('✅ CLI module found')"

# Alternative execution
python feriq/cli/main.py --help
```

#### Model Integration Issues

```bash
# Issue: Ollama connection failed
# Solution: Verify Ollama installation and service

# Check Ollama status
ollama list

# Start Ollama service (if needed)
ollama serve

# Test model
ollama run llama2 "Hello, how are you?"

# Test in framework
python -m feriq.cli.main plan demo --type analytical --model ollama
```

#### Permission Issues

```bash
# Issue: Permission denied for output directory
# Solution: Check and fix permissions

# Check permissions
ls -la feriq/outputs/

# Fix permissions (Linux/macOS)
chmod -R 755 feriq/outputs/

# Create directories if missing
mkdir -p feriq/outputs/{roles,tasks,plans,observations,agents,workflows,choreographies,reasoning}
```

### Performance Issues

#### Slow Startup

```bash
# Check Python version (should be 3.9+)
python --version

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# Optimize imports
python -O -m feriq.cli.main list components
```

#### Memory Usage

```bash
# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Reduce sample generation count if needed
python -m feriq.cli.main list generate-samples --confirm --count 3
```

---

## 🔄 Upgrade & Maintenance

### Updating Dependencies

```bash
# Update pip
pip install --upgrade pip

# Update all packages
pip install --upgrade -r requirements.txt

# Check for outdated packages
pip list --outdated
```

### Framework Updates

```bash
# Pull latest changes (if using git)
git pull origin main

# Reinstall dependencies
pip install -r requirements.txt

# Verify update
python -m feriq.cli.main --version
python -m feriq.cli.main list components
```

### Data Migration

```bash
# Backup existing outputs
cp -r feriq/outputs feriq/outputs_backup_$(date +%Y%m%d)

# Regenerate samples with new format
python -m feriq.cli.main list generate-samples --confirm --count 5

# Verify migration
python -m feriq.cli.main list outputs --detailed
```

---

## ✅ Installation Checklist

### Basic Installation
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed from requirements.txt
- [ ] CLI accessible via `python -m feriq.cli.main --help`
- [ ] Framework components listed successfully
- [ ] Sample data generated

### Model Integration
- [ ] Ollama installed and running (recommended)
- [ ] At least one model downloaded (llama2, codellama, or mistral)
- [ ] Model integration tested with `plan demo`
- [ ] Alternative models configured (OpenAI/Azure if needed)

### Configuration
- [ ] Output directories created
- [ ] Configuration file created (optional)
- [ ] Environment variables set (if needed)
- [ ] Logging configured

### Verification
- [ ] All 8 components functional
- [ ] CLI listing commands working
- [ ] Reasoning capabilities tested
- [ ] Planning with reasoning functional
- [ ] Performance acceptable

### Documentation
- [ ] Quick start guide reviewed
- [ ] CLI listing guide accessible
- [ ] Architecture documentation available
- [ ] Examples explored

---

## 🎯 Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quick_start.md)** for immediate usage
2. **Explore the [CLI Listing Guide](cli_listing_guide.md)** for comprehensive CLI capabilities
3. **Review the [Architecture Overview](architecture.md)** for system understanding
4. **Try the [Programming Examples](examples/programming_examples.md)** for development
5. **Join the community** and contribute to the framework

---

**🎉 Congratulations! Feriq is now installed and ready to use.**

Need help? Check the [troubleshooting section](#-troubleshooting) or explore the [documentation index](README.md).