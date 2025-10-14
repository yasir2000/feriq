"""
CLI Utilities

Utility functions and classes for the Feriq CLI.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import click
from datetime import datetime


class CliContext:
    """Global CLI context for managing state and configuration."""
    
    def __init__(self):
        self.verbose = False
        self.debug = False
        self.config = {}
        self.project_path = str(Path.cwd())  # Convert to string to avoid Click issues
        self.framework = None
        
    def load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    self.config = json.load(f)
            
            if self.verbose:
                print_info(f"Loaded configuration from {config_path}")
        except Exception as e:
            print_error(f"Failed to load config: {e}")
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        try:
            with open(config_path, 'w') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
            
            print_success(f"Configuration saved to {config_path}")
        except Exception as e:
            print_error(f"Failed to save config: {e}")
    
    def get_framework(self):
        """Get or create framework instance."""
        if self.framework is None:
            from ..core.framework import FeriqFramework
            from ..utils.config import Config
            
            config = Config()
            if self.config:
                config.update(self.config)
            
            self.framework = FeriqFramework(config)
        
        return self.framework


def print_banner():
    """Print the Feriq CLI banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ███████╗███████╗██████╗ ██╗ ██████╗                   ║
║   ██╔════╝██╔════╝██╔══██╗██║██╔═══██╗                  ║
║   █████╗  █████╗  ██████╔╝██║██║   ██║                  ║
║   ██╔══╝  ██╔══╝  ██╔══██╗██║██║▄▄ ██║                  ║
║   ██║     ███████╗██║  ██║██║╚██████╔╝                  ║
║   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝ ╚══▀▀═╝                   ║
║                                                           ║
║        Collaborative AI Agents Framework CLI             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    click.echo(click.style(banner, fg='cyan', bold=True))


def print_success(message: str):
    """Print success message."""
    click.echo(click.style(f"✅ {message}", fg='green', bold=True))


def print_error(message: str):
    """Print error message."""
    click.echo(click.style(f"❌ {message}", fg='red', bold=True), err=True)


def print_warning(message: str):
    """Print warning message."""
    click.echo(click.style(f"⚠️  {message}", fg='yellow', bold=True))


def print_info(message: str):
    """Print info message."""
    click.echo(click.style(f"ℹ️  {message}", fg='blue', bold=True))


def print_header(message: str):
    """Print section header."""
    click.echo(click.style(f"\n🔸 {message}", fg='magenta', bold=True))


def print_table(headers: list, rows: list, title: Optional[str] = None):
    """Print a formatted table."""
    if title:
        print_header(title)
    
    if not rows:
        print_info("No data to display")
        return
    
    # Calculate column widths
    widths = []
    for i, header in enumerate(headers):
        max_width = len(str(header))  # Ensure header is string
        for row in rows:
            if i < len(row):
                cell_str = str(row[i])  # Ensure cell is string
                max_width = max(max_width, len(cell_str))
        widths.append(max_width + 2)
    
    # Print header
    header_line = "│"
    separator_line = "├"
    for i, (header, width) in enumerate(zip(headers, widths)):
        header_line += f" {header:<{width-1}}│"
        separator_line += "─" * width + ("┼" if i < len(headers) - 1 else "┤")
    
    print("┌" + "─" * (len(header_line) - 2) + "┐")
    print(header_line)
    print(separator_line)
    
    # Print rows
    for row in rows:
        row_line = "│"
        for i, (cell, width) in enumerate(zip(row, widths)):
            row_line += f" {str(cell):<{width-1}}│"
        print(row_line)
    
    print("└" + "─" * (len(header_line) - 2) + "┘")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp in human-readable format."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_size(bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}PB"


def confirm_action(message: str, default: bool = False) -> bool:
    """Prompt user for confirmation."""
    prompt = f"{message} {'[Y/n]' if default else '[y/N]'}"
    result = click.prompt(prompt, default='y' if default else 'n', show_default=False)
    return result.lower() in ['y', 'yes', 'true', '1']


def get_user_input(prompt: str, type=str, default=None, required=True):
    """Get user input with validation."""
    while True:
        try:
            value = click.prompt(prompt, type=type, default=default, show_default=True)
            if required and not value:
                print_error("This field is required.")
                continue
            return value
        except click.Abort:
            raise KeyboardInterrupt()


def validate_project_directory(path: Path) -> bool:
    """Validate if directory is a Feriq project."""
    return (path / 'feriq.yaml').exists() or (path / 'feriq.json').exists()


def find_project_root() -> Optional[Path]:
    """Find the project root directory."""
    current = Path.cwd()
    
    while current != current.parent:
        if validate_project_directory(current):
            return current
        current = current.parent
    
    return None


def ensure_project_directory():
    """Ensure we're in a Feriq project directory."""
    if not find_project_root():
        print_error("Not in a Feriq project directory. Run 'feriq init' to create a new project.")
        raise click.Abort()


def load_project_config() -> Dict[str, Any]:
    """Load project configuration."""
    project_root = find_project_root()
    if not project_root:
        return {}
    
    config_files = ['feriq.yaml', 'feriq.yml', 'feriq.json']
    
    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    if config_file.endswith('.json'):
                        return json.load(f)
                    else:
                        return yaml.safe_load(f)
            except Exception as e:
                print_warning(f"Failed to load {config_file}: {e}")
    
    return {}


def save_project_config(config: Dict[str, Any], format: str = 'yaml'):
    """Save project configuration."""
    project_root = find_project_root() or Path.cwd()
    
    if format == 'json':
        config_path = project_root / 'feriq.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        config_path = project_root / 'feriq.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print_success(f"Project configuration saved to {config_path}")


def create_directory_structure(base_path: Path):
    """Create standard Feriq project directory structure."""
    directories = [
        'agents',
        'goals',
        'plans',
        'workflows',
        'logs',
        'data',
        'configs'
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(exist_ok=True)
        
        # Create .gitkeep files
        gitkeep = dir_path / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()


def get_available_models() -> Dict[str, list]:
    """Get available models from different providers."""
    from .models import ModelManager
    
    model_manager = ModelManager()
    models = {
        'ollama': [],
        'openai': [],
        'anthropic': []
    }
    
    try:
        models['ollama'] = model_manager.list_ollama_models()
    except Exception:
        pass
    
    # Add common OpenAI models if API key is available
    if os.getenv('OPENAI_API_KEY'):
        models['openai'] = [
            'gpt-4',
            'gpt-4-turbo',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k'
        ]
    
    # Add common Anthropic models if API key is available
    if os.getenv('ANTHROPIC_API_KEY'):
        models['anthropic'] = [
            'claude-3-opus',
            'claude-3-sonnet',
            'claude-3-haiku'
        ]
    
    return models


def select_model_interactive() -> tuple:
    """Interactive model selection."""
    models = get_available_models()
    
    if not any(models.values()):
        print_error("No models available. Please configure at least one model provider.")
        return None, None
    
    print_header("Available Models")
    
    all_models = []
    for provider, provider_models in models.items():
        for model in provider_models:
            all_models.append((provider, model))
    
    if not all_models:
        print_error("No models found.")
        return None, None
    
    # Display models
    for i, (provider, model) in enumerate(all_models, 1):
        print(f"{i:2d}. {provider:10s} - {model}")
    
    while True:
        try:
            choice = click.prompt("\nSelect model", type=int)
            if 1 <= choice <= len(all_models):
                return all_models[choice - 1]
            else:
                print_error(f"Please enter a number between 1 and {len(all_models)}")
        except click.Abort:
            raise KeyboardInterrupt()
        except (ValueError, IndexError):
            print_error("Invalid selection. Please enter a number.")


def display_progress(items, description="Processing"):
    """Display progress for a list of items."""
    import time
    
    with click.progressbar(items, label=description) as bar:
        for item in bar:
            # Simulate work
            time.sleep(0.1)
            yield item