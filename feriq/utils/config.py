"""Configuration management for the Feriq framework."""

from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field
import os
import json
import yaml
from pathlib import Path


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    format_type: str = Field(default="console", description="Log format (console, json)")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: str = Field(default="10MB", description="Maximum log file size")
    backup_count: int = Field(default=5, description="Number of backup log files")


class AgentConfig(BaseModel):
    """Agent configuration."""
    max_concurrent_tasks: int = Field(default=3, description="Maximum concurrent tasks per agent")
    default_efficiency_score: float = Field(default=0.5, description="Default efficiency score")
    learning_enabled: bool = Field(default=True, description="Whether learning is enabled")
    learning_rate: float = Field(default=0.1, description="Learning rate")
    collaboration_timeout: int = Field(default=300, description="Collaboration timeout in seconds")


class TaskConfig(BaseModel):
    """Task configuration."""
    default_priority: str = Field(default="medium", description="Default task priority")
    auto_decomposition: bool = Field(default=True, description="Enable automatic task decomposition")
    max_decomposition_depth: int = Field(default=3, description="Maximum decomposition depth")
    default_timeout: int = Field(default=3600, description="Default task timeout in seconds")
    retry_attempts: int = Field(default=3, description="Maximum retry attempts")


class PlanConfig(BaseModel):
    """Plan configuration."""
    auto_planning: bool = Field(default=True, description="Enable automatic plan generation")
    max_plan_duration: int = Field(default=86400, description="Maximum plan duration in seconds")
    monitoring_interval: int = Field(default=60, description="Monitoring interval in seconds")
    adaptation_threshold: float = Field(default=0.7, description="Threshold for plan adaptation")


class AllocationConfig(BaseModel):
    """Task allocation configuration."""
    default_strategy: str = Field(default="balanced", description="Default allocation strategy")
    rebalancing_enabled: bool = Field(default=True, description="Enable workload rebalancing")
    rebalancing_threshold: float = Field(default=0.8, description="Workload threshold for rebalancing")
    queue_processing_interval: int = Field(default=30, description="Queue processing interval in seconds")


class PerformanceConfig(BaseModel):
    """Performance monitoring configuration."""
    metrics_collection: bool = Field(default=True, description="Enable metrics collection")
    history_retention_days: int = Field(default=30, description="Metrics history retention period")
    performance_alerts: bool = Field(default=True, description="Enable performance alerts")
    efficiency_threshold: float = Field(default=0.3, description="Minimum efficiency threshold")


class SecurityConfig(BaseModel):
    """Security configuration."""
    audit_logging: bool = Field(default=True, description="Enable audit logging")
    encryption_enabled: bool = Field(default=False, description="Enable data encryption")
    api_key_required: bool = Field(default=False, description="Require API key for access")
    max_session_duration: int = Field(default=3600, description="Maximum session duration in seconds")


class IntegrationConfig(BaseModel):
    """Integration configuration."""
    crewai_config: Dict[str, Any] = Field(default_factory=dict, description="CrewAI configuration")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="LLM configuration")
    external_apis: Dict[str, Any] = Field(default_factory=dict, description="External API configurations")
    database_url: Optional[str] = Field(default=None, description="Database connection URL")


class Config(BaseModel):
    """
    Main configuration class for the Feriq framework.
    
    Handles loading, validation, and management of all configuration settings.
    """
    
    # Core framework settings
    framework_name: str = Field(default="Feriq", description="Framework name")
    version: str = Field(default="0.1.0", description="Framework version")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Component configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    agents: AgentConfig = Field(default_factory=AgentConfig, description="Agent configuration")
    tasks: TaskConfig = Field(default_factory=TaskConfig, description="Task configuration")
    plans: PlanConfig = Field(default_factory=PlanConfig, description="Plan configuration")
    allocation: AllocationConfig = Field(default_factory=AllocationConfig, description="Allocation configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    integrations: IntegrationConfig = Field(default_factory=IntegrationConfig, description="Integration configuration")
    
    # Custom settings
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration settings")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        validate_assignment = True
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "Config":
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Config instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
        
        return cls(**data)
    
    @classmethod
    def load_from_env(cls, prefix: str = "FERIQ_") -> "Config":
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Config instance
        """
        config_data = {}
        
        # Mapping of environment variables to config paths
        env_mappings = {
            f"{prefix}DEBUG": "debug",
            f"{prefix}ENVIRONMENT": "environment",
            f"{prefix}LOG_LEVEL": "logging.level",
            f"{prefix}LOG_FORMAT": "logging.format_type",
            f"{prefix}LOG_FILE": "logging.file_path",
            f"{prefix}MAX_CONCURRENT_TASKS": "agents.max_concurrent_tasks",
            f"{prefix}LEARNING_ENABLED": "agents.learning_enabled",
            f"{prefix}AUTO_DECOMPOSITION": "tasks.auto_decomposition",
            f"{prefix}DEFAULT_STRATEGY": "allocation.default_strategy",
            f"{prefix}AUDIT_LOGGING": "security.audit_logging",
            f"{prefix}DATABASE_URL": "integrations.database_url"
        }
        
        # Process environment variables
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
                
                # Set nested config value
                cls._set_nested_value(config_data, config_path, value)
        
        return cls(**config_data)
    
    @staticmethod
    def _set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested dictionary value using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def save_to_file(self, file_path: Union[str, Path], format_type: str = "yaml") -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Output file path
            format_type: File format ('yaml' or 'json')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.model_dump()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format_type.lower() == 'yaml':
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format_type.lower() == 'json':
                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates using dot notation for nested values
        """
        for key, value in updates.items():
            if '.' in key:
                # Handle nested updates
                self._update_nested_value(key, value)
            else:
                # Handle top-level updates
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self.custom[key] = value
    
    def _update_nested_value(self, path: str, value: Any) -> None:
        """Update a nested configuration value using dot notation."""
        keys = path.split('.')
        current = self
        
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                # Handle custom nested values
                if not hasattr(current, 'custom'):
                    current.custom = {}
                if key not in current.custom:
                    current.custom[key] = {}
                current = current.custom[key]
        
        final_key = keys[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        else:
            if not hasattr(current, 'custom'):
                current.custom = {}
            current.custom[final_key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            path: Configuration path (e.g., 'logging.level')
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        current = self
        
        try:
            for key in keys:
                if hasattr(current, key):
                    current = getattr(current, key)
                elif hasattr(current, 'custom') and key in current.custom:
                    current = current.custom[key]
                else:
                    return default
            return current
        except (AttributeError, KeyError, TypeError):
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def to_env_vars(self, prefix: str = "FERIQ_") -> Dict[str, str]:
        """
        Convert configuration to environment variables format.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Dictionary of environment variables
        """
        env_vars = {}
        
        # Flatten configuration
        flat_config = self._flatten_dict(self.to_dict())
        
        for key, value in flat_config.items():
            env_key = f"{prefix}{key.upper().replace('.', '_')}"
            env_vars[env_key] = str(value)
        
        return env_vars
    
    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Validate logging configuration
        if self.logging.level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            issues.append(f"Invalid log level: {self.logging.level}")
        
        if self.logging.format_type not in ['console', 'json']:
            issues.append(f"Invalid log format: {self.logging.format_type}")
        
        # Validate agent configuration
        if self.agents.max_concurrent_tasks < 1:
            issues.append("Max concurrent tasks must be at least 1")
        
        if not 0.0 <= self.agents.default_efficiency_score <= 1.0:
            issues.append("Default efficiency score must be between 0.0 and 1.0")
        
        if not 0.0 <= self.agents.learning_rate <= 1.0:
            issues.append("Learning rate must be between 0.0 and 1.0")
        
        # Validate task configuration
        if self.tasks.max_decomposition_depth < 1:
            issues.append("Max decomposition depth must be at least 1")
        
        if self.tasks.retry_attempts < 0:
            issues.append("Retry attempts cannot be negative")
        
        # Validate allocation configuration
        if not 0.0 <= self.allocation.rebalancing_threshold <= 1.0:
            issues.append("Rebalancing threshold must be between 0.0 and 1.0")
        
        # Validate performance configuration
        if self.performance.history_retention_days < 1:
            issues.append("History retention days must be at least 1")
        
        if not 0.0 <= self.performance.efficiency_threshold <= 1.0:
            issues.append("Efficiency threshold must be between 0.0 and 1.0")
        
        return issues
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate_config()) == 0
    
    def get_component_config(self, component_name: str) -> Optional[BaseModel]:
        """
        Get configuration for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component configuration or None
        """
        component_configs = {
            'logging': self.logging,
            'agents': self.agents,
            'tasks': self.tasks,
            'plans': self.plans,
            'allocation': self.allocation,
            'performance': self.performance,
            'security': self.security,
            'integrations': self.integrations
        }
        
        return component_configs.get(component_name)
    
    def create_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Create a configuration profile for specific environments.
        
        Args:
            profile_name: Name of the profile (development, staging, production)
            
        Returns:
            Profile configuration
        """
        base_config = self.to_dict()
        
        profiles = {
            'development': {
                'debug': True,
                'logging': {'level': 'DEBUG', 'format_type': 'console'},
                'performance': {'metrics_collection': True, 'performance_alerts': False},
                'security': {'audit_logging': False, 'encryption_enabled': False}
            },
            'staging': {
                'debug': False,
                'logging': {'level': 'INFO', 'format_type': 'json'},
                'performance': {'metrics_collection': True, 'performance_alerts': True},
                'security': {'audit_logging': True, 'encryption_enabled': False}
            },
            'production': {
                'debug': False,
                'logging': {'level': 'WARNING', 'format_type': 'json'},
                'performance': {'metrics_collection': True, 'performance_alerts': True},
                'security': {'audit_logging': True, 'encryption_enabled': True}
            }
        }
        
        profile_overrides = profiles.get(profile_name, {})
        
        # Apply profile overrides
        for key, value in profile_overrides.items():
            if isinstance(value, dict) and key in base_config:
                base_config[key].update(value)
            else:
                base_config[key] = value
        
        return base_config
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"FeriqConfig(environment={self.environment}, debug={self.debug})"
    
    def __repr__(self) -> str:
        """Developer representation of configuration."""
        return self.__str__()


# Default configuration instance
default_config = Config()


def load_config(file_path: Optional[Union[str, Path]] = None, 
               from_env: bool = True) -> Config:
    """
    Load configuration from file and/or environment variables.
    
    Args:
        file_path: Optional path to configuration file
        from_env: Whether to load from environment variables
        
    Returns:
        Config instance
    """
    if file_path:
        config = Config.load_from_file(file_path)
    else:
        config = Config()
    
    if from_env:
        env_config = Config.load_from_env()
        # Merge environment config into file config
        config.update(env_config.to_dict())
    
    return config