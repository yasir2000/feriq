"""Logging utilities for the Feriq framework."""

import logging
import structlog
from typing import Any, Dict, Optional
from datetime import datetime
import sys
import os


class FeriqLogger:
    """Custom logger for Feriq framework with structured logging."""
    
    def __init__(self, name: str, level: str = "INFO", format_type: str = "console"):
        """Initialize the logger."""
        self.name = name
        self.level = level
        self.format_type = format_type
        self._logger = None
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up the structured logger."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._format_processor(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Set up standard library logger
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, self.level.upper()),
        )
        
        # Create the logger
        self._logger = structlog.get_logger(self.name)
    
    def _format_processor(self):
        """Return appropriate format processor based on format type."""
        if self.format_type == "json":
            return structlog.processors.JSONRenderer()
        else:
            return structlog.dev.ConsoleRenderer(colors=True)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, **kwargs)


# Global logger registry
_loggers: Dict[str, FeriqLogger] = {}


def get_logger(name: str, level: str = None, format_type: str = None) -> FeriqLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (console, json)
        
    Returns:
        FeriqLogger instance
    """
    # Use environment variables for defaults
    default_level = os.getenv("FERIQ_LOG_LEVEL", "INFO")
    default_format = os.getenv("FERIQ_LOG_FORMAT", "console")
    
    level = level or default_level
    format_type = format_type or default_format
    
    logger_key = f"{name}:{level}:{format_type}"
    
    if logger_key not in _loggers:
        _loggers[logger_key] = FeriqLogger(name, level, format_type)
    
    return _loggers[logger_key]


def configure_logging(level: str = "INFO", format_type: str = "console", 
                     log_file: Optional[str] = None) -> None:
    """
    Configure global logging settings.
    
    Args:
        level: Global log level
        format_type: Format type (console, json)
        log_file: Optional log file path
    """
    # Clear existing loggers
    global _loggers
    _loggers.clear()
    
    # Set environment variables
    os.environ["FERIQ_LOG_LEVEL"] = level
    os.environ["FERIQ_LOG_FORMAT"] = format_type
    
    if log_file:
        os.environ["FERIQ_LOG_FILE"] = log_file
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


# Context managers for temporary logging configuration
class LogLevel:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, logger: FeriqLogger, level: str):
        """Initialize with logger and new level."""
        self.logger = logger
        self.new_level = level
        self.old_level = None
    
    def __enter__(self):
        """Enter context - change log level."""
        self.old_level = self.logger.level
        self.logger.level = self.new_level
        self.logger._setup_logger()
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore log level."""
        self.logger.level = self.old_level
        self.logger._setup_logger()


class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: FeriqLogger, **context):
        """Initialize with logger and context."""
        self.logger = logger
        self.context = context
        self.bound_logger = None
    
    def __enter__(self):
        """Enter context - bind context to logger."""
        self.bound_logger = self.logger._logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass


# Decorator for automatic logging
def log_function_call(logger_name: str = None, level: str = "INFO"):
    """
    Decorator to automatically log function calls.
    
    Args:
        logger_name: Name of logger to use (defaults to module name)
        level: Log level for the messages
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger_name
            if logger_name is None:
                logger_name = func.__module__
            
            logger = get_logger(logger_name)
            
            # Log function entry
            getattr(logger, level.lower())(
                f"Entering function {func.__name__}",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful completion
                getattr(logger, level.lower())(
                    f"Function {func.__name__} completed successfully",
                    function=func.__name__,
                    has_result=result is not None
                )
                
                return result
                
            except Exception as e:
                # Log exception
                logger.error(
                    f"Function {func.__name__} failed with exception",
                    function=func.__name__,
                    exception=str(e),
                    exception_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator


# Performance logging utilities
class PerformanceTimer:
    """Context manager for measuring and logging performance."""
    
    def __init__(self, logger: FeriqLogger, operation_name: str, 
                 threshold_ms: float = None):
        """
        Initialize performance timer.
        
        Args:
            logger: Logger instance
            operation_name: Name of operation being timed
            threshold_ms: Log warning if operation takes longer than this
        """
        self.logger = logger
        self.operation_name = operation_name
        self.threshold_ms = threshold_ms
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        self.logger.debug(
            f"Starting operation: {self.operation_name}",
            operation=self.operation_name,
            start_time=self.start_time.isoformat()
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        self.end_time = datetime.now()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        log_level = "info"
        if self.threshold_ms and duration_ms > self.threshold_ms:
            log_level = "warning"
        
        getattr(self.logger, log_level)(
            f"Operation completed: {self.operation_name}",
            operation=self.operation_name,
            duration_ms=duration_ms,
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat(),
            success=exc_type is None
        )
    
    def get_duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


# Audit logging
class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self, name: str = "feriq.audit"):
        """Initialize audit logger."""
        self.logger = get_logger(name, level="INFO", format_type="json")
    
    def log_event(self, event_type: str, actor: str, resource: str, 
                  action: str, result: str, **metadata) -> None:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (e.g., "task_assignment", "role_change")
            actor: Who performed the action
            resource: What was acted upon
            action: What action was performed
            result: Result of the action
            **metadata: Additional metadata
        """
        self.logger.info(
            f"Audit event: {event_type}",
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            result=result,
            timestamp=datetime.now().isoformat(),
            **metadata
        )
    
    def log_task_assignment(self, task_id: str, agent_id: str, 
                           assigner: str, success: bool) -> None:
        """Log task assignment event."""
        self.log_event(
            event_type="task_assignment",
            actor=assigner,
            resource=f"task:{task_id}",
            action="assign",
            result="success" if success else "failure",
            agent_id=agent_id
        )
    
    def log_role_change(self, agent_id: str, old_role: str, new_role: str, 
                       changer: str) -> None:
        """Log role change event."""
        self.log_event(
            event_type="role_change",
            actor=changer,
            resource=f"agent:{agent_id}",
            action="change_role",
            result="success",
            old_role=old_role,
            new_role=new_role
        )
    
    def log_plan_execution(self, plan_id: str, executor: str, 
                          status: str, **details) -> None:
        """Log plan execution event."""
        self.log_event(
            event_type="plan_execution",
            actor=executor,
            resource=f"plan:{plan_id}",
            action="execute",
            result=status,
            **details
        )


# Module-level audit logger instance
audit_logger = AuditLogger()