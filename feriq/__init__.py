"""
Feriq - Collaborative AI Agents Framework

A sophisticated multi-agent collaboration framework built on CrewAI
that provides dynamic role assignment, intelligent task orchestration,
and advanced workflow management capabilities.
"""

__version__ = "0.1.0"
__author__ = "Yasir Anwar"
__email__ = "yasir2000@example.com"

from .core.framework import FeriqFramework
from .core.agent import FeriqAgent
from .core.task import FeriqTask
from .core.goal import Goal
from .core.role import Role
from .core.plan import Plan

# Core components
from .components.role_designer import DynamicRoleDesigner
from .components.task_designer import TaskDesigner
from .components.task_allocator import TaskAllocator
from .components.plan_designer import PlanDesigner
from .components.plan_observer import PlanObserver
from .components.orchestrator import WorkflowOrchestrator
from .components.choreographer import Choreographer
from .components.reasoner import Reasoner

# Utilities
from .utils.logger import get_logger
from .utils.config import Config

__all__ = [
    # Core framework
    "FeriqFramework",
    "FeriqAgent",
    "FeriqTask",
    "Goal",
    "Role",
    "Plan",
    # Components
    "DynamicRoleDesigner",
    "TaskDesigner",
    "TaskAllocator",
    "PlanDesigner",
    "PlanObserver",
    "WorkflowOrchestrator",
    "Choreographer",
    "Reasoner",
    # Utilities
    "get_logger",
    "Config",
]