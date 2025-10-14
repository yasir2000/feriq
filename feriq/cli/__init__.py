"""
Feriq CLI Module

Command-line interface for the Feriq collaborative AI agents framework.
"""

from .main import main, cli
from .commands import *
from .models import ModelManager
from .utils import *

__all__ = ['main', 'cli', 'ModelManager']