"""Goal class for representing objectives in the Feriq framework."""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class GoalPriority(str, Enum):
    """Priority levels for goals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GoalStatus(str, Enum):
    """Status states for goal execution."""
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalType(str, Enum):
    """Types of goals in the framework."""
    SIMPLE = "simple"          # Single-step goal
    COMPLEX = "complex"        # Multi-step goal requiring decomposition
    COLLABORATIVE = "collaborative"  # Requires multiple agents
    SEQUENTIAL = "sequential"  # Steps must be executed in order
    PARALLEL = "parallel"     # Steps can be executed simultaneously
    ADAPTIVE = "adaptive"     # Goal that changes based on context


class SuccessCriteria(BaseModel):
    """Defines criteria for goal success."""
    name: str = Field(..., description="Name of the success criterion")
    description: str = Field(..., description="Detailed description")
    measurable: bool = Field(default=True, description="Whether this criterion is measurable")
    target_value: Optional[Union[str, float, int]] = Field(default=None, 
                                                          description="Target value if measurable")
    validation_method: str = Field(default="manual", 
                                 description="How to validate this criterion")
    weight: float = Field(default=1.0, ge=0.0, 
                         description="Relative importance weight")


class Goal(BaseModel):
    """
    Represents a goal or objective to be accomplished by the framework.
    
    Goals can be simple or complex, and may require coordination between
    multiple agents to achieve.
    """
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
                   description="Unique identifier for this goal")
    name: str = Field(..., description="Human-readable name for the goal")
    description: str = Field(..., description="Detailed description of what needs to be achieved")
    
    # Classification
    goal_type: GoalType = Field(default=GoalType.SIMPLE, 
                               description="Type/category of this goal")
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM, 
                                  description="Priority level of this goal")
    
    # Status and lifecycle
    status: GoalStatus = Field(default=GoalStatus.PENDING, 
                              description="Current execution status")
    created_at: datetime = Field(default_factory=datetime.now, 
                                description="When this goal was created")
    updated_at: datetime = Field(default_factory=datetime.now, 
                                description="When this goal was last updated")
    deadline: Optional[datetime] = Field(default=None, 
                                        description="Optional deadline for completion")
    
    # Requirements and constraints
    required_capabilities: List[str] = Field(default_factory=list,
                                           description="Capabilities needed to achieve this goal")
    required_resources: List[str] = Field(default_factory=list,
                                        description="Resources needed for this goal")
    constraints: List[str] = Field(default_factory=list,
                                 description="Constraints that must be respected")
    
    # Success definition
    success_criteria: List[SuccessCriteria] = Field(default_factory=list,
                                                   description="Criteria for determining success")
    expected_outcome: str = Field(default="", 
                                description="Description of expected outcome")
    
    # Decomposition and relationships
    parent_goal_id: Optional[str] = Field(default=None, 
                                         description="ID of parent goal if this is a sub-goal")
    sub_goal_ids: List[str] = Field(default_factory=list,
                                   description="IDs of sub-goals")
    dependencies: List[str] = Field(default_factory=list,
                                  description="IDs of goals that must be completed first")
    
    # Execution context
    context: Dict[str, Any] = Field(default_factory=dict,
                                  description="Additional context information")
    parameters: Dict[str, Any] = Field(default_factory=dict,
                                     description="Parameters for goal execution")
    
    # Progress tracking
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0,
                                     description="Completion percentage")
    metrics: Dict[str, Any] = Field(default_factory=dict,
                                  description="Execution metrics and measurements")
    
    # Assignment and ownership
    assigned_agent_ids: List[str] = Field(default_factory=list,
                                        description="IDs of agents assigned to this goal")
    primary_agent_id: Optional[str] = Field(default=None,
                                          description="ID of primary responsible agent")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                   description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_success_criterion(self, criterion: SuccessCriteria) -> None:
        """Add a success criterion to this goal."""
        self.success_criteria.append(criterion)
        self.updated_at = datetime.now()
    
    def add_sub_goal(self, sub_goal_id: str) -> None:
        """Add a sub-goal to this goal."""
        if sub_goal_id not in self.sub_goal_ids:
            self.sub_goal_ids.append(sub_goal_id)
            self.updated_at = datetime.now()
    
    def add_dependency(self, dependency_goal_id: str) -> None:
        """Add a dependency to this goal."""
        if dependency_goal_id not in self.dependencies:
            self.dependencies.append(dependency_goal_id)
            self.updated_at = datetime.now()
    
    def assign_agent(self, agent_id: str, is_primary: bool = False) -> None:
        """Assign an agent to this goal."""
        if agent_id not in self.assigned_agent_ids:
            self.assigned_agent_ids.append(agent_id)
        
        if is_primary:
            self.primary_agent_id = agent_id
        
        self.updated_at = datetime.now()
    
    def update_progress(self, percentage: float) -> None:
        """Update the progress percentage."""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        self.updated_at = datetime.now()
        
        # Auto-update status based on progress
        if percentage >= 100.0 and self.status == GoalStatus.IN_PROGRESS:
            self.status = GoalStatus.COMPLETED
        elif percentage > 0.0 and self.status == GoalStatus.PENDING:
            self.status = GoalStatus.IN_PROGRESS
    
    def update_status(self, new_status: GoalStatus) -> None:
        """Update the goal status."""
        self.status = new_status
        self.updated_at = datetime.now()
        
        # Update progress based on status
        if new_status == GoalStatus.COMPLETED:
            self.progress_percentage = 100.0
        elif new_status == GoalStatus.PENDING:
            self.progress_percentage = 0.0
    
    def is_overdue(self) -> bool:
        """Check if this goal is overdue."""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline and self.status not in [
            GoalStatus.COMPLETED, GoalStatus.CANCELLED
        ]
    
    def is_ready_to_start(self, completed_goal_ids: set) -> bool:
        """Check if this goal is ready to start (all dependencies met)."""
        if self.status != GoalStatus.PENDING:
            return False
        
        # Check if all dependencies are completed
        return all(dep_id in completed_goal_ids for dep_id in self.dependencies)
    
    def calculate_complexity_score(self) -> float:
        """Calculate a complexity score for this goal."""
        score = 1.0  # Base complexity
        
        # Add complexity based on goal type
        type_weights = {
            GoalType.SIMPLE: 1.0,
            GoalType.COMPLEX: 2.0,
            GoalType.COLLABORATIVE: 1.5,
            GoalType.SEQUENTIAL: 1.3,
            GoalType.PARALLEL: 1.2,
            GoalType.ADAPTIVE: 2.5
        }
        score *= type_weights.get(self.goal_type, 1.0)
        
        # Add complexity based on sub-goals
        score += len(self.sub_goal_ids) * 0.2
        
        # Add complexity based on dependencies
        score += len(self.dependencies) * 0.1
        
        # Add complexity based on required capabilities
        score += len(self.required_capabilities) * 0.1
        
        # Add complexity based on success criteria
        score += len(self.success_criteria) * 0.1
        
        return score
    
    def get_estimated_effort(self) -> float:
        """Estimate the effort required for this goal."""
        # This is a simple heuristic - could be enhanced with ML models
        complexity = self.calculate_complexity_score()
        
        # Priority affects effort estimation
        priority_multipliers = {
            GoalPriority.LOW: 0.8,
            GoalPriority.MEDIUM: 1.0,
            GoalPriority.HIGH: 1.2,
            GoalPriority.CRITICAL: 1.5
        }
        
        return complexity * priority_multipliers.get(self.priority, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary representation."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        """Create goal from dictionary representation."""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of the goal."""
        return f"Goal(id='{self.id[:8]}...', name='{self.name}', status='{self.status}')"
    
    def __repr__(self) -> str:
        """Developer representation of the goal."""
        return self.__str__()