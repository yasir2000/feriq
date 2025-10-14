"""Task class for representing work units in the Feriq framework."""

from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Status states for task execution."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskType(str, Enum):
    """Types of tasks in the framework."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"
    COORDINATION = "coordination"
    COMMUNICATION = "communication"
    DECISION = "decision"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class TaskComplexity(str, Enum):
    """Complexity levels for tasks."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class TaskDependency(BaseModel):
    """Represents a dependency between tasks."""
    task_id: str = Field(..., description="ID of the dependent task")
    dependency_type: str = Field(default="finish_to_start", 
                               description="Type of dependency relationship")
    lag_time: float = Field(default=0.0, description="Lag time in hours")


class TaskResource(BaseModel):
    """Represents a resource required by a task."""
    name: str = Field(..., description="Name of the resource")
    resource_type: str = Field(..., description="Type of resource")
    quantity: float = Field(default=1.0, description="Required quantity")
    unit: str = Field(default="unit", description="Unit of measurement")
    availability: bool = Field(default=True, description="Whether resource is available")


class FeriqTask(BaseModel):
    """
    Represents a task or work unit to be executed by agents in the Feriq framework.
    
    Tasks are the fundamental units of work that agents perform to achieve goals.
    They can be simple atomic operations or complex multi-step processes.
    """
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
                   description="Unique identifier for this task")
    name: str = Field(..., description="Human-readable name for the task")
    description: str = Field(..., description="Detailed description of what needs to be done")
    
    # Classification
    task_type: TaskType = Field(default=TaskType.CUSTOM, 
                               description="Type/category of this task")
    complexity: TaskComplexity = Field(default=TaskComplexity.MODERATE,
                                     description="Complexity level of this task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, 
                                  description="Priority level of this task")
    
    # Status and lifecycle
    status: TaskStatus = Field(default=TaskStatus.PENDING, 
                              description="Current execution status")
    created_at: datetime = Field(default_factory=datetime.now, 
                                description="When this task was created")
    updated_at: datetime = Field(default_factory=datetime.now, 
                                description="When this task was last updated")
    started_at: Optional[datetime] = Field(default=None, 
                                          description="When execution started")
    completed_at: Optional[datetime] = Field(default=None, 
                                            description="When execution completed")
    deadline: Optional[datetime] = Field(default=None, 
                                        description="Optional deadline for completion")
    
    # Goal relationship
    goal_id: Optional[str] = Field(default=None, 
                                  description="ID of the goal this task contributes to")
    
    # Requirements and constraints
    required_capabilities: List[str] = Field(default_factory=list,
                                           description="Capabilities needed to execute this task")
    required_role_types: List[str] = Field(default_factory=list,
                                         description="Types of roles suitable for this task")
    required_resources: List[TaskResource] = Field(default_factory=list,
                                                 description="Resources needed for this task")
    constraints: List[str] = Field(default_factory=list,
                                 description="Constraints that must be respected")
    
    # Dependencies and relationships
    dependencies: List[TaskDependency] = Field(default_factory=list,
                                             description="Tasks that must be completed first")
    dependent_task_ids: List[str] = Field(default_factory=list,
                                        description="Tasks that depend on this one")
    parent_task_id: Optional[str] = Field(default=None, 
                                         description="ID of parent task if this is a subtask")
    subtask_ids: List[str] = Field(default_factory=list,
                                 description="IDs of subtasks")
    
    # Execution details
    instructions: str = Field(default="", 
                            description="Detailed instructions for execution")
    expected_output: str = Field(default="", 
                               description="Description of expected output")
    validation_criteria: List[str] = Field(default_factory=list,
                                         description="Criteria for validating completion")
    
    # Assignment and execution
    assigned_agent_id: Optional[str] = Field(default=None,
                                           description="ID of assigned agent")
    assigned_role_id: Optional[str] = Field(default=None,
                                          description="ID of assigned role")
    execution_context: Dict[str, Any] = Field(default_factory=dict,
                                            description="Context for execution")
    parameters: Dict[str, Any] = Field(default_factory=dict,
                                     description="Parameters for task execution")
    
    # Progress and results
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0,
                                     description="Completion percentage")
    result: Optional[Any] = Field(default=None, 
                                 description="Task execution result")
    output: Optional[str] = Field(default=None, 
                                 description="Text output from task execution")
    error_message: Optional[str] = Field(default=None, 
                                        description="Error message if task failed")
    
    # Estimation and metrics
    estimated_duration: Optional[float] = Field(default=None, 
                                               description="Estimated duration in hours")
    actual_duration: Optional[float] = Field(default=None, 
                                           description="Actual duration in hours")
    effort_estimate: float = Field(default=1.0, ge=0.0, 
                                 description="Estimated effort units")
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0,
                                         description="Quality score of execution")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                   description="Additional metadata")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_dependency(self, task_id: str, dependency_type: str = "finish_to_start",
                      lag_time: float = 0.0) -> None:
        """Add a dependency to this task."""
        dependency = TaskDependency(
            task_id=task_id,
            dependency_type=dependency_type,
            lag_time=lag_time
        )
        self.dependencies.append(dependency)
        self.updated_at = datetime.now()
    
    def add_subtask(self, subtask_id: str) -> None:
        """Add a subtask to this task."""
        if subtask_id not in self.subtask_ids:
            self.subtask_ids.append(subtask_id)
            self.updated_at = datetime.now()
    
    def add_resource(self, resource: TaskResource) -> None:
        """Add a required resource to this task."""
        self.required_resources.append(resource)
        self.updated_at = datetime.now()
    
    def assign_to_agent(self, agent_id: str, role_id: Optional[str] = None) -> None:
        """Assign this task to an agent."""
        self.assigned_agent_id = agent_id
        if role_id:
            self.assigned_role_id = role_id
        
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.ASSIGNED
        
        self.updated_at = datetime.now()
    
    def start_execution(self) -> None:
        """Mark the task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
        self.updated_at = datetime.now()
    
    def complete_task(self, result: Any = None, output: str = None) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.progress_percentage = 100.0
        
        if result is not None:
            self.result = result
        if output is not None:
            self.output = output
        
        # Calculate actual duration
        if self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds() / 3600
            self.actual_duration = duration
    
    def fail_task(self, error_message: str) -> None:
        """Mark the task as failed."""
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.now()
        
        if self.started_at and not self.completed_at:
            self.completed_at = datetime.now()
            duration = (self.completed_at - self.started_at).total_seconds() / 3600
            self.actual_duration = duration
    
    def pause_task(self) -> None:
        """Pause the task execution."""
        if self.status == TaskStatus.IN_PROGRESS:
            self.status = TaskStatus.PAUSED
            self.updated_at = datetime.now()
    
    def resume_task(self) -> None:
        """Resume the task execution."""
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.IN_PROGRESS
            self.updated_at = datetime.now()
    
    def block_task(self, reason: str = "") -> None:
        """Block the task execution."""
        self.status = TaskStatus.BLOCKED
        if reason:
            self.metadata["block_reason"] = reason
        self.updated_at = datetime.now()
    
    def update_progress(self, percentage: float) -> None:
        """Update the progress percentage."""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        self.updated_at = datetime.now()
        
        # Auto-complete if 100%
        if percentage >= 100.0 and self.status == TaskStatus.IN_PROGRESS:
            self.complete_task()
    
    def can_retry(self) -> bool:
        """Check if the task can be retried."""
        return (self.status == TaskStatus.FAILED and 
                self.retry_count < self.max_retries)
    
    def retry_task(self) -> None:
        """Retry the task execution."""
        if self.can_retry():
            self.retry_count += 1
            self.status = TaskStatus.PENDING
            self.error_message = None
            self.progress_percentage = 0.0
            self.updated_at = datetime.now()
    
    def is_ready_to_execute(self, completed_task_ids: set) -> bool:
        """Check if this task is ready to execute (all dependencies met)."""
        if self.status not in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
            return False
        
        # Check if all dependencies are completed
        for dep in self.dependencies:
            if dep.task_id not in completed_task_ids:
                return False
        
        # Check if resources are available
        return all(resource.availability for resource in self.required_resources)
    
    def is_overdue(self) -> bool:
        """Check if this task is overdue."""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline and self.status not in [
            TaskStatus.COMPLETED, TaskStatus.CANCELLED
        ]
    
    def calculate_priority_score(self) -> float:
        """Calculate a numerical priority score for task ordering."""
        priority_scores = {
            TaskPriority.LOW: 1.0,
            TaskPriority.MEDIUM: 2.0,
            TaskPriority.HIGH: 3.0,
            TaskPriority.CRITICAL: 4.0
        }
        
        base_score = priority_scores.get(self.priority, 2.0)
        
        # Adjust for deadline urgency
        if self.deadline:
            time_to_deadline = (self.deadline - datetime.now()).total_seconds() / 3600
            if time_to_deadline < 24:  # Less than 24 hours
                base_score *= 1.5
            elif time_to_deadline < 72:  # Less than 3 days
                base_score *= 1.2
        
        # Adjust for blocking other tasks
        if self.dependent_task_ids:
            base_score *= (1 + len(self.dependent_task_ids) * 0.1)
        
        return base_score
    
    def estimate_effort(self) -> float:
        """Estimate the effort required for this task."""
        complexity_multipliers = {
            TaskComplexity.TRIVIAL: 0.25,
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 2.0,
            TaskComplexity.EXPERT: 4.0
        }
        
        base_effort = complexity_multipliers.get(self.complexity, 1.0)
        
        # Adjust for subtasks
        if self.subtask_ids:
            base_effort *= (1 + len(self.subtask_ids) * 0.2)
        
        # Adjust for required capabilities
        if self.required_capabilities:
            base_effort *= (1 + len(self.required_capabilities) * 0.1)
        
        return base_effort
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeriqTask":
        """Create task from dictionary representation."""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of the task."""
        return f"Task(id='{self.id[:8]}...', name='{self.name}', status='{self.status}')"
    
    def __repr__(self) -> str:
        """Developer representation of the task."""
        return self.__str__()