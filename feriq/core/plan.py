"""Plan class for representing execution plans in the Feriq framework."""

from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime, timedelta
import networkx as nx


class PlanStatus(str, Enum):
    """Status states for plan execution."""
    DRAFT = "draft"
    VALIDATED = "validated"
    APPROVED = "approved"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanType(str, Enum):
    """Types of execution plans."""
    SEQUENTIAL = "sequential"    # Tasks executed in sequence
    PARALLEL = "parallel"       # Tasks executed in parallel
    HYBRID = "hybrid"          # Mix of sequential and parallel
    ADAPTIVE = "adaptive"      # Plan changes based on results
    ITERATIVE = "iterative"    # Repeated cycles
    CONTINGENCY = "contingency" # Backup plan


class ExecutionStrategy(str, Enum):
    """Strategy for plan execution."""
    IMMEDIATE = "immediate"     # Execute immediately when approved
    SCHEDULED = "scheduled"     # Execute at a specific time
    TRIGGERED = "triggered"     # Execute when conditions are met
    MANUAL = "manual"          # Execute manually by user


class MilestoneStatus(str, Enum):
    """Status of a milestone."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Milestone(BaseModel):
    """A milestone in the plan execution."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()),
                   description="Unique identifier for this milestone")
    name: str = Field(..., description="Name of the milestone")
    description: str = Field(..., description="Description of what this milestone represents")
    status: MilestoneStatus = Field(default=MilestoneStatus.PENDING,
                                  description="Current status of the milestone")
    target_date: Optional[datetime] = Field(default=None,
                                          description="Target completion date")
    completed_at: Optional[datetime] = Field(default=None,
                                           description="When this milestone was completed")
    dependencies: List[str] = Field(default_factory=list,
                                  description="IDs of milestones this depends on")
    tasks: List[str] = Field(default_factory=list,
                           description="Task IDs associated with this milestone")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                   description="Additional milestone metadata")


class PlanResource(BaseModel):
    """Represents a resource allocation in a plan."""
    resource_type: str = Field(..., description="Type of resource")
    quantity: float = Field(default=1.0, description="Required quantity")
    allocated_from: Optional[datetime] = Field(default=None, description="Start of allocation")
    allocated_until: Optional[datetime] = Field(default=None, description="End of allocation")
    cost: float = Field(default=0.0, description="Cost of resource")


class PlanMilestone(BaseModel):
    """Represents a milestone in a plan."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Milestone ID")
    name: str = Field(..., description="Milestone name")
    description: str = Field(default="", description="Milestone description")
    target_date: Optional[datetime] = Field(default=None, description="Target completion date")
    criteria: List[str] = Field(default_factory=list, description="Completion criteria")
    task_ids: List[str] = Field(default_factory=list, description="Related task IDs")
    completed: bool = Field(default=False, description="Whether milestone is completed")
    completion_date: Optional[datetime] = Field(default=None, description="Actual completion date")


class PlanRisk(BaseModel):
    """Represents a risk in plan execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Risk ID")
    name: str = Field(..., description="Risk name")
    description: str = Field(..., description="Risk description")
    probability: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability of occurrence")
    impact: float = Field(default=0.5, ge=0.0, le=1.0, description="Impact severity")
    mitigation_strategy: str = Field(default="", description="How to mitigate this risk")
    contingency_plan: str = Field(default="", description="What to do if risk occurs")
    status: str = Field(default="identified", description="Risk status")


class Plan(BaseModel):
    """
    Represents an execution plan that coordinates multiple tasks and agents
    to achieve one or more goals in the Feriq framework.
    """
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
                   description="Unique identifier for this plan")
    name: str = Field(..., description="Human-readable name for the plan")
    description: str = Field(..., description="Detailed description of the plan")
    version: str = Field(default="1.0", description="Plan version")
    
    # Classification and strategy
    plan_type: PlanType = Field(default=PlanType.HYBRID, 
                               description="Type of execution plan")
    execution_strategy: ExecutionStrategy = Field(default=ExecutionStrategy.IMMEDIATE,
                                                 description="How the plan should be executed")
    
    # Status and lifecycle
    status: PlanStatus = Field(default=PlanStatus.DRAFT, 
                              description="Current plan status")
    created_at: datetime = Field(default_factory=datetime.now, 
                                description="When this plan was created")
    updated_at: datetime = Field(default_factory=datetime.now, 
                                description="When this plan was last updated")
    approved_at: Optional[datetime] = Field(default=None, 
                                           description="When plan was approved")
    started_at: Optional[datetime] = Field(default=None, 
                                          description="When execution started")
    completed_at: Optional[datetime] = Field(default=None, 
                                            description="When execution completed")
    
    # Goals and objectives
    goal_ids: List[str] = Field(default_factory=list,
                               description="IDs of goals this plan addresses")
    primary_goal_id: Optional[str] = Field(default=None,
                                          description="ID of primary goal")
    success_criteria: List[str] = Field(default_factory=list,
                                       description="Overall success criteria")
    
    # Tasks and workflow
    task_ids: List[str] = Field(default_factory=list,
                               description="IDs of tasks in this plan")
    task_dependencies: Dict[str, List[str]] = Field(default_factory=dict,
                                                   description="Task dependency relationships")
    critical_path: List[str] = Field(default_factory=list,
                                   description="Critical path task IDs")
    
    # Agent assignments
    agent_assignments: Dict[str, List[str]] = Field(default_factory=dict,
                                                   description="Agent ID to task IDs mapping")
    role_assignments: Dict[str, List[str]] = Field(default_factory=dict,
                                                  description="Role ID to task IDs mapping")
    
    # Timeline and scheduling
    estimated_start_date: Optional[datetime] = Field(default=None,
                                                    description="Estimated start date")
    estimated_end_date: Optional[datetime] = Field(default=None,
                                                  description="Estimated completion date")
    actual_start_date: Optional[datetime] = Field(default=None,
                                                 description="Actual start date")
    actual_end_date: Optional[datetime] = Field(default=None,
                                               description="Actual completion date")
    
    # Resources and constraints
    required_resources: List[PlanResource] = Field(default_factory=list,
                                                  description="Resources required for execution")
    budget: float = Field(default=0.0, description="Budget allocation")
    constraints: List[str] = Field(default_factory=list,
                                 description="Execution constraints")
    
    # Progress tracking
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0,
                                     description="Overall progress percentage")
    completed_task_ids: Set[str] = Field(default_factory=set,
                                        description="IDs of completed tasks")
    failed_task_ids: Set[str] = Field(default_factory=set,
                                     description="IDs of failed tasks")
    
    # Milestones and monitoring
    milestones: List[PlanMilestone] = Field(default_factory=list,
                                           description="Plan milestones")
    risks: List[PlanRisk] = Field(default_factory=list,
                                 description="Identified risks")
    
    # Quality and performance
    quality_metrics: Dict[str, float] = Field(default_factory=dict,
                                            description="Quality measurements")
    performance_metrics: Dict[str, float] = Field(default_factory=dict,
                                                 description="Performance measurements")
    
    # Adaptability
    adaptation_triggers: List[str] = Field(default_factory=list,
                                         description="Conditions that trigger plan adaptation")
    fallback_plan_id: Optional[str] = Field(default=None,
                                           description="ID of fallback plan")
    
    # Metadata
    created_by: Optional[str] = Field(default=None, description="Creator identifier")
    approved_by: Optional[str] = Field(default=None, description="Approver identifier")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                   description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v)
        }
    
    def add_goal(self, goal_id: str, is_primary: bool = False) -> None:
        """Add a goal to this plan."""
        if goal_id not in self.goal_ids:
            self.goal_ids.append(goal_id)
        
        if is_primary:
            self.primary_goal_id = goal_id
        
        self.updated_at = datetime.now()
    
    def add_task(self, task_id: str) -> None:
        """Add a task to this plan."""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.updated_at = datetime.now()
    
    def add_task_dependency(self, task_id: str, depends_on: List[str]) -> None:
        """Add dependencies for a task."""
        self.task_dependencies[task_id] = depends_on
        self.updated_at = datetime.now()
    
    def assign_task_to_agent(self, task_id: str, agent_id: str) -> None:
        """Assign a task to an agent."""
        if agent_id not in self.agent_assignments:
            self.agent_assignments[agent_id] = []
        
        if task_id not in self.agent_assignments[agent_id]:
            self.agent_assignments[agent_id].append(task_id)
        
        self.updated_at = datetime.now()
    
    def assign_task_to_role(self, task_id: str, role_id: str) -> None:
        """Assign a task to a role."""
        if role_id not in self.role_assignments:
            self.role_assignments[role_id] = []
        
        if task_id not in self.role_assignments[role_id]:
            self.role_assignments[role_id].append(task_id)
        
        self.updated_at = datetime.now()
    
    def add_milestone(self, milestone: PlanMilestone) -> None:
        """Add a milestone to the plan."""
        self.milestones.append(milestone)
        self.updated_at = datetime.now()
    
    def add_risk(self, risk: PlanRisk) -> None:
        """Add a risk to the plan."""
        self.risks.append(risk)
        self.updated_at = datetime.now()
    
    def calculate_critical_path(self) -> List[str]:
        """Calculate and update the critical path."""
        if not self.task_ids or not self.task_dependencies:
            return []
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all tasks as nodes
        for task_id in self.task_ids:
            G.add_node(task_id)
        
        # Add dependencies as edges
        for task_id, dependencies in self.task_dependencies.items():
            for dep_id in dependencies:
                if dep_id in self.task_ids:
                    G.add_edge(dep_id, task_id)
        
        try:
            # Find the longest path (critical path)
            # This is a simplified version - real implementation would use task durations
            if G.nodes():
                # Find source nodes (no incoming edges)
                sources = [n for n in G.nodes() if G.in_degree(n) == 0]
                # Find sink nodes (no outgoing edges)
                sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
                
                if sources and sinks:
                    # For simplicity, find longest path from first source to first sink
                    try:
                        path = nx.shortest_path(G, sources[0], sinks[0])
                        self.critical_path = path
                        return path
                    except nx.NetworkXNoPath:
                        # No path exists, return empty
                        self.critical_path = []
                        return []
                
        except Exception:
            # Handle any graph-related errors
            pass
        
        self.critical_path = []
        return []
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute (dependencies met)."""
        ready_tasks = []
        
        for task_id in self.task_ids:
            if task_id in self.completed_task_ids or task_id in self.failed_task_ids:
                continue
            
            # Check if all dependencies are completed
            dependencies = self.task_dependencies.get(task_id, [])
            if all(dep_id in self.completed_task_ids for dep_id in dependencies):
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def mark_task_completed(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id in self.task_ids:
            self.completed_task_ids.add(task_id)
            if task_id in self.failed_task_ids:
                self.failed_task_ids.remove(task_id)
            
            self.update_progress()
            self.updated_at = datetime.now()
    
    def mark_task_failed(self, task_id: str) -> None:
        """Mark a task as failed."""
        if task_id in self.task_ids:
            self.failed_task_ids.add(task_id)
            if task_id in self.completed_task_ids:
                self.completed_task_ids.remove(task_id)
            
            self.update_progress()
            self.updated_at = datetime.now()
    
    def update_progress(self) -> None:
        """Update the overall progress percentage."""
        total_tasks = len(self.task_ids)
        if total_tasks == 0:
            self.progress_percentage = 0.0
            return
        
        completed_tasks = len(self.completed_task_ids)
        self.progress_percentage = (completed_tasks / total_tasks) * 100.0
        
        # Check if plan is completed
        if completed_tasks == total_tasks and self.status == PlanStatus.ACTIVE:
            self.status = PlanStatus.COMPLETED
            self.completed_at = datetime.now()
    
    def estimate_duration(self, task_durations: Dict[str, float]) -> float:
        """Estimate total plan duration given task durations."""
        if not self.task_ids:
            return 0.0
        
        # Create a simple critical path duration calculation
        critical_path_duration = 0.0
        for task_id in self.critical_path:
            duration = task_durations.get(task_id, 1.0)  # Default 1 hour
            critical_path_duration += duration
        
        return critical_path_duration
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization across the plan."""
        utilization = {}
        
        for resource in self.required_resources:
            # Simple utilization calculation
            if resource.allocated_from and resource.allocated_until:
                duration = (resource.allocated_until - resource.allocated_from).total_seconds() / 3600
                utilization[resource.resource_type] = resource.quantity * duration
            else:
                utilization[resource.resource_type] = resource.quantity
        
        return utilization
    
    def validate_plan(self) -> List[str]:
        """Validate the plan and return list of issues."""
        issues = []
        
        # Check for circular dependencies
        if self.task_dependencies:
            try:
                G = nx.DiGraph(self.task_dependencies)
                if not nx.is_directed_acyclic_graph(G):
                    issues.append("Circular dependencies detected in task relationships")
            except Exception:
                issues.append("Invalid task dependency structure")
        
        # Check for orphaned tasks
        referenced_tasks = set()
        for deps in self.task_dependencies.values():
            referenced_tasks.update(deps)
        referenced_tasks.update(self.task_dependencies.keys())
        
        orphaned = set(self.task_ids) - referenced_tasks
        if orphaned and len(self.task_ids) > 1:
            issues.append(f"Orphaned tasks found: {list(orphaned)}")
        
        # Check resource constraints
        if not self.required_resources and self.task_ids:
            issues.append("No resources allocated for plan execution")
        
        # Check timeline consistency
        if (self.estimated_start_date and self.estimated_end_date and 
            self.estimated_start_date >= self.estimated_end_date):
            issues.append("Estimated end date is before or equal to start date")
        
        return issues
    
    def start_execution(self) -> None:
        """Start plan execution."""
        self.status = PlanStatus.ACTIVE
        self.started_at = datetime.now()
        self.actual_start_date = self.started_at
        self.updated_at = datetime.now()
    
    def pause_execution(self) -> None:
        """Pause plan execution."""
        if self.status == PlanStatus.ACTIVE:
            self.status = PlanStatus.PAUSED
            self.updated_at = datetime.now()
    
    def resume_execution(self) -> None:
        """Resume plan execution."""
        if self.status == PlanStatus.PAUSED:
            self.status = PlanStatus.ACTIVE
            self.updated_at = datetime.now()
    
    def cancel_execution(self) -> None:
        """Cancel plan execution."""
        self.status = PlanStatus.CANCELLED
        self.updated_at = datetime.now()
        if not self.completed_at:
            self.completed_at = datetime.now()
            self.actual_end_date = self.completed_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary representation."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        """Create plan from dictionary representation."""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of the plan."""
        return f"Plan(id='{self.id[:8]}...', name='{self.name}', status='{self.status}')"
    
    def __repr__(self) -> str:
        """Developer representation of the plan."""
        return self.__str__()