"""Main framework class that orchestrates all Feriq components."""

from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import uuid

from .agent import FeriqAgent, AgentState
from .task import FeriqTask, TaskStatus, TaskType
from .goal import Goal, GoalStatus, GoalPriority
from .role import Role, RoleType
from .plan import Plan, PlanStatus

# Import components (will be created next)
from ..components.role_designer import DynamicRoleDesigner
from ..components.task_designer import TaskDesigner
from ..components.task_allocator import TaskAllocator
from ..components.plan_designer import PlanDesigner
from ..components.plan_observer import PlanObserver
from ..components.orchestrator import WorkflowOrchestrator
from ..components.choreographer import Choreographer
from ..components.reasoner import Reasoner

from ..utils.logger import get_logger
from ..utils.config import Config


class FeriqFramework(BaseModel):
    """
    Main framework class that orchestrates all Feriq components for
    collaborative AI agent coordination and task execution.
    """
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
                   description="Unique identifier for this framework instance")
    name: str = Field(default="Feriq Framework", 
                     description="Name of this framework instance")
    version: str = Field(default="0.1.0", description="Framework version")
    
    # Component instances
    role_designer: DynamicRoleDesigner = Field(default_factory=DynamicRoleDesigner,
                                              description="Dynamic role designer component")
    task_designer: TaskDesigner = Field(default_factory=TaskDesigner,
                                       description="Task designer component")
    task_allocator: TaskAllocator = Field(default_factory=TaskAllocator,
                                         description="Task allocator component")
    plan_designer: PlanDesigner = Field(default_factory=PlanDesigner,
                                       description="Plan designer component")
    plan_observer: PlanObserver = Field(default_factory=PlanObserver,
                                       description="Plan observer component")
    orchestrator: WorkflowOrchestrator = Field(default_factory=WorkflowOrchestrator,
                                              description="Workflow orchestrator")
    choreographer: Choreographer = Field(default_factory=Choreographer,
                                        description="Agent choreographer")
    reasoner: Reasoner = Field(default_factory=Reasoner,
                              description="Reasoning engine")
    
    # Data storage
    agents: Dict[str, FeriqAgent] = Field(default_factory=dict,
                                         description="Registered agents")
    roles: Dict[str, Role] = Field(default_factory=dict,
                                  description="Available roles")
    tasks: Dict[str, FeriqTask] = Field(default_factory=dict,
                                       description="All tasks")
    goals: Dict[str, Goal] = Field(default_factory=dict,
                                  description="All goals")
    plans: Dict[str, Plan] = Field(default_factory=dict,
                                  description="All plans")
    
    # Framework state
    is_running: bool = Field(default=False, description="Whether framework is running")
    created_at: datetime = Field(default_factory=datetime.now,
                                description="Framework creation time")
    last_activity: datetime = Field(default_factory=datetime.now,
                                   description="Last activity timestamp")
    
    # Configuration
    config: Config = Field(default_factory=Config, description="Framework configuration")
    logger: Any = Field(default_factory=lambda: get_logger("FeriqFramework"),
                       description="Framework logger")
    
    # Metrics and monitoring
    execution_metrics: Dict[str, Any] = Field(default_factory=dict,
                                            description="Framework execution metrics")
    performance_history: List[Dict[str, Any]] = Field(default_factory=list,
                                                     description="Performance history")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __init__(self, **kwargs):
        """Initialize the Feriq framework."""
        super().__init__(**kwargs)
        self.logger.info(f"Feriq Framework {self.version} initialized with ID: {self.id}")
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all framework components."""
        # Set framework reference in components
        self.role_designer.framework = self
        self.task_designer.framework = self
        self.task_allocator.framework = self
        self.plan_designer.framework = self
        self.plan_observer.framework = self
        self.orchestrator.framework = self
        self.choreographer.framework = self
        self.reasoner.framework = self
        
        self.logger.info("All framework components initialized")
    
    # Agent Management
    def create_agent(self, name: str, description: str = "", 
                    base_capabilities: Dict[str, float] = None,
                    **kwargs) -> FeriqAgent:
        """Create and register a new agent."""
        agent = FeriqAgent(
            name=name,
            description=description,
            base_capabilities=base_capabilities or {},
            **kwargs
        )
        
        self.agents[agent.id] = agent
        self.last_activity = datetime.now()
        
        self.logger.info(f"Created agent: {agent.name} (ID: {agent.id})")
        return agent
    
    def register_agent(self, agent: FeriqAgent) -> None:
        """Register an existing agent with the framework."""
        self.agents[agent.id] = agent
        self.last_activity = datetime.now()
        self.logger.info(f"Registered agent: {agent.name} (ID: {agent.id})")
    
    def get_agent(self, agent_id: str) -> Optional[FeriqAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_agents_by_state(self, state: AgentState) -> List[FeriqAgent]:
        """Get all agents in a specific state."""
        return [agent for agent in self.agents.values() if agent.state == state]
    
    def get_available_agents(self) -> List[FeriqAgent]:
        """Get all available (idle) agents."""
        return self.get_agents_by_state(AgentState.IDLE)
    
    # Role Management
    def create_role(self, name: str, role_type: RoleType, description: str,
                   **kwargs) -> Role:
        """Create and register a new role."""
        role = Role(
            name=name,
            role_type=role_type,
            description=description,
            **kwargs
        )
        
        self.roles[role.name] = role
        self.last_activity = datetime.now()
        
        self.logger.info(f"Created role: {role.name}")
        return role
    
    def assign_role_to_agent(self, agent_id: str, role_name: str) -> bool:
        """Assign a role to an agent."""
        agent = self.get_agent(agent_id)
        role = self.roles.get(role_name)
        
        if not agent or not role:
            self.logger.error(f"Failed to assign role: agent={agent_id}, role={role_name}")
            return False
        
        success = agent.assign_role(role)
        if success:
            self.logger.info(f"Assigned role {role_name} to agent {agent.name}")
        
        return success
    
    def get_dynamic_role(self, requirements: Dict[str, Any]) -> Optional[Role]:
        """Get or create a role dynamically based on requirements."""
        return self.role_designer.design_role(requirements)
    
    # Task Management
    def create_task(self, name: str, description: str, goal_id: str = None,
                   **kwargs) -> FeriqTask:
        """Create and register a new task."""
        task = FeriqTask(
            name=name,
            description=description,
            goal_id=goal_id,
            **kwargs
        )
        
        self.tasks[task.id] = task
        self.last_activity = datetime.now()
        
        self.logger.info(f"Created task: {task.name} (ID: {task.id})")
        return task
    
    def assign_task(self, task_id: str, agent_id: str = None) -> bool:
        """Assign a task to an agent (auto-assign if agent_id not provided)."""
        task = self.tasks.get(task_id)
        if not task:
            self.logger.error(f"Task not found: {task_id}")
            return False
        
        if agent_id:
            # Direct assignment
            agent = self.get_agent(agent_id)
            if not agent:
                self.logger.error(f"Agent not found: {agent_id}")
                return False
            
            success = agent.assign_task(task_id)
            if success:
                task.assign_to_agent(agent_id)
                self.logger.info(f"Assigned task {task.name} to agent {agent.name}")
            
            return success
        else:
            # Auto-assignment using task allocator
            return self.task_allocator.allocate_task(task_id)
    
    def complete_task(self, task_id: str, result: Any = None, 
                     success: bool = True) -> None:
        """Mark a task as completed."""
        task = self.tasks.get(task_id)
        if not task:
            self.logger.error(f"Task not found: {task_id}")
            return
        
        # Update task
        if success:
            task.complete_task(result)
        else:
            task.fail_task("Task failed during execution")
        
        # Update agent
        if task.assigned_agent_id:
            agent = self.get_agent(task.assigned_agent_id)
            if agent:
                agent.complete_task(task_id, success)
        
        self.last_activity = datetime.now()
        self.logger.info(f"Task {task.name} marked as {'completed' if success else 'failed'}")
    
    # Goal Management
    def create_goal(self, name: str, description: str, 
                   priority: GoalPriority = GoalPriority.MEDIUM,
                   **kwargs) -> Goal:
        """Create and register a new goal."""
        goal = Goal(
            name=name,
            description=description,
            priority=priority,
            **kwargs
        )
        
        self.goals[goal.id] = goal
        self.last_activity = datetime.now()
        
        self.logger.info(f"Created goal: {goal.name} (ID: {goal.id})")
        return goal
    
    def execute_goal(self, goal: Union[Goal, str], auto_plan: bool = True) -> Dict[str, Any]:
        """Execute a goal with automatic planning and task creation."""
        if isinstance(goal, str):
            # Treat as goal description and create goal
            goal_obj = self.create_goal(
                name=f"Goal: {goal[:50]}...",
                description=goal
            )
        else:
            goal_obj = goal
            if goal_obj.id not in self.goals:
                self.goals[goal_obj.id] = goal_obj
        
        self.logger.info(f"Starting execution of goal: {goal_obj.name}")
        
        # Update goal status
        goal_obj.update_status(GoalStatus.PLANNING)
        
        result = {
            "goal_id": goal_obj.id,
            "status": "started",
            "plan_id": None,
            "task_ids": [],
            "message": "Goal execution started"
        }
        
        if auto_plan:
            # Create execution plan
            plan = self.plan_designer.create_plan_for_goal(goal_obj.id)
            if plan:
                result["plan_id"] = plan.id
                result["task_ids"] = plan.task_ids
                
                # Start plan execution
                execution_result = self.execute_plan(plan.id)
                result.update(execution_result)
            else:
                result["status"] = "failed"
                result["message"] = "Failed to create execution plan"
        
        return result
    
    # Plan Management
    def create_plan(self, name: str, description: str, goal_ids: List[str] = None,
                   **kwargs) -> Plan:
        """Create and register a new plan."""
        plan = Plan(
            name=name,
            description=description,
            goal_ids=goal_ids or [],
            **kwargs
        )
        
        self.plans[plan.id] = plan
        self.last_activity = datetime.now()
        
        self.logger.info(f"Created plan: {plan.name} (ID: {plan.id})")
        return plan
    
    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a plan using the orchestrator."""
        plan = self.plans.get(plan_id)
        if not plan:
            self.logger.error(f"Plan not found: {plan_id}")
            return {"status": "error", "message": "Plan not found"}
        
        self.logger.info(f"Starting execution of plan: {plan.name}")
        return self.orchestrator.execute_plan(plan_id)
    
    def monitor_plan(self, plan_id: str) -> Dict[str, Any]:
        """Monitor plan execution progress."""
        return self.plan_observer.get_plan_status(plan_id)
    
    # High-level Framework Operations
    def start(self) -> None:
        """Start the framework."""
        if self.is_running:
            self.logger.warning("Framework is already running")
            return
        
        self.is_running = True
        self.last_activity = datetime.now()
        
        # Start all components
        self.orchestrator.start()
        self.plan_observer.start()
        self.choreographer.start()
        
        self.logger.info("Feriq Framework started successfully")
    
    def stop(self) -> None:
        """Stop the framework."""
        if not self.is_running:
            self.logger.warning("Framework is not running")
            return
        
        self.is_running = False
        
        # Stop all components
        self.orchestrator.stop()
        self.plan_observer.stop()
        self.choreographer.stop()
        
        self.logger.info("Feriq Framework stopped")
    
    def pause(self) -> None:
        """Pause framework operations."""
        self.orchestrator.pause()
        self.plan_observer.pause()
        self.logger.info("Framework operations paused")
    
    def resume(self) -> None:
        """Resume framework operations."""
        self.orchestrator.resume()
        self.plan_observer.resume()
        self.logger.info("Framework operations resumed")
    
    # Reasoning and Decision Making
    def reason_about(self, context: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Use the reasoner to analyze a situation and provide insights."""
        return self.reasoner.reason(context, question)
    
    def make_framework_decision(self, options: List[Dict[str, Any]], 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a framework-level decision."""
        return self.reasoner.make_decision(options, context or {})
    
    # Collaboration and Coordination
    def orchestrate_collaboration(self, agent_ids: List[str], task_id: str,
                                collaboration_type: str = "joint") -> bool:
        """Orchestrate collaboration between multiple agents."""
        return self.choreographer.orchestrate_collaboration(
            agent_ids, task_id, collaboration_type
        )
    
    def coordinate_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Coordinate a complex workflow."""
        return self.orchestrator.coordinate_workflow(workflow_definition)
    
    # Monitoring and Analytics
    def get_framework_status(self) -> Dict[str, Any]:
        """Get comprehensive framework status."""
        return {
            "framework_id": self.id,
            "name": self.name,
            "version": self.version,
            "is_running": self.is_running,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "agents": {
                "total": len(self.agents),
                "idle": len(self.get_agents_by_state(AgentState.IDLE)),
                "busy": len(self.get_agents_by_state(AgentState.BUSY)),
                "collaborating": len(self.get_agents_by_state(AgentState.COLLABORATING))
            },
            "tasks": {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                "in_progress": len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
                "completed": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
            },
            "goals": {
                "total": len(self.goals),
                "pending": len([g for g in self.goals.values() if g.status == GoalStatus.PENDING]),
                "in_progress": len([g for g in self.goals.values() if g.status == GoalStatus.IN_PROGRESS]),
                "completed": len([g for g in self.goals.values() if g.status == GoalStatus.COMPLETED])
            },
            "plans": {
                "total": len(self.plans),
                "active": len([p for p in self.plans.values() if p.status == PlanStatus.ACTIVE]),
                "completed": len([p for p in self.plans.values() if p.status == PlanStatus.COMPLETED])
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get framework performance metrics."""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Calculate average agent efficiency
        agent_efficiencies = [agent.efficiency_score for agent in self.agents.values()]
        avg_efficiency = sum(agent_efficiencies) / len(agent_efficiencies) if agent_efficiencies else 0.0
        
        return {
            "task_success_rate": success_rate,
            "total_tasks_processed": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "average_agent_efficiency": avg_efficiency,
            "active_collaborations": len(self.get_agents_by_state(AgentState.COLLABORATING)),
            "framework_uptime": (datetime.now() - self.created_at).total_seconds(),
            "last_activity": self.last_activity.isoformat()
        }
    
    # Utility Methods
    def export_configuration(self) -> Dict[str, Any]:
        """Export framework configuration."""
        return {
            "framework": {
                "id": self.id,
                "name": self.name,
                "version": self.version,
                "config": self.config.to_dict()
            },
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "roles": [role.to_dict() for role in self.roles.values()],
            "goals": [goal.to_dict() for goal in self.goals.values()],
            "plans": [plan.to_dict() for plan in self.plans.values()]
        }
    
    def import_configuration(self, config_data: Dict[str, Any]) -> None:
        """Import framework configuration."""
        # Import agents
        if "agents" in config_data:
            for agent_data in config_data["agents"]:
                agent = FeriqAgent.from_dict(agent_data)
                self.register_agent(agent)
        
        # Import roles
        if "roles" in config_data:
            for role_data in config_data["roles"]:
                role = Role.from_dict(role_data)
                self.roles[role.name] = role
        
        # Import goals
        if "goals" in config_data:
            for goal_data in config_data["goals"]:
                goal = Goal.from_dict(goal_data)
                self.goals[goal.id] = goal
        
        # Import plans
        if "plans" in config_data:
            for plan_data in config_data["plans"]:
                plan = Plan.from_dict(plan_data)
                self.plans[plan.id] = plan
        
        self.logger.info("Configuration imported successfully")
    
    def __str__(self) -> str:
        """String representation of the framework."""
        return f"FeriqFramework(name='{self.name}', agents={len(self.agents)}, running={self.is_running})"
    
    def __repr__(self) -> str:
        """Developer representation of the framework."""
        return self.__str__()