"""
Workflow Orchestrator Component

This module implements the central orchestrator that manages workflow execution,
task scheduling, agent coordination, and workflow state management.
"""

from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import asyncio
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from pydantic import BaseModel, Field
import heapq
import time

from ..core.goal import Goal, GoalStatus
from ..core.task import FeriqTask, TaskStatus, TaskPriority
from ..core.plan import Plan, PlanStatus
from ..core.agent import FeriqAgent, AgentStatus
from ..core.role import Role
from ..components.plan_observer import PlanObserver, Observation, ObservationType
from ..utils.logger import FeriqLogger
from ..utils.config import Config


class ExecutionStrategy(str, Enum):
    """Strategies for workflow execution."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DYNAMIC = "dynamic"
    PRIORITY_BASED = "priority_based"
    RESOURCE_AWARE = "resource_aware"


class WorkflowEvent(str, Enum):
    """Types of workflow events."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    TASK_SCHEDULED = "task_scheduled"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_ASSIGNED = "agent_assigned"
    AGENT_RELEASED = "agent_released"
    RESOURCE_ALLOCATED = "resource_allocated"
    RESOURCE_RELEASED = "resource_released"


@dataclass
class ScheduledTask:
    """Represents a task scheduled for execution."""
    task: FeriqTask
    assigned_agent: Optional[FeriqAgent] = None
    scheduled_time: Optional[datetime] = None
    priority_score: float = 0.0
    dependencies_met: bool = False
    resource_reserved: bool = False
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority_score > other.priority_score


@dataclass
class WorkflowState:
    """Represents the current state of workflow execution."""
    workflow_id: str
    plan: Plan
    status: PlanStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    current_phase: str = "initialization"
    active_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    assigned_agents: Dict[str, str] = field(default_factory=dict)  # task_id -> agent_id
    resource_allocations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Handles task scheduling and prioritization."""
    
    def __init__(self, strategy: ExecutionStrategy, logger: FeriqLogger):
        self.strategy = strategy
        self.logger = logger
        self.ready_queue: List[ScheduledTask] = []
        self.waiting_queue: List[ScheduledTask] = []
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def add_task(self, task: FeriqTask) -> ScheduledTask:
        """Add a task to the scheduler."""
        scheduled_task = ScheduledTask(
            task=task,
            priority_score=self._calculate_priority_score(task),
            dependencies_met=len(task.dependencies) == 0
        )
        
        # Build dependency graph
        for dep_id in task.dependencies:
            self.dependency_graph[dep_id].add(task.task_id)
        
        if scheduled_task.dependencies_met:
            heapq.heappush(self.ready_queue, scheduled_task)
        else:
            self.waiting_queue.append(scheduled_task)
        
        self.logger.debug(
            "Task added to scheduler",
            task_id=task.task_id,
            priority_score=scheduled_task.priority_score,
            dependencies_met=scheduled_task.dependencies_met
        )
        
        return scheduled_task
    
    def get_next_task(self) -> Optional[ScheduledTask]:
        """Get the next task to execute."""
        if self.ready_queue:
            return heapq.heappop(self.ready_queue)
        return None
    
    def mark_task_completed(self, task_id: str):
        """Mark a task as completed and update dependencies."""
        # Check if any waiting tasks can now be scheduled
        newly_ready = []
        remaining_waiting = []
        
        for scheduled_task in self.waiting_queue:
            scheduled_task.task.dependencies.discard(task_id)
            if len(scheduled_task.task.dependencies) == 0:
                scheduled_task.dependencies_met = True
                newly_ready.append(scheduled_task)
            else:
                remaining_waiting.append(scheduled_task)
        
        self.waiting_queue = remaining_waiting
        
        # Add newly ready tasks to ready queue
        for task in newly_ready:
            heapq.heappush(self.ready_queue, task)
        
        self.logger.debug(
            "Task completed, dependencies updated",
            completed_task_id=task_id,
            newly_ready_count=len(newly_ready)
        )
    
    def _calculate_priority_score(self, task: FeriqTask) -> float:
        """Calculate priority score for a task."""
        base_score = {
            TaskPriority.LOW: 1.0,
            TaskPriority.MEDIUM: 2.0,
            TaskPriority.HIGH: 3.0,
            TaskPriority.URGENT: 4.0
        }[task.priority]
        
        # Adjust for complexity (higher complexity = higher priority)
        complexity_bonus = task.complexity_score * 0.5
        
        # Adjust for estimated duration (shorter tasks = higher priority)
        duration_penalty = task.estimated_duration.total_seconds() / 3600 * 0.1
        
        return base_score + complexity_bonus - duration_penalty
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "strategy": self.strategy,
            "ready_queue_size": len(self.ready_queue),
            "waiting_queue_size": len(self.waiting_queue),
            "total_dependencies": sum(len(deps) for deps in self.dependency_graph.values())
        }


class ResourceManager:
    """Manages resource allocation and constraints."""
    
    def __init__(self, logger: FeriqLogger):
        self.logger = logger
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.allocations: Dict[str, Dict[str, Any]] = defaultdict(dict)  # resource_type -> {task_id: amount}
        
    def register_resource(self, resource_type: str, capacity: int, properties: Optional[Dict[str, Any]] = None):
        """Register a resource type with capacity."""
        self.resources[resource_type] = {
            "capacity": capacity,
            "available": capacity,
            "properties": properties or {}
        }
        
        self.logger.info(
            "Resource registered",
            resource_type=resource_type,
            capacity=capacity
        )
    
    def can_allocate(self, task_id: str, requirements: Dict[str, int]) -> bool:
        """Check if resources can be allocated for a task."""
        for resource_type, amount in requirements.items():
            if resource_type not in self.resources:
                self.logger.warning(f"Unknown resource type: {resource_type}")
                return False
            
            available = self.resources[resource_type]["available"]
            if available < amount:
                return False
        
        return True
    
    def allocate_resources(self, task_id: str, requirements: Dict[str, int]) -> bool:
        """Allocate resources for a task."""
        if not self.can_allocate(task_id, requirements):
            return False
        
        # Allocate resources
        for resource_type, amount in requirements.items():
            self.resources[resource_type]["available"] -= amount
            self.allocations[resource_type][task_id] = amount
        
        self.logger.info(
            "Resources allocated",
            task_id=task_id,
            requirements=requirements
        )
        
        return True
    
    def release_resources(self, task_id: str):
        """Release resources allocated to a task."""
        for resource_type, allocations in self.allocations.items():
            if task_id in allocations:
                amount = allocations[task_id]
                self.resources[resource_type]["available"] += amount
                del allocations[task_id]
                
                self.logger.info(
                    "Resources released",
                    task_id=task_id,
                    resource_type=resource_type,
                    amount=amount
                )
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        status = {}
        for resource_type, info in self.resources.items():
            allocated = info["capacity"] - info["available"]
            status[resource_type] = {
                "capacity": info["capacity"],
                "available": info["available"],
                "allocated": allocated,
                "utilization": allocated / info["capacity"] if info["capacity"] > 0 else 0
            }
        return status


class AgentManager:
    """Manages agent lifecycle and assignments."""
    
    def __init__(self, logger: FeriqLogger):
        self.logger = logger
        self.agents: Dict[str, FeriqAgent] = {}
        self.agent_assignments: Dict[str, str] = {}  # agent_id -> task_id
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
    def register_agent(self, agent: FeriqAgent):
        """Register an agent for task execution."""
        self.agents[agent.agent_id] = agent
        self.logger.info("Agent registered", agent_id=agent.agent_id, role=agent.role.name if agent.role else None)
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            # Release any assigned tasks
            if agent_id in self.agent_assignments:
                task_id = self.agent_assignments[agent_id]
                self.release_agent(agent_id)
            
            del self.agents[agent_id]
            self.logger.info("Agent unregistered", agent_id=agent_id)
    
    def find_suitable_agent(self, task: FeriqTask) -> Optional[FeriqAgent]:
        """Find the most suitable available agent for a task."""
        available_agents = [
            agent for agent_id, agent in self.agents.items()
            if agent_id not in self.agent_assignments and agent.status == AgentStatus.IDLE
        ]
        
        if not available_agents:
            return None
        
        # Score agents based on suitability
        best_agent = None
        best_score = 0.0
        
        for agent in available_agents:
            score = 0.0
            
            # Role suitability
            if agent.role:
                score += agent.role.calculate_suitability_score(task)
            
            # Capability match
            agent_capabilities = set(agent.capabilities)
            task_capabilities = set(task.required_capabilities)
            capability_overlap = len(agent_capabilities.intersection(task_capabilities))
            score += capability_overlap * 0.2
            
            # Performance history
            score += agent.performance_metrics.get("success_rate", 0.5) * 0.3
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent if best_score > 0.3 else None
    
    def assign_agent(self, agent_id: str, task_id: str) -> bool:
        """Assign an agent to a task."""
        if agent_id not in self.agents:
            self.logger.warning(f"Agent {agent_id} not found")
            return False
        
        if agent_id in self.agent_assignments:
            self.logger.warning(f"Agent {agent_id} already assigned")
            return False
        
        self.agent_assignments[agent_id] = task_id
        self.task_assignments[task_id] = agent_id
        
        # Update agent status
        agent = self.agents[agent_id]
        agent.status = AgentStatus.BUSY
        
        self.logger.info("Agent assigned to task", agent_id=agent_id, task_id=task_id)
        return True
    
    def release_agent(self, agent_id: str):
        """Release an agent from its current assignment."""
        if agent_id in self.agent_assignments:
            task_id = self.agent_assignments[agent_id]
            del self.agent_assignments[agent_id]
            
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            
            # Update agent status
            if agent_id in self.agents:
                self.agents[agent_id].status = AgentStatus.IDLE
            
            self.logger.info("Agent released", agent_id=agent_id, task_id=task_id)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "total_agents": len(self.agents),
            "available_agents": sum(1 for agent in self.agents.values() if agent.status == AgentStatus.IDLE),
            "busy_agents": sum(1 for agent in self.agents.values() if agent.status == AgentStatus.BUSY),
            "agent_assignments": len(self.agent_assignments)
        }


class WorkflowOrchestrator:
    """
    Workflow Orchestrator component that manages workflow execution.
    
    This component provides:
    - Central coordination of plan execution
    - Task scheduling and agent assignment
    - Resource management and allocation
    - Real-time workflow monitoring
    - Error handling and recovery
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = FeriqLogger("WorkflowOrchestrator", self.config)
        
        # Core managers
        self.task_scheduler = TaskScheduler(ExecutionStrategy.DYNAMIC, self.logger)
        self.resource_manager = ResourceManager(self.logger)
        self.agent_manager = AgentManager(self.logger)
        
        # Workflow state management
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.completed_workflows: Dict[str, WorkflowState] = {}
        
        # Execution control
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.orchestration_thread: Optional[threading.Thread] = None
        
        # Event system
        self.event_handlers: Dict[WorkflowEvent, List[Callable]] = defaultdict(list)
        
        # Observer integration
        self.plan_observer: Optional[PlanObserver] = None
        
        # Performance tracking
        self.execution_metrics: Dict[str, Any] = defaultdict(float)
        
        self.logger.info("WorkflowOrchestrator initialized successfully")
    
    def set_plan_observer(self, observer: PlanObserver):
        """Set the plan observer for monitoring integration."""
        self.plan_observer = observer
        self.logger.info("Plan observer integrated")
    
    def register_agent(self, agent: FeriqAgent):
        """Register an agent for workflow execution."""
        self.agent_manager.register_agent(agent)
        self._emit_event(WorkflowEvent.AGENT_ASSIGNED, {"agent_id": agent.agent_id})
    
    def register_resource(self, resource_type: str, capacity: int, properties: Optional[Dict[str, Any]] = None):
        """Register a resource type."""
        self.resource_manager.register_resource(resource_type, capacity, properties)
    
    def start_workflow(self, plan: Plan, strategy: ExecutionStrategy = ExecutionStrategy.DYNAMIC) -> str:
        """
        Start executing a workflow plan.
        
        Args:
            plan: The plan to execute
            strategy: Execution strategy to use
            
        Returns:
            Workflow ID for tracking
        """
        workflow_id = str(uuid.uuid4())
        
        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            plan=plan,
            status=PlanStatus.IN_PROGRESS,
            start_time=datetime.now(),
            current_phase="task_scheduling"
        )
        
        self.active_workflows[workflow_id] = workflow_state
        
        # Set up scheduler strategy
        self.task_scheduler.strategy = strategy
        
        # Schedule all tasks
        for task in plan.tasks:
            scheduled_task = self.task_scheduler.add_task(task)
            
        # Start plan observation if observer is available
        if self.plan_observer:
            monitor_id = self.plan_observer.start_observing_plan(plan)
            workflow_state.metrics["monitor_id"] = monitor_id
        
        # Start orchestration if not already running
        if not self.running:
            self.start_orchestration()
        
        self._emit_event(WorkflowEvent.WORKFLOW_STARTED, {
            "workflow_id": workflow_id,
            "plan_id": plan.plan_id,
            "task_count": len(plan.tasks)
        })
        
        self.logger.info(
            "Workflow started",
            workflow_id=workflow_id,
            plan_id=plan.plan_id,
            strategy=strategy,
            task_count=len(plan.tasks)
        )
        
        return workflow_id
    
    def start_orchestration(self):
        """Start the orchestration loop."""
        if self.running:
            return
        
        self.running = True
        self.orchestration_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.orchestration_thread.start()
        
        self.logger.info("Orchestration started")
    
    def stop_orchestration(self):
        """Stop the orchestration loop."""
        self.running = False
        if self.orchestration_thread:
            self.orchestration_thread.join(timeout=10)
        
        self.logger.info("Orchestration stopped")
    
    def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.running:
            try:
                self._orchestration_cycle()
                time.sleep(1)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                self.logger.error(f"Error in orchestration loop: {e}")
    
    def _orchestration_cycle(self):
        """Single orchestration cycle."""
        # Check for ready tasks
        next_task_scheduled = self.task_scheduler.get_next_task()
        if next_task_scheduled is None:
            return
        
        task = next_task_scheduled.task
        
        # Check resource availability
        if not self.resource_manager.can_allocate(task.task_id, task.resource_requirements):
            # Put task back in queue with lower priority
            next_task_scheduled.priority_score *= 0.9
            heapq.heappush(self.task_scheduler.ready_queue, next_task_scheduled)
            return
        
        # Find suitable agent
        agent = self.agent_manager.find_suitable_agent(task)
        if agent is None:
            # Put task back in queue
            heapq.heappush(self.task_scheduler.ready_queue, next_task_scheduled)
            return
        
        # Allocate resources
        if self.resource_manager.allocate_resources(task.task_id, task.resource_requirements):
            # Assign agent
            if self.agent_manager.assign_agent(agent.agent_id, task.task_id):
                # Execute task
                self._execute_task(task, agent, next_task_scheduled)
            else:
                # Release resources if agent assignment failed
                self.resource_manager.release_resources(task.task_id)
    
    def _execute_task(self, task: FeriqTask, agent: FeriqAgent, scheduled_task: ScheduledTask):
        """Execute a task with an assigned agent."""
        # Update workflow state
        for workflow_state in self.active_workflows.values():
            if task.goal_id == workflow_state.plan.goal_id:
                workflow_state.active_tasks.add(task.task_id)
                workflow_state.assigned_agents[task.task_id] = agent.agent_id
                break
        
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        # Record in observer
        if self.plan_observer:
            for workflow_state in self.active_workflows.values():
                if task.goal_id == workflow_state.plan.goal_id:
                    monitor_id = workflow_state.metrics.get("monitor_id")
                    if monitor_id:
                        # Find the monitor and record task start
                        for monitor in self.plan_observer.active_monitors.values():
                            if monitor.plan.goal_id == task.goal_id:
                                monitor.record_task_start(task.task_id)
                                break
                    break
        
        self._emit_event(WorkflowEvent.TASK_STARTED, {
            "task_id": task.task_id,
            "agent_id": agent.agent_id
        })
        
        # Submit task for execution
        future = self.executor.submit(self._run_task, task, agent, scheduled_task)
        future.add_done_callback(lambda f: self._task_completed_callback(f, task, agent, scheduled_task))
        
        self.logger.info(
            "Task execution started",
            task_id=task.task_id,
            agent_id=agent.agent_id
        )
    
    def _run_task(self, task: FeriqTask, agent: FeriqAgent, scheduled_task: ScheduledTask) -> bool:
        """Run a task with the assigned agent."""
        try:
            # This would integrate with the actual agent execution
            # For now, we'll simulate task execution
            
            # Simulate variable execution time based on complexity
            execution_time = task.estimated_duration.total_seconds() * (0.8 + 0.4 * task.complexity_score)
            time.sleep(min(execution_time, 30))  # Cap at 30 seconds for simulation
            
            # Simulate success/failure based on agent performance and task complexity
            success_probability = agent.performance_metrics.get("success_rate", 0.8) * (1.1 - task.complexity_score)
            import random
            success = random.random() < success_probability
            
            if success:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                # Update agent performance
                if "tasks_completed" not in agent.performance_metrics:
                    agent.performance_metrics["tasks_completed"] = 0
                agent.performance_metrics["tasks_completed"] += 1
                
                return True
            else:
                # Handle failure
                if scheduled_task.retry_count < scheduled_task.max_retries:
                    scheduled_task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    return False
                else:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    task.failure_reason = "Max retries exceeded"
                    
                    # Update agent performance
                    if "tasks_failed" not in agent.performance_metrics:
                        agent.performance_metrics["tasks_failed"] = 0
                    agent.performance_metrics["tasks_failed"] += 1
                    
                    return False
                    
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.failure_reason = str(e)
            self.logger.error(f"Task execution failed: {e}", task_id=task.task_id)
            return False
    
    def _task_completed_callback(self, future: Future, task: FeriqTask, agent: FeriqAgent, scheduled_task: ScheduledTask):
        """Callback when task execution completes."""
        try:
            success = future.result()
            
            # Update workflow state
            for workflow_state in self.active_workflows.values():
                if task.goal_id == workflow_state.plan.goal_id:
                    workflow_state.active_tasks.discard(task.task_id)
                    
                    if success:
                        workflow_state.completed_tasks.add(task.task_id)
                        self.task_scheduler.mark_task_completed(task.task_id)
                        self._emit_event(WorkflowEvent.TASK_COMPLETED, {
                            "task_id": task.task_id,
                            "agent_id": agent.agent_id
                        })
                    else:
                        if scheduled_task.retry_count < scheduled_task.max_retries:
                            # Re-schedule for retry
                            heapq.heappush(self.task_scheduler.ready_queue, scheduled_task)
                        else:
                            workflow_state.failed_tasks.add(task.task_id)
                            self._emit_event(WorkflowEvent.TASK_FAILED, {
                                "task_id": task.task_id,
                                "agent_id": agent.agent_id,
                                "reason": task.failure_reason
                            })
                    
                    # Check if workflow is complete
                    self._check_workflow_completion(workflow_state)
                    break
            
            # Release resources and agent
            self.resource_manager.release_resources(task.task_id)
            self.agent_manager.release_agent(agent.agent_id)
            
            # Record in observer
            if self.plan_observer and success:
                for workflow_state in self.active_workflows.values():
                    if task.goal_id == workflow_state.plan.goal_id:
                        monitor_id = workflow_state.metrics.get("monitor_id")
                        if monitor_id:
                            # Find the monitor and record task completion
                            for monitor in self.plan_observer.active_monitors.values():
                                if monitor.plan.goal_id == task.goal_id:
                                    monitor.record_task_completion(task.task_id)
                                    break
                        break
            
            self.logger.info(
                "Task completed",
                task_id=task.task_id,
                agent_id=agent.agent_id,
                success=success,
                retry_count=scheduled_task.retry_count
            )
            
        except Exception as e:
            self.logger.error(f"Error in task completion callback: {e}")
    
    def _check_workflow_completion(self, workflow_state: WorkflowState):
        """Check if a workflow has completed."""
        total_tasks = len(workflow_state.plan.tasks)
        completed_count = len(workflow_state.completed_tasks)
        failed_count = len(workflow_state.failed_tasks)
        
        if completed_count + failed_count >= total_tasks:
            # Workflow finished
            workflow_state.end_time = datetime.now()
            
            if failed_count == 0:
                workflow_state.status = PlanStatus.COMPLETED
                self._emit_event(WorkflowEvent.WORKFLOW_COMPLETED, {
                    "workflow_id": workflow_state.workflow_id,
                    "completed_tasks": completed_count,
                    "failed_tasks": failed_count
                })
            else:
                workflow_state.status = PlanStatus.FAILED
                self._emit_event(WorkflowEvent.WORKFLOW_FAILED, {
                    "workflow_id": workflow_state.workflow_id,
                    "completed_tasks": completed_count,
                    "failed_tasks": failed_count
                })
            
            # Move to completed workflows
            self.completed_workflows[workflow_state.workflow_id] = workflow_state
            del self.active_workflows[workflow_state.workflow_id]
            
            # Stop observing if observer is available
            if self.plan_observer:
                monitor_id = workflow_state.metrics.get("monitor_id")
                if monitor_id:
                    self.plan_observer.stop_observing_plan(monitor_id)
            
            self.logger.info(
                "Workflow completed",
                workflow_id=workflow_state.workflow_id,
                status=workflow_state.status,
                completed_tasks=completed_count,
                failed_tasks=failed_count,
                duration=(workflow_state.end_time - workflow_state.start_time).total_seconds()
            )
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a workflow execution."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow_state = self.active_workflows[workflow_id]
        workflow_state.status = PlanStatus.PAUSED
        
        self._emit_event(WorkflowEvent.WORKFLOW_PAUSED, {"workflow_id": workflow_id})
        
        self.logger.info("Workflow paused", workflow_id=workflow_id)
        return True
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow_state = self.active_workflows[workflow_id]
        if workflow_state.status != PlanStatus.PAUSED:
            return False
        
        workflow_state.status = PlanStatus.IN_PROGRESS
        
        self._emit_event(WorkflowEvent.WORKFLOW_RESUMED, {"workflow_id": workflow_id})
        
        self.logger.info("Workflow resumed", workflow_id=workflow_id)
        return True
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow execution."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow_state = self.active_workflows[workflow_id]
        workflow_state.status = PlanStatus.CANCELLED
        workflow_state.end_time = datetime.now()
        
        # Release all resources and agents for this workflow
        for task_id in workflow_state.active_tasks:
            self.resource_manager.release_resources(task_id)
            if task_id in workflow_state.assigned_agents:
                agent_id = workflow_state.assigned_agents[task_id]
                self.agent_manager.release_agent(agent_id)
        
        # Move to completed workflows
        self.completed_workflows[workflow_id] = workflow_state
        del self.active_workflows[workflow_id]
        
        self.logger.info("Workflow cancelled", workflow_id=workflow_id)
        return True
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow_state = self.active_workflows[workflow_id]
        elif workflow_id in self.completed_workflows:
            workflow_state = self.completed_workflows[workflow_id]
        else:
            return None
        
        return {
            "workflow_id": workflow_state.workflow_id,
            "plan_id": workflow_state.plan.plan_id,
            "status": workflow_state.status,
            "start_time": workflow_state.start_time.isoformat(),
            "end_time": workflow_state.end_time.isoformat() if workflow_state.end_time else None,
            "current_phase": workflow_state.current_phase,
            "total_tasks": len(workflow_state.plan.tasks),
            "active_tasks": len(workflow_state.active_tasks),
            "completed_tasks": len(workflow_state.completed_tasks),
            "failed_tasks": len(workflow_state.failed_tasks),
            "progress": len(workflow_state.completed_tasks) / len(workflow_state.plan.tasks) if workflow_state.plan.tasks else 0
        }
    
    def add_event_handler(self, event: WorkflowEvent, handler: Callable[[Dict[str, Any]], None]):
        """Add an event handler for workflow events."""
        self.event_handlers[event].append(handler)
    
    def _emit_event(self, event: WorkflowEvent, data: Dict[str, Any]):
        """Emit a workflow event."""
        for handler in self.event_handlers[event]:
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        return {
            "running": self.running,
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "scheduler_status": self.task_scheduler.get_scheduler_status(),
            "resource_status": self.resource_manager.get_resource_status(),
            "agent_status": self.agent_manager.get_agent_status(),
            "executor_status": {
                "max_workers": self.executor._max_workers,
                "active_threads": threading.active_count()
            }
        }