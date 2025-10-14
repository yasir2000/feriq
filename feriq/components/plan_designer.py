"""
Plan Designer Component

This module implements the plan generation system that converts goals into executable plans
with task orchestration, resource allocation, and timeline management.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
from pydantic import BaseModel, Field, validator
import networkx as nx

from ..core.goal import Goal, GoalType, GoalStatus
from ..core.task import FeriqTask, TaskStatus, TaskPriority, TaskComplexity
from ..core.plan import Plan, PlanStatus, Milestone
from ..core.role import Role
from ..utils.logger import FeriqLogger
from ..utils.config import Config


class PlanningStrategy(str, Enum):
    """Planning strategies for generating execution plans."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    CRITICAL_PATH = "critical_path"
    RESOURCE_OPTIMAL = "resource_optimal"
    TIME_OPTIMAL = "time_optimal"


class PlanTemplate(BaseModel):
    """Template for generating plans from goals."""
    name: str
    description: str
    strategy: PlanningStrategy
    goal_types: List[GoalType]
    task_patterns: List[Dict[str, Any]]
    resource_requirements: Dict[str, Any]
    estimated_duration: Optional[timedelta] = None
    complexity_threshold: float = 0.5
    parallel_execution_factor: float = 0.7
    
    class Config:
        use_enum_values = True


@dataclass
class ResourceConstraint:
    """Represents resource constraints for plan execution."""
    resource_type: str
    max_capacity: int
    current_usage: int = 0
    priority: int = 1
    
    @property
    def available_capacity(self) -> int:
        return max(0, self.max_capacity - self.current_usage)
    
    @property
    def utilization_rate(self) -> float:
        return self.current_usage / self.max_capacity if self.max_capacity > 0 else 0.0


@dataclass
class TimelineNode:
    """Represents a node in the execution timeline."""
    task_id: str
    start_time: datetime
    end_time: datetime
    dependencies: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, int] = field(default_factory=dict)


class PlanOptimizer:
    """Optimizes execution plans for different criteria."""
    
    def __init__(self, logger: FeriqLogger):
        self.logger = logger
    
    def optimize_for_time(self, plan: Plan, resources: Dict[str, ResourceConstraint]) -> Plan:
        """Optimize plan for minimum execution time."""
        self.logger.info("Optimizing plan for minimum execution time", plan_id=plan.plan_id)
        
        # Create dependency graph
        graph = nx.DiGraph()
        task_map = {task.task_id: task for task in plan.tasks}
        
        for task in plan.tasks:
            graph.add_node(task.task_id, task=task)
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    graph.add_edge(dep_id, task.task_id)
        
        # Calculate critical path
        critical_path = self._calculate_critical_path(graph, task_map)
        
        # Optimize non-critical tasks for parallel execution
        optimized_tasks = []
        for task in plan.tasks:
            if task.task_id in critical_path:
                task.priority = TaskPriority.HIGH
            else:
                # Check if task can be parallelized
                if self._can_parallelize(task, resources):
                    task.priority = TaskPriority.MEDIUM
            optimized_tasks.append(task)
        
        plan.tasks = optimized_tasks
        return plan
    
    def optimize_for_resources(self, plan: Plan, resources: Dict[str, ResourceConstraint]) -> Plan:
        """Optimize plan for resource efficiency."""
        self.logger.info("Optimizing plan for resource efficiency", plan_id=plan.plan_id)
        
        # Sort tasks by resource efficiency score
        def resource_efficiency_score(task: FeriqTask) -> float:
            base_score = 1.0 / (task.complexity_score + 0.1)
            resource_penalty = sum(
                req_amount / resources.get(req_type, ResourceConstraint(req_type, 1)).max_capacity
                for req_type, req_amount in task.resource_requirements.items()
            )
            return base_score - resource_penalty
        
        plan.tasks.sort(key=resource_efficiency_score, reverse=True)
        return plan
    
    def _calculate_critical_path(self, graph: nx.DiGraph, task_map: Dict[str, FeriqTask]) -> List[str]:
        """Calculate critical path in the task dependency graph."""
        try:
            # Calculate longest path (critical path)
            critical_path = nx.dag_longest_path(graph, weight='duration')
            return critical_path
        except nx.NetworkXError:
            self.logger.warning("Could not calculate critical path, graph may have cycles")
            return []
    
    def _can_parallelize(self, task: FeriqTask, resources: Dict[str, ResourceConstraint]) -> bool:
        """Check if a task can be executed in parallel with others."""
        for req_type, req_amount in task.resource_requirements.items():
            if req_type in resources:
                if resources[req_type].available_capacity < req_amount:
                    return False
        return True


class PlanDesigner:
    """
    Plan Designer component that converts goals into executable plans.
    
    This component analyzes goals and creates detailed execution plans with:
    - Task breakdown and dependency analysis
    - Resource allocation and constraint management
    - Timeline optimization and milestone planning
    - Risk assessment and contingency planning
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = FeriqLogger("PlanDesigner", self.config)
        self.optimizer = PlanOptimizer(self.logger)
        
        # Plan templates for different goal types
        self.plan_templates: Dict[GoalType, List[PlanTemplate]] = {
            GoalType.RESEARCH: [
                PlanTemplate(
                    name="Research Investigation",
                    description="Systematic research approach",
                    strategy=PlanningStrategy.SEQUENTIAL,
                    goal_types=[GoalType.RESEARCH],
                    task_patterns=[
                        {"type": "information_gathering", "complexity": 0.3},
                        {"type": "analysis", "complexity": 0.6},
                        {"type": "synthesis", "complexity": 0.8}
                    ],
                    resource_requirements={"research_agents": 2, "analysis_tools": 1},
                    estimated_duration=timedelta(hours=4)
                )
            ],
            GoalType.DEVELOPMENT: [
                PlanTemplate(
                    name="Software Development",
                    description="Agile development approach",
                    strategy=PlanningStrategy.HYBRID,
                    goal_types=[GoalType.DEVELOPMENT],
                    task_patterns=[
                        {"type": "requirement_analysis", "complexity": 0.4},
                        {"type": "design", "complexity": 0.6},
                        {"type": "implementation", "complexity": 0.8},
                        {"type": "testing", "complexity": 0.5}
                    ],
                    resource_requirements={"developers": 3, "testers": 1},
                    estimated_duration=timedelta(days=7)
                )
            ],
            GoalType.ANALYSIS: [
                PlanTemplate(
                    name="Data Analysis",
                    description="Comprehensive data analysis workflow",
                    strategy=PlanningStrategy.SEQUENTIAL,
                    goal_types=[GoalType.ANALYSIS],
                    task_patterns=[
                        {"type": "data_collection", "complexity": 0.3},
                        {"type": "data_cleaning", "complexity": 0.4},
                        {"type": "analysis", "complexity": 0.7},
                        {"type": "reporting", "complexity": 0.5}
                    ],
                    resource_requirements={"analysts": 2, "compute_resources": 1},
                    estimated_duration=timedelta(hours=6)
                )
            ]
        }
        
        # Resource constraints
        self.resource_constraints: Dict[str, ResourceConstraint] = {}
        self._initialize_default_resources()
        
        self.logger.info("PlanDesigner initialized successfully")
    
    def _initialize_default_resources(self):
        """Initialize default resource constraints."""
        default_resources = [
            ResourceConstraint("agents", 10),
            ResourceConstraint("compute_resources", 5),
            ResourceConstraint("memory", 1000),
            ResourceConstraint("storage", 10000)
        ]
        
        for resource in default_resources:
            self.resource_constraints[resource.resource_type] = resource
    
    def design_plan_from_goal(
        self,
        goal: Goal,
        available_roles: List[Role],
        strategy: Optional[PlanningStrategy] = None,
        resource_constraints: Optional[Dict[str, ResourceConstraint]] = None
    ) -> Plan:
        """
        Create an execution plan from a goal.
        
        Args:
            goal: The goal to create a plan for
            available_roles: Available roles for task assignment
            strategy: Planning strategy to use
            resource_constraints: Resource constraints for the plan
            
        Returns:
            Generated execution plan
        """
        self.logger.info(
            "Designing plan from goal",
            goal_id=goal.goal_id,
            goal_type=goal.goal_type,
            strategy=strategy
        )
        
        try:
            # Select appropriate template
            template = self._select_template(goal, strategy)
            
            # Generate tasks from goal and template
            tasks = self._generate_tasks_from_goal(goal, template, available_roles)
            
            # Create milestones
            milestones = self._generate_milestones(goal, tasks, template)
            
            # Create initial plan
            plan = Plan(
                plan_id=str(uuid.uuid4()),
                goal_id=goal.goal_id,
                name=f"Plan for {goal.name}",
                description=f"Execution plan for goal: {goal.description}",
                tasks=tasks,
                milestones=milestones,
                created_at=datetime.now(),
                estimated_duration=template.estimated_duration or timedelta(hours=2)
            )
            
            # Apply resource constraints
            if resource_constraints:
                self.resource_constraints.update(resource_constraints)
            
            # Optimize plan based on strategy
            plan = self._optimize_plan(plan, template.strategy)
            
            # Validate plan
            validation_result = self._validate_plan(plan)
            if not validation_result["valid"]:
                self.logger.warning(
                    "Plan validation failed",
                    plan_id=plan.plan_id,
                    issues=validation_result["issues"]
                )
            
            self.logger.info(
                "Plan designed successfully",
                plan_id=plan.plan_id,
                task_count=len(tasks),
                milestone_count=len(milestones)
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to design plan from goal: {e}", goal_id=goal.goal_id)
            raise
    
    def _select_template(self, goal: Goal, strategy: Optional[PlanningStrategy]) -> PlanTemplate:
        """Select the best template for the goal."""
        templates = self.plan_templates.get(goal.goal_type, [])
        
        if not templates:
            # Create default template
            return PlanTemplate(
                name="Default Plan",
                description="Default planning template",
                strategy=strategy or PlanningStrategy.SEQUENTIAL,
                goal_types=[goal.goal_type],
                task_patterns=[{"type": "generic_task", "complexity": 0.5}],
                resource_requirements={"agents": 1}
            )
        
        # Select template based on strategy preference
        if strategy:
            for template in templates:
                if template.strategy == strategy:
                    return template
        
        # Return first available template
        return templates[0]
    
    def _generate_tasks_from_goal(
        self,
        goal: Goal,
        template: PlanTemplate,
        available_roles: List[Role]
    ) -> List[FeriqTask]:
        """Generate tasks from goal using template patterns."""
        tasks = []
        
        for i, pattern in enumerate(template.task_patterns):
            task = FeriqTask(
                task_id=str(uuid.uuid4()),
                name=f"{pattern['type'].replace('_', ' ').title()} - {goal.name}",
                description=f"Execute {pattern['type']} for goal: {goal.description}",
                goal_id=goal.goal_id,
                complexity=TaskComplexity(pattern.get('complexity', 0.5)),
                priority=TaskPriority.MEDIUM,
                estimated_duration=timedelta(hours=pattern.get('duration', 1)),
                required_capabilities=goal.required_capabilities[:3],  # Use top capabilities
                resource_requirements=template.resource_requirements.copy()
            )
            
            # Add dependencies for sequential patterns
            if i > 0 and template.strategy in [PlanningStrategy.SEQUENTIAL, PlanningStrategy.HYBRID]:
                task.dependencies.add(tasks[i-1].task_id)
            
            # Assign suitable role if available
            suitable_role = self._find_suitable_role(task, available_roles)
            if suitable_role:
                task.assigned_role_id = suitable_role.role_id
            
            tasks.append(task)
        
        return tasks
    
    def _find_suitable_role(self, task: FeriqTask, available_roles: List[Role]) -> Optional[Role]:
        """Find the most suitable role for a task."""
        best_role = None
        best_score = 0.0
        
        for role in available_roles:
            score = role.calculate_suitability_score(task)
            if score > best_score:
                best_score = score
                best_role = role
        
        return best_role if best_score > 0.5 else None
    
    def _generate_milestones(
        self,
        goal: Goal,
        tasks: List[FeriqTask],
        template: PlanTemplate
    ) -> List[Milestone]:
        """Generate milestones for the plan."""
        milestones = []
        
        # Create milestone at 25%, 50%, 75%, and 100% completion
        milestone_points = [0.25, 0.5, 0.75, 1.0]
        milestone_names = ["Quarter Complete", "Half Complete", "Three Quarters Complete", "Complete"]
        
        for i, (point, name) in enumerate(zip(milestone_points, milestone_names)):
            milestone_task_index = min(int(len(tasks) * point), len(tasks) - 1)
            
            milestone = Milestone(
                milestone_id=str(uuid.uuid4()),
                name=f"{goal.name} - {name}",
                description=f"Milestone at {int(point * 100)}% completion",
                target_date=datetime.now() + timedelta(
                    seconds=int((template.estimated_duration or timedelta(hours=2)).total_seconds() * point)
                ),
                completion_criteria=[f"Complete {milestone_task_index + 1} of {len(tasks)} tasks"],
                dependencies=[tasks[j].task_id for j in range(milestone_task_index + 1)]
            )
            
            milestones.append(milestone)
        
        return milestones
    
    def _optimize_plan(self, plan: Plan, strategy: PlanningStrategy) -> Plan:
        """Optimize plan based on the specified strategy."""
        if strategy == PlanningStrategy.TIME_OPTIMAL:
            return self.optimizer.optimize_for_time(plan, self.resource_constraints)
        elif strategy == PlanningStrategy.RESOURCE_OPTIMAL:
            return self.optimizer.optimize_for_resources(plan, self.resource_constraints)
        elif strategy == PlanningStrategy.CRITICAL_PATH:
            return self.optimizer.optimize_for_time(plan, self.resource_constraints)
        
        return plan
    
    def _validate_plan(self, plan: Plan) -> Dict[str, Any]:
        """Validate the generated plan."""
        issues = []
        
        # Check for circular dependencies
        task_map = {task.task_id: task for task in plan.tasks}
        graph = nx.DiGraph()
        
        for task in plan.tasks:
            graph.add_node(task.task_id)
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    graph.add_edge(dep_id, task.task_id)
        
        if not nx.is_directed_acyclic_graph(graph):
            issues.append("Plan contains circular dependencies")
        
        # Check resource constraints
        total_resource_usage = {}
        for task in plan.tasks:
            for resource, amount in task.resource_requirements.items():
                total_resource_usage[resource] = total_resource_usage.get(resource, 0) + amount
        
        for resource, usage in total_resource_usage.items():
            if resource in self.resource_constraints:
                if usage > self.resource_constraints[resource].max_capacity:
                    issues.append(f"Resource {resource} over capacity: {usage} > {self.resource_constraints[resource].max_capacity}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def create_contingency_plan(
        self,
        primary_plan: Plan,
        risk_factors: List[str],
        backup_resources: Optional[Dict[str, ResourceConstraint]] = None
    ) -> Plan:
        """
        Create a contingency plan for handling failures in the primary plan.
        
        Args:
            primary_plan: The primary execution plan
            risk_factors: Identified risk factors
            backup_resources: Additional resources for contingency
            
        Returns:
            Contingency plan
        """
        self.logger.info(
            "Creating contingency plan",
            primary_plan_id=primary_plan.plan_id,
            risk_count=len(risk_factors)
        )
        
        contingency_tasks = []
        
        # Create alternative tasks for high-risk primary tasks
        for task in primary_plan.tasks:
            if task.priority == TaskPriority.HIGH or task.complexity_score > 0.7:
                contingency_task = FeriqTask(
                    task_id=str(uuid.uuid4()),
                    name=f"Contingency: {task.name}",
                    description=f"Alternative approach for: {task.description}",
                    goal_id=task.goal_id,
                    complexity=TaskComplexity(min(task.complexity_score + 0.1, 1.0)),
                    priority=TaskPriority.LOW,
                    estimated_duration=task.estimated_duration * 1.2,  # 20% buffer
                    required_capabilities=task.required_capabilities,
                    resource_requirements=task.resource_requirements.copy()
                )
                contingency_tasks.append(contingency_task)
        
        contingency_plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal_id=primary_plan.goal_id,
            name=f"Contingency: {primary_plan.name}",
            description=f"Contingency plan for: {primary_plan.description}",
            tasks=contingency_tasks,
            milestones=[],
            created_at=datetime.now(),
            estimated_duration=primary_plan.estimated_duration * 1.3  # 30% buffer
        )
        
        self.logger.info(
            "Contingency plan created",
            contingency_plan_id=contingency_plan.plan_id,
            contingency_task_count=len(contingency_tasks)
        )
        
        return contingency_plan
    
    def update_resource_constraints(self, resource_updates: Dict[str, ResourceConstraint]):
        """Update resource constraints for plan generation."""
        self.resource_constraints.update(resource_updates)
        self.logger.info("Resource constraints updated", updated_resources=list(resource_updates.keys()))
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get statistics about plan generation."""
        return {
            "available_templates": sum(len(templates) for templates in self.plan_templates.values()),
            "resource_constraints": len(self.resource_constraints),
            "planning_strategies": len(PlanningStrategy),
            "resource_utilization": {
                resource_type: constraint.utilization_rate
                for resource_type, constraint in self.resource_constraints.items()
            }
        }