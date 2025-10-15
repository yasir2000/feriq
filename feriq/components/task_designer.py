"""Task Designer component for creating and decomposing tasks with team support."""

from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import uuid

from ..core.task import FeriqTask, TaskType, TaskComplexity, TaskPriority, TaskDependency
from ..core.goal import Goal, GoalType, GoalPriority
from ..core.plan import Plan


class TeamTaskAssignment(BaseModel):
    """Assignment of tasks to teams"""
    team_id: str = Field(..., description="Team ID")
    team_name: str = Field(..., description="Team name")
    assigned_tasks: List[str] = Field(default_factory=list, description="Task IDs assigned to team")
    coordination_requirements: List[str] = Field(default_factory=list, description="Inter-team coordination needs")
    estimated_effort: float = Field(default=0.0, description="Total estimated effort in hours")
    parallel_execution: bool = Field(default=True, description="Can tasks be executed in parallel")


class TaskCollaboration(BaseModel):
    """Collaboration requirements between teams for task execution"""
    collaboration_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Collaboration ID")
    participating_teams: List[str] = Field(..., description="Teams involved in collaboration")
    shared_tasks: List[str] = Field(default_factory=list, description="Tasks requiring collaboration")
    coordination_pattern: str = Field(default="sequential", description="How teams coordinate (sequential, parallel, hybrid)")
    communication_frequency: str = Field(default="daily", description="Communication frequency")
    synchronization_points: List[str] = Field(default_factory=list, description="Points where teams must synchronize")


class TaskTemplate(BaseModel):
    """Template for creating tasks."""
    name: str = Field(..., description="Template name")
    task_type: TaskType = Field(..., description="Type of task")
    complexity: TaskComplexity = Field(default=TaskComplexity.MODERATE, description="Task complexity")
    description_template: str = Field(..., description="Description template")
    required_capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
    estimated_duration: float = Field(default=1.0, description="Estimated duration in hours")
    instructions_template: str = Field(default="", description="Instructions template")
    validation_criteria: List[str] = Field(default_factory=list, description="Validation criteria")
    decomposition_pattern: Optional[str] = Field(default=None, description="How to decompose this task type")
    team_suitability: List[str] = Field(default_factory=list, description="Which team types are suitable for this task")
    collaboration_requirements: Dict[str, Any] = Field(default_factory=dict, description="Requirements for team collaboration")


class DecompositionStrategy(BaseModel):
    """Strategy for decomposing complex tasks."""
    name: str = Field(..., description="Strategy name")
    applicable_types: List[TaskType] = Field(..., description="Task types this strategy applies to")
    min_complexity: TaskComplexity = Field(default=TaskComplexity.MODERATE, description="Minimum complexity for application")
    decomposition_rules: List[str] = Field(..., description="Rules for decomposition")
    subtask_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Patterns for creating subtasks")


class TaskDesigner(BaseModel):
    """
    Task Designer component that creates, decomposes, and manages tasks
    based on goals, requirements, and intelligent analysis.
    """
    
    # Component identification
    name: str = Field(default="TaskDesigner", description="Component name")
    version: str = Field(default="1.0", description="Component version")
    
    # Templates and patterns
    task_templates: Dict[str, TaskTemplate] = Field(default_factory=dict, description="Available task templates")
    decomposition_strategies: Dict[str, DecompositionStrategy] = Field(default_factory=dict, 
                                                                       description="Decomposition strategies")
    
    # Task generation rules
    generation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Task generation rules")
    complexity_assessment_rules: Dict[str, Any] = Field(default_factory=dict, description="Complexity assessment rules")
    
    # Learning and adaptation
    task_success_patterns: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, 
                                                                  description="Successful task patterns")
    failure_analysis: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, 
                                                             description="Task failure analysis")
    
    # Framework reference
    framework: Optional[Any] = Field(default=None, description="Reference to main framework")
    
    # Configuration
    auto_decompose: bool = Field(default=True, description="Whether to auto-decompose complex tasks")
    max_decomposition_depth: int = Field(default=3, description="Maximum decomposition depth")
    complexity_threshold: TaskComplexity = Field(default=TaskComplexity.COMPLEX, 
                                                 description="Threshold for auto-decomposition")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize the task designer with default templates."""
        super().__init__(**kwargs)
        self._initialize_default_templates()
        self._initialize_decomposition_strategies()
        self._initialize_generation_rules()
    
    def _initialize_default_templates(self) -> None:
        """Initialize default task templates."""
        # Research Task Template
        research_template = TaskTemplate(
            name="Research Task",
            task_type=TaskType.RESEARCH,
            complexity=TaskComplexity.MODERATE,
            description_template="Research {topic} and provide comprehensive analysis",
            required_capabilities=["information_gathering", "data_analysis", "critical_thinking"],
            estimated_duration=4.0,
            instructions_template="1. Define research scope\n2. Gather information from reliable sources\n3. Analyze findings\n4. Prepare report",
            validation_criteria=["Information accuracy", "Source reliability", "Analysis depth", "Report completeness"],
            decomposition_pattern="sequential"
        )
        
        # Analysis Task Template
        analysis_template = TaskTemplate(
            name="Analysis Task",
            task_type=TaskType.ANALYSIS,
            complexity=TaskComplexity.MODERATE,
            description_template="Analyze {data_type} and extract {insights_type} insights",
            required_capabilities=["data_analysis", "pattern_recognition", "statistical_modeling"],
            estimated_duration=3.0,
            instructions_template="1. Prepare data\n2. Apply analytical methods\n3. Identify patterns\n4. Generate insights",
            validation_criteria=["Data quality", "Method appropriateness", "Insight validity", "Documentation"],
            decomposition_pattern="parallel"
        )
        
        # Planning Task Template
        planning_template = TaskTemplate(
            name="Planning Task",
            task_type=TaskType.PLANNING,
            complexity=TaskComplexity.MODERATE,
            description_template="Create {plan_type} plan for {objective}",
            required_capabilities=["project_management", "strategic_thinking", "resource_planning"],
            estimated_duration=2.0,
            instructions_template="1. Define objectives\n2. Assess resources\n3. Create timeline\n4. Identify risks",
            validation_criteria=["Objective clarity", "Resource feasibility", "Timeline realism", "Risk assessment"],
            decomposition_pattern="sequential"
        )
        
        # Coordination Task Template
        coordination_template = TaskTemplate(
            name="Coordination Task",
            task_type=TaskType.COORDINATION,
            complexity=TaskComplexity.SIMPLE,
            description_template="Coordinate {activity} between {stakeholders}",
            required_capabilities=["communication", "project_management", "conflict_resolution"],
            estimated_duration=1.5,
            instructions_template="1. Identify stakeholders\n2. Establish communication channels\n3. Schedule activities\n4. Monitor progress",
            validation_criteria=["Stakeholder engagement", "Communication effectiveness", "Timeline adherence"],
            decomposition_pattern="hybrid"
        )
        
        self.task_templates = {
            "research": research_template,
            "analysis": analysis_template,
            "planning": planning_template,
            "coordination": coordination_template
        }
    
    def _initialize_decomposition_strategies(self) -> None:
        """Initialize decomposition strategies."""
        # Sequential Decomposition
        sequential_strategy = DecompositionStrategy(
            name="Sequential Decomposition",
            applicable_types=[TaskType.RESEARCH, TaskType.PLANNING, TaskType.EXECUTION],
            min_complexity=TaskComplexity.MODERATE,
            decomposition_rules=[
                "Break down into chronological steps",
                "Each step depends on the previous one",
                "Maintain clear handoff points"
            ],
            subtask_patterns=[
                {"name": "Planning Phase", "type": TaskType.PLANNING, "order": 1},
                {"name": "Execution Phase", "type": TaskType.EXECUTION, "order": 2},
                {"name": "Review Phase", "type": TaskType.REVIEW, "order": 3}
            ]
        )
        
        # Parallel Decomposition
        parallel_strategy = DecompositionStrategy(
            name="Parallel Decomposition",
            applicable_types=[TaskType.ANALYSIS, TaskType.RESEARCH],
            min_complexity=TaskComplexity.MODERATE,
            decomposition_rules=[
                "Break down into independent components",
                "Components can be executed simultaneously",
                "Requires final synthesis step"
            ],
            subtask_patterns=[
                {"name": "Component A", "type": TaskType.ANALYSIS, "parallel": True},
                {"name": "Component B", "type": TaskType.ANALYSIS, "parallel": True},
                {"name": "Synthesis", "type": TaskType.ANALYSIS, "depends_on": ["Component A", "Component B"]}
            ]
        )
        
        # Hierarchical Decomposition
        hierarchical_strategy = DecompositionStrategy(
            name="Hierarchical Decomposition",
            applicable_types=[TaskType.PLANNING, TaskType.COORDINATION],
            min_complexity=TaskComplexity.COMPLEX,
            decomposition_rules=[
                "Break down by organizational levels",
                "Higher levels coordinate lower levels",
                "Clear responsibility boundaries"
            ],
            subtask_patterns=[
                {"name": "Strategic Level", "type": TaskType.PLANNING, "level": 1},
                {"name": "Tactical Level", "type": TaskType.COORDINATION, "level": 2},
                {"name": "Operational Level", "type": TaskType.EXECUTION, "level": 3}
            ]
        )
        
        self.decomposition_strategies = {
            "sequential": sequential_strategy,
            "parallel": parallel_strategy,
            "hierarchical": hierarchical_strategy
        }
    
    def _initialize_generation_rules(self) -> None:
        """Initialize task generation rules."""
        self.generation_rules = [
            {
                "name": "Goal-to-Task Mapping",
                "condition": "goal_type == 'simple'",
                "action": "create_single_task",
                "parameters": {"inherit_priority": True}
            },
            {
                "name": "Complex Goal Decomposition",
                "condition": "goal_type == 'complex'",
                "action": "decompose_and_create_tasks",
                "parameters": {"max_tasks": 10, "use_templates": True}
            },
            {
                "name": "Collaborative Goal Handling",
                "condition": "goal_type == 'collaborative'",
                "action": "create_collaborative_tasks",
                "parameters": {"assign_coordination_task": True}
            }
        ]
        
        self.complexity_assessment_rules = {
            "factors": [
                {"name": "scope_breadth", "weight": 0.3},
                {"name": "technical_difficulty", "weight": 0.25},
                {"name": "resource_requirements", "weight": 0.2},
                {"name": "time_constraints", "weight": 0.15},
                {"name": "stakeholder_complexity", "weight": 0.1}
            ],
            "thresholds": {
                TaskComplexity.TRIVIAL: 0.2,
                TaskComplexity.SIMPLE: 0.4,
                TaskComplexity.MODERATE: 0.6,
                TaskComplexity.COMPLEX: 0.8,
                TaskComplexity.EXPERT: 1.0
            }
        }
    
    def create_task_from_goal(self, goal: Union[Goal, str], context: Dict[str, Any] = None) -> List[FeriqTask]:
        """Create tasks from a goal."""
        if isinstance(goal, str):
            # Create a simple goal from string
            goal_obj = Goal(
                name=f"Goal: {goal[:50]}...",
                description=goal,
                goal_type=GoalType.SIMPLE
            )
        else:
            goal_obj = goal
        
        context = context or {}
        tasks = []
        
        # Determine task creation strategy based on goal type
        if goal_obj.goal_type == GoalType.SIMPLE:
            task = self._create_single_task_from_goal(goal_obj, context)
            if task:
                tasks.append(task)
        
        elif goal_obj.goal_type in [GoalType.COMPLEX, GoalType.COLLABORATIVE]:
            tasks = self._decompose_goal_into_tasks(goal_obj, context)
        
        elif goal_obj.goal_type == GoalType.SEQUENTIAL:
            tasks = self._create_sequential_tasks(goal_obj, context)
        
        elif goal_obj.goal_type == GoalType.PARALLEL:
            tasks = self._create_parallel_tasks(goal_obj, context)
        
        elif goal_obj.goal_type == GoalType.ADAPTIVE:
            tasks = self._create_adaptive_tasks(goal_obj, context)
        
        # Store tasks in framework
        if self.framework:
            for task in tasks:
                self.framework.tasks[task.id] = task
        
        return tasks
    
    def _create_single_task_from_goal(self, goal: Goal, context: Dict[str, Any]) -> Optional[FeriqTask]:
        """Create a single task from a simple goal."""
        # Determine task type based on goal description and context
        task_type = self._infer_task_type(goal.description, context)
        
        # Select appropriate template
        template = self._select_template(task_type)
        
        # Assess complexity
        complexity = self._assess_complexity(goal, context)
        
        # Create task
        task = FeriqTask(
            name=f"Task: {goal.name}",
            description=goal.description,
            goal_id=goal.id,
            task_type=task_type,
            complexity=complexity,
            priority=self._map_goal_priority_to_task_priority(goal.priority),
            required_capabilities=goal.required_capabilities.copy(),
            deadline=goal.deadline,
            expected_output=goal.expected_outcome,
            validation_criteria=[criterion.description for criterion in goal.success_criteria]
        )
        
        # Apply template if available
        if template:
            self._apply_template(task, template, context)
        
        return task
    
    def _decompose_goal_into_tasks(self, goal: Goal, context: Dict[str, Any]) -> List[FeriqTask]:
        """Decompose a complex goal into multiple tasks."""
        tasks = []
        
        # Analyze goal for decomposition opportunities
        decomposition_plan = self._create_decomposition_plan(goal, context)
        
        # Create tasks based on decomposition plan
        for task_spec in decomposition_plan:
            task = FeriqTask(
                name=task_spec["name"],
                description=task_spec["description"],
                goal_id=goal.id,
                task_type=task_spec.get("task_type", TaskType.CUSTOM),
                complexity=task_spec.get("complexity", TaskComplexity.MODERATE),
                priority=self._map_goal_priority_to_task_priority(goal.priority),
                required_capabilities=task_spec.get("required_capabilities", []),
                instructions=task_spec.get("instructions", ""),
                validation_criteria=task_spec.get("validation_criteria", [])
            )
            
            # Add dependencies
            if "depends_on" in task_spec:
                for dep_name in task_spec["depends_on"]:
                    # Find the dependency task
                    dep_task = next((t for t in tasks if t.name.endswith(dep_name)), None)
                    if dep_task:
                        task.add_dependency(dep_task.id)
            
            tasks.append(task)
        
        return tasks
    
    def _create_decomposition_plan(self, goal: Goal, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a plan for decomposing a goal into tasks."""
        plan = []
        
        # Determine decomposition strategy
        strategy_name = self._select_decomposition_strategy(goal, context)
        strategy = self.decomposition_strategies.get(strategy_name)
        
        if strategy:
            # Use strategy patterns
            for i, pattern in enumerate(strategy.subtask_patterns):
                task_spec = {
                    "name": f"{goal.name} - {pattern['name']}",
                    "description": f"{pattern['name']} for {goal.description}",
                    "task_type": pattern.get("type", TaskType.CUSTOM),
                    "complexity": self._derive_subtask_complexity(goal),
                    "required_capabilities": self._derive_subtask_capabilities(pattern, goal),
                    "validation_criteria": [f"Complete {pattern['name']} successfully"]
                }
                
                # Add dependencies if specified
                if "depends_on" in pattern:
                    task_spec["depends_on"] = pattern["depends_on"]
                
                plan.append(task_spec)
        else:
            # Fallback: create basic decomposition
            plan = self._create_basic_decomposition_plan(goal)
        
        return plan
    
    def _create_basic_decomposition_plan(self, goal: Goal) -> List[Dict[str, Any]]:
        """Create a basic decomposition plan when no strategy is available."""
        return [
            {
                "name": f"{goal.name} - Planning",
                "description": f"Plan the approach for {goal.description}",
                "task_type": TaskType.PLANNING,
                "complexity": TaskComplexity.SIMPLE,
                "required_capabilities": ["planning", "analysis"],
                "validation_criteria": ["Plan is comprehensive and feasible"]
            },
            {
                "name": f"{goal.name} - Execution",
                "description": f"Execute the planned approach for {goal.description}",
                "task_type": TaskType.EXECUTION,
                "complexity": TaskComplexity.MODERATE,
                "required_capabilities": goal.required_capabilities.copy(),
                "depends_on": ["Planning"],
                "validation_criteria": ["Execution follows the plan", "Objectives are met"]
            },
            {
                "name": f"{goal.name} - Review",
                "description": f"Review and validate completion of {goal.description}",
                "task_type": TaskType.REVIEW,
                "complexity": TaskComplexity.SIMPLE,
                "required_capabilities": ["evaluation", "quality_assurance"],
                "depends_on": ["Execution"],
                "validation_criteria": ["Review is thorough", "Quality standards are met"]
            }
        ]
    
    def _infer_task_type(self, description: str, context: Dict[str, Any]) -> TaskType:
        """Infer task type from description and context."""
        description_lower = description.lower()
        
        # Keyword-based inference
        if any(word in description_lower for word in ["research", "investigate", "study", "analyze data"]):
            return TaskType.RESEARCH
        elif any(word in description_lower for word in ["analyze", "examine", "evaluate", "assess"]):
            return TaskType.ANALYSIS
        elif any(word in description_lower for word in ["plan", "design", "strategy", "roadmap"]):
            return TaskType.PLANNING
        elif any(word in description_lower for word in ["execute", "implement", "build", "create"]):
            return TaskType.EXECUTION
        elif any(word in description_lower for word in ["review", "validate", "check", "audit"]):
            return TaskType.REVIEW
        elif any(word in description_lower for word in ["coordinate", "manage", "organize", "facilitate"]):
            return TaskType.COORDINATION
        elif any(word in description_lower for word in ["communicate", "present", "report", "notify"]):
            return TaskType.COMMUNICATION
        elif any(word in description_lower for word in ["decide", "choose", "select", "determine"]):
            return TaskType.DECISION
        
        # Context-based inference
        if "task_type_hint" in context:
            return TaskType(context["task_type_hint"])
        
        return TaskType.CUSTOM
    
    def _assess_complexity(self, goal: Goal, context: Dict[str, Any]) -> TaskComplexity:
        """Assess task complexity based on goal and context."""
        complexity_score = 0.0
        
        # Factor: Scope breadth
        scope_indicators = len(goal.required_capabilities) + len(goal.success_criteria)
        scope_score = min(1.0, scope_indicators / 10.0)
        complexity_score += scope_score * 0.3
        
        # Factor: Technical difficulty (from context or goal complexity)
        if "technical_difficulty" in context:
            tech_score = context["technical_difficulty"]
        else:
            tech_score = len(goal.required_capabilities) / 10.0
        complexity_score += min(1.0, tech_score) * 0.25
        
        # Factor: Resource requirements
        resource_score = len(goal.required_resources) / 5.0
        complexity_score += min(1.0, resource_score) * 0.2
        
        # Factor: Time constraints
        if goal.deadline:
            time_to_deadline = (goal.deadline - datetime.now()).total_seconds() / 3600
            if time_to_deadline < 24:
                time_score = 1.0  # Very urgent
            elif time_to_deadline < 168:  # 1 week
                time_score = 0.7
            else:
                time_score = 0.3
        else:
            time_score = 0.5
        complexity_score += time_score * 0.15
        
        # Factor: Stakeholder complexity
        stakeholder_count = len(goal.assigned_agent_ids)
        stakeholder_score = min(1.0, stakeholder_count / 5.0)
        complexity_score += stakeholder_score * 0.1
        
        # Map score to complexity level
        for complexity_level, threshold in self.complexity_assessment_rules["thresholds"].items():
            if complexity_score <= threshold:
                return complexity_level
        
        return TaskComplexity.EXPERT
    
    def _select_template(self, task_type: TaskType) -> Optional[TaskTemplate]:
        """Select appropriate template for task type."""
        template_mapping = {
            TaskType.RESEARCH: "research",
            TaskType.ANALYSIS: "analysis",
            TaskType.PLANNING: "planning",
            TaskType.COORDINATION: "coordination"
        }
        
        template_key = template_mapping.get(task_type)
        return self.task_templates.get(template_key) if template_key else None
    
    def _apply_template(self, task: FeriqTask, template: TaskTemplate, context: Dict[str, Any]) -> None:
        """Apply template to a task."""
        # Update instructions if not already set
        if not task.instructions and template.instructions_template:
            task.instructions = template.instructions_template
        
        # Add template capabilities
        for cap in template.required_capabilities:
            if cap not in task.required_capabilities:
                task.required_capabilities.append(cap)
        
        # Add template validation criteria
        for criterion in template.validation_criteria:
            if criterion not in task.validation_criteria:
                task.validation_criteria.append(criterion)
        
        # Set estimated duration
        if not task.estimated_duration:
            task.estimated_duration = template.estimated_duration
    
    def _select_decomposition_strategy(self, goal: Goal, context: Dict[str, Any]) -> str:
        """Select appropriate decomposition strategy."""
        # Check if goal specifies a preference
        if "decomposition_strategy" in context:
            return context["decomposition_strategy"]
        
        # Infer from goal type
        if goal.goal_type == GoalType.SEQUENTIAL:
            return "sequential"
        elif goal.goal_type == GoalType.PARALLEL:
            return "parallel"
        elif goal.goal_type == GoalType.COLLABORATIVE:
            return "hierarchical"
        
        # Default based on complexity
        complexity = self._assess_complexity(goal, context)
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            return "hierarchical"
        else:
            return "sequential"
    
    def _derive_subtask_complexity(self, goal: Goal) -> TaskComplexity:
        """Derive complexity for subtasks based on parent goal."""
        parent_complexity = goal.calculate_complexity_score()
        
        # Subtasks are typically less complex than the parent
        if parent_complexity > 2.0:
            return TaskComplexity.MODERATE
        elif parent_complexity > 1.5:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.TRIVIAL
    
    def _derive_subtask_capabilities(self, pattern: Dict[str, Any], goal: Goal) -> List[str]:
        """Derive required capabilities for a subtask based on pattern and goal."""
        capabilities = []
        
        # Add pattern-specific capabilities
        task_type = pattern.get("type", TaskType.CUSTOM)
        type_capabilities = {
            TaskType.PLANNING: ["planning", "strategic_thinking"],
            TaskType.EXECUTION: ["execution", "implementation"],
            TaskType.REVIEW: ["evaluation", "quality_assurance"],
            TaskType.ANALYSIS: ["data_analysis", "critical_thinking"],
            TaskType.RESEARCH: ["information_gathering", "research"],
            TaskType.COORDINATION: ["coordination", "communication"]
        }
        
        capabilities.extend(type_capabilities.get(task_type, []))
        
        # Add relevant capabilities from parent goal
        relevant_caps = [cap for cap in goal.required_capabilities 
                        if any(keyword in cap for keyword in pattern.get("keywords", []))]
        capabilities.extend(relevant_caps)
        
        return list(set(capabilities))  # Remove duplicates
    
    def _map_goal_priority_to_task_priority(self, goal_priority: GoalPriority) -> TaskPriority:
        """Map goal priority to task priority."""
        mapping = {
            GoalPriority.LOW: TaskPriority.LOW,
            GoalPriority.MEDIUM: TaskPriority.MEDIUM,
            GoalPriority.HIGH: TaskPriority.HIGH,
            GoalPriority.CRITICAL: TaskPriority.CRITICAL
        }
        return mapping.get(goal_priority, TaskPriority.MEDIUM)
    
    def _create_sequential_tasks(self, goal: Goal, context: Dict[str, Any]) -> List[FeriqTask]:
        """Create sequential tasks for a sequential goal."""
        return self._decompose_goal_into_tasks(goal, {**context, "decomposition_strategy": "sequential"})
    
    def _create_parallel_tasks(self, goal: Goal, context: Dict[str, Any]) -> List[FeriqTask]:
        """Create parallel tasks for a parallel goal."""
        return self._decompose_goal_into_tasks(goal, {**context, "decomposition_strategy": "parallel"})
    
    def _create_adaptive_tasks(self, goal: Goal, context: Dict[str, Any]) -> List[FeriqTask]:
        """Create adaptive tasks that can change based on execution."""
        # Start with basic tasks and mark them as adaptive
        tasks = self._create_single_task_from_goal(goal, context)
        if isinstance(tasks, FeriqTask):
            tasks = [tasks]
        
        for task in tasks:
            task.metadata["adaptive"] = True
            task.metadata["adaptation_triggers"] = [
                "performance_threshold_not_met",
                "resource_availability_changed",
                "priority_changed"
            ]
        
        return tasks
    
    def decompose_task(self, task: FeriqTask, strategy: str = None) -> List[FeriqTask]:
        """Decompose a complex task into subtasks."""
        if not self.auto_decompose:
            return [task]
        
        # Check if task needs decomposition
        if task.complexity < self.complexity_threshold:
            return [task]
        
        # Select decomposition strategy
        if not strategy:
            strategy = self._select_task_decomposition_strategy(task)
        
        # Get decomposition strategy
        decomp_strategy = self.decomposition_strategies.get(strategy)
        if not decomp_strategy:
            return [task]
        
        # Create subtasks
        subtasks = []
        for i, pattern in enumerate(decomp_strategy.subtask_patterns):
            subtask = FeriqTask(
                name=f"{task.name} - {pattern['name']}",
                description=f"{pattern['name']} for {task.description}",
                task_type=pattern.get("type", task.task_type),
                complexity=self._derive_subtask_complexity_from_task(task),
                priority=task.priority,
                goal_id=task.goal_id,
                parent_task_id=task.id,
                required_capabilities=self._derive_subtask_capabilities_from_task(task, pattern),
                validation_criteria=[f"Complete {pattern['name']} successfully"]
            )
            
            # Add dependencies
            if "depends_on" in pattern:
                for dep_name in pattern["depends_on"]:
                    dep_task = next((t for t in subtasks if t.name.endswith(dep_name)), None)
                    if dep_task:
                        subtask.add_dependency(dep_task.id)
            
            subtasks.append(subtask)
            
            # Update parent task
            task.add_subtask(subtask.id)
        
        # Store subtasks in framework
        if self.framework:
            for subtask in subtasks:
                self.framework.tasks[subtask.id] = subtask
        
        return subtasks
    
    def _select_task_decomposition_strategy(self, task: FeriqTask) -> str:
        """Select decomposition strategy for a task."""
        # Find applicable strategies
        applicable = []
        for name, strategy in self.decomposition_strategies.items():
            if (task.task_type in strategy.applicable_types and 
                task.complexity >= strategy.min_complexity):
                applicable.append(name)
        
        # Return first applicable or default
        return applicable[0] if applicable else "sequential"
    
    def _derive_subtask_complexity_from_task(self, task: FeriqTask) -> TaskComplexity:
        """Derive subtask complexity from parent task."""
        complexity_map = {
            TaskComplexity.EXPERT: TaskComplexity.COMPLEX,
            TaskComplexity.COMPLEX: TaskComplexity.MODERATE,
            TaskComplexity.MODERATE: TaskComplexity.SIMPLE,
            TaskComplexity.SIMPLE: TaskComplexity.TRIVIAL,
            TaskComplexity.TRIVIAL: TaskComplexity.TRIVIAL
        }
        return complexity_map.get(task.complexity, TaskComplexity.SIMPLE)
    
    def _derive_subtask_capabilities_from_task(self, task: FeriqTask, pattern: Dict[str, Any]) -> List[str]:
        """Derive capabilities for subtask from parent task and pattern."""
        capabilities = []
        
        # Add pattern-specific capabilities
        task_type = pattern.get("type", task.task_type)
        type_capabilities = {
            TaskType.PLANNING: ["planning", "strategic_thinking"],
            TaskType.EXECUTION: ["execution", "implementation"],
            TaskType.REVIEW: ["evaluation", "quality_assurance"]
        }
        capabilities.extend(type_capabilities.get(task_type, []))
        
        # Add subset of parent capabilities
        capabilities.extend(task.required_capabilities[:3])  # Take first 3
        
        return list(set(capabilities))
    
    def get_task_recommendations(self, goal_id: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get task recommendations for a goal."""
        if not self.framework:
            return []
        
        goal = self.framework.goals.get(goal_id)
        if not goal:
            return []
        
        context = context or {}
        
        # Generate task options
        task_options = []
        
        # Option 1: Single comprehensive task
        single_task = self._create_single_task_from_goal(goal, context)
        if single_task:
            task_options.append({
                "approach": "single_task",
                "tasks": [single_task],
                "pros": ["Simple coordination", "Single point of responsibility"],
                "cons": ["High complexity", "Potential bottleneck"],
                "estimated_duration": single_task.estimated_duration or 4.0
            })
        
        # Option 2: Decomposed tasks
        decomposed_tasks = self._decompose_goal_into_tasks(goal, context)
        if len(decomposed_tasks) > 1:
            total_duration = sum(task.estimated_duration or 2.0 for task in decomposed_tasks)
            task_options.append({
                "approach": "decomposed",
                "tasks": decomposed_tasks,
                "pros": ["Parallel execution possible", "Lower individual complexity", "Better resource allocation"],
                "cons": ["Requires coordination", "Multiple dependencies"],
                "estimated_duration": total_duration * 0.7  # Assume some parallelization
            })
        
        # Option 3: Template-based approach
        for template_name, template in self.task_templates.items():
            template_task = self._create_task_from_template(template, goal, context)
            if template_task:
                task_options.append({
                    "approach": f"template_{template_name}",
                    "tasks": [template_task],
                    "pros": ["Proven approach", "Clear structure", "Predictable outcome"],
                    "cons": ["May not fit perfectly", "Less flexibility"],
                    "estimated_duration": template.estimated_duration
                })
        
        return task_options
    
    def _create_task_from_template(self, template: TaskTemplate, goal: Goal, context: Dict[str, Any]) -> FeriqTask:
        """Create a task from a template for a specific goal."""
        task = FeriqTask(
            name=goal.name,
            description=template.description_template.format(
                topic=goal.name,
                objective=goal.description
            ),
            goal_id=goal.id,
            task_type=template.task_type,
            complexity=template.complexity,
            priority=self._map_goal_priority_to_task_priority(goal.priority),
            required_capabilities=template.required_capabilities.copy(),
            instructions=template.instructions_template,
            validation_criteria=template.validation_criteria.copy(),
            estimated_duration=template.estimated_duration
        )
        
        return task
    
    # Team-specific methods for collaborative task management
    
    def create_tasks_for_teams(self, 
                              goal: Union[Goal, str], 
                              teams: List[Dict[str, Any]], 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create and assign tasks to multiple teams for collaborative execution.
        
        Args:
            goal: Goal to accomplish
            teams: List of team information (id, name, capabilities, discipline)
            context: Additional context for task creation
        
        Returns:
            Dictionary containing team assignments, collaborations, and execution plan
        """
        context = context or {}
        
        # Create tasks from goal
        tasks = self.create_task_from_goal(goal, context)
        
        # Analyze tasks for team suitability
        team_assignments = self._assign_tasks_to_teams(tasks, teams)
        
        # Identify collaboration requirements
        collaborations = self._identify_task_collaborations(team_assignments, tasks)
        
        # Create execution coordination plan
        coordination_plan = self._create_team_coordination_plan(team_assignments, collaborations)
        
        return {
            "goal": goal.name if hasattr(goal, 'name') else str(goal),
            "tasks": [task.dict() for task in tasks],
            "team_assignments": [assignment.dict() for assignment in team_assignments],
            "collaborations": [collab.dict() for collab in collaborations],
            "coordination_plan": coordination_plan,
            "estimated_completion_time": coordination_plan.get("total_time", 0),
            "parallel_execution_opportunities": coordination_plan.get("parallel_tasks", []),
            "critical_path": coordination_plan.get("critical_path", [])
        }
    
    def create_cross_functional_tasks(self, 
                                     problem_description: str, 
                                     teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create tasks that require cross-functional team collaboration.
        
        Args:
            problem_description: Description of the problem to solve
            teams: Available teams with their capabilities
        
        Returns:
            Cross-functional task breakdown with collaboration requirements
        """
        
        # Analyze problem for cross-functional requirements
        cross_functional_needs = self._analyze_cross_functional_requirements(problem_description, teams)
        
        # Create tasks that leverage multiple team disciplines
        cross_functional_tasks = []
        
        for need in cross_functional_needs:
            task_data = self._create_cross_functional_task(need, teams)
            
            task = FeriqTask(
                name=task_data["name"],
                description=task_data["description"],
                task_type=TaskType.COORDINATION,
                complexity=TaskComplexity.COMPLEX,
                required_capabilities=task_data["required_capabilities"],
                estimated_duration=task_data["estimated_duration"],
                collaboration_requirements=task_data["collaboration_requirements"]
            )
            
            cross_functional_tasks.append(task)
        
        # Assign teams to tasks
        team_assignments = self._assign_cross_functional_teams(cross_functional_tasks, teams)
        
        # Create collaboration framework
        collaboration_framework = self._create_collaboration_framework(team_assignments)
        
        return {
            "problem": problem_description,
            "cross_functional_tasks": [task.dict() for task in cross_functional_tasks],
            "team_assignments": team_assignments,
            "collaboration_framework": collaboration_framework,
            "success_metrics": self._define_cross_functional_success_metrics(cross_functional_tasks),
            "communication_plan": self._create_team_communication_plan(team_assignments)
        }
    
    def optimize_task_distribution(self, 
                                  tasks: List[FeriqTask], 
                                  teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize task distribution across teams for maximum efficiency and collaboration.
        
        Args:
            tasks: List of tasks to distribute
            teams: Available teams
        
        Returns:
            Optimized task distribution plan
        """
        
        # Calculate team capabilities and workload capacity
        team_analysis = self._analyze_team_capabilities(teams)
        
        # Optimize task assignments using multiple criteria
        optimization_results = self._optimize_task_assignments(tasks, team_analysis)
        
        # Create load balancing recommendations
        load_balancing = self._create_load_balancing_plan(optimization_results)
        
        # Identify bottlenecks and mitigation strategies
        bottleneck_analysis = self._identify_task_bottlenecks(optimization_results)
        
        return {
            "optimized_assignments": optimization_results["assignments"],
            "efficiency_metrics": optimization_results["metrics"],
            "load_balancing": load_balancing,
            "bottleneck_analysis": bottleneck_analysis,
            "recommendations": optimization_results["recommendations"],
            "alternative_scenarios": optimization_results["alternatives"]
        }
    
    def create_autonomous_task_workflow(self, 
                                       teams: List[Dict[str, Any]], 
                                       objectives: List[str]) -> Dict[str, Any]:
        """
        Create autonomous task workflows where teams can self-organize and adapt.
        
        Args:
            teams: Teams that will participate in autonomous workflow
            objectives: High-level objectives to achieve
        
        Returns:
            Autonomous workflow structure with self-organization capabilities
        """
        
        # Create adaptive task framework
        adaptive_framework = self._create_adaptive_task_framework(teams, objectives)
        
        # Define autonomous decision-making rules
        autonomy_rules = self._define_team_autonomy_rules(teams)
        
        # Create goal extraction mechanisms
        goal_extraction = self._create_autonomous_goal_extraction(teams, objectives)
        
        # Set up inter-team coordination protocols
        coordination_protocols = self._create_autonomous_coordination_protocols(teams)
        
        return {
            "adaptive_framework": adaptive_framework,
            "autonomy_rules": autonomy_rules,
            "goal_extraction_mechanisms": goal_extraction,
            "coordination_protocols": coordination_protocols,
            "self_organization_guidelines": self._create_self_organization_guidelines(teams),
            "adaptation_triggers": self._define_adaptation_triggers(objectives),
            "performance_monitoring": self._create_autonomous_performance_monitoring(teams)
        }
    
    # Private helper methods for team functionality
    
    def _assign_tasks_to_teams(self, tasks: List[FeriqTask], teams: List[Dict[str, Any]]) -> List[TeamTaskAssignment]:
        """Assign tasks to teams based on capabilities and workload"""
        assignments = []
        
        for team in teams:
            team_assignment = TeamTaskAssignment(
                team_id=team["id"],
                team_name=team["name"]
            )
            
            # Match tasks to team capabilities
            for task in tasks:
                capability_match = self._calculate_team_task_match(team, task)
                
                if capability_match > 0.6:  # Good match threshold
                    team_assignment.assigned_tasks.append(task.id)
                    team_assignment.estimated_effort += task.estimated_duration
                    
                    # Check if task requires coordination with other teams
                    if task.complexity == TaskComplexity.COMPLEX:
                        team_assignment.coordination_requirements.append(
                            f"Coordinate with other teams for task: {task.name}"
                        )
            
            # Determine if tasks can be executed in parallel
            team_assignment.parallel_execution = self._can_execute_in_parallel(
                [task for task in tasks if task.id in team_assignment.assigned_tasks]
            )
            
            assignments.append(team_assignment)
        
        return assignments
    
    def _identify_task_collaborations(self, 
                                    team_assignments: List[TeamTaskAssignment], 
                                    tasks: List[FeriqTask]) -> List[TaskCollaboration]:
        """Identify where teams need to collaborate on tasks"""
        collaborations = []
        
        # Find tasks assigned to multiple teams
        task_team_mapping = {}
        for assignment in team_assignments:
            for task_id in assignment.assigned_tasks:
                if task_id not in task_team_mapping:
                    task_team_mapping[task_id] = []
                task_team_mapping[task_id].append(assignment.team_id)
        
        # Create collaborations for multi-team tasks
        for task_id, team_ids in task_team_mapping.items():
            if len(team_ids) > 1:
                task = next((t for t in tasks if t.id == task_id), None)
                if task:
                    collaboration = TaskCollaboration(
                        participating_teams=team_ids,
                        shared_tasks=[task_id],
                        coordination_pattern=self._determine_coordination_pattern(task),
                        synchronization_points=self._identify_sync_points(task)
                    )
                    collaborations.append(collaboration)
        
        return collaborations
    
    def _create_team_coordination_plan(self, 
                                     team_assignments: List[TeamTaskAssignment], 
                                     collaborations: List[TaskCollaboration]) -> Dict[str, Any]:
        """Create coordination plan for team task execution"""
        
        # Calculate total effort and timeline
        total_effort = sum(assignment.estimated_effort for assignment in team_assignments)
        max_team_effort = max((assignment.estimated_effort for assignment in team_assignments), default=0)
        
        # Identify parallel execution opportunities
        parallel_tasks = []
        for assignment in team_assignments:
            if assignment.parallel_execution and len(assignment.assigned_tasks) > 1:
                parallel_tasks.extend(assignment.assigned_tasks)
        
        # Create critical path
        critical_path = self._calculate_critical_path(team_assignments, collaborations)
        
        # Estimate completion time
        completion_time = max_team_effort if any(a.parallel_execution for a in team_assignments) else total_effort
        
        return {
            "total_effort_hours": total_effort,
            "estimated_completion_time": completion_time,
            "parallel_tasks": parallel_tasks,
            "critical_path": critical_path,
            "coordination_meetings": len(collaborations) * 2,  # Pre and post coordination
            "team_dependencies": self._identify_team_dependencies(team_assignments, collaborations),
            "risk_factors": self._identify_coordination_risks(team_assignments, collaborations)
        }
    
    def _analyze_cross_functional_requirements(self, 
                                             problem_description: str, 
                                             teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze what cross-functional capabilities are needed"""
        requirements = []
        
        # Extract keywords that indicate cross-functional needs
        cross_functional_indicators = {
            "integration": ["software_development", "design", "operations"],
            "user_experience": ["design", "psychology", "data_science"],
            "data_pipeline": ["data_science", "software_development", "operations"],
            "market_analysis": ["marketing", "data_science", "research"],
            "product_development": ["design", "software_development", "marketing", "research"]
        }
        
        problem_lower = problem_description.lower()
        
        for indicator, required_disciplines in cross_functional_indicators.items():
            if indicator in problem_lower:
                available_teams = [team for team in teams 
                                 if team.get("discipline", "").lower() in [d.lower() for d in required_disciplines]]
                
                if len(available_teams) >= 2:  # Need at least 2 teams for cross-functional work
                    requirements.append({
                        "type": indicator,
                        "required_disciplines": required_disciplines,
                        "available_teams": available_teams,
                        "complexity": len(required_disciplines) * 0.2,
                        "collaboration_intensity": "high" if len(required_disciplines) > 3 else "medium"
                    })
        
        return requirements
    
    def _calculate_team_task_match(self, team: Dict[str, Any], task: FeriqTask) -> float:
        """Calculate how well a team matches a task's requirements"""
        team_capabilities = set(team.get("capabilities", []))
        task_requirements = set(task.required_capabilities)
        
        if not task_requirements:
            return 0.5  # Neutral match if no specific requirements
        
        overlap = team_capabilities & task_requirements
        match_score = len(overlap) / len(task_requirements)
        
        # Bonus for discipline alignment
        team_discipline = team.get("discipline", "").lower()
        task_description = task.description.lower()
        
        if team_discipline and team_discipline in task_description:
            match_score += 0.2
        
        return min(1.0, match_score)
    
    def _can_execute_in_parallel(self, tasks: List[FeriqTask]) -> bool:
        """Determine if tasks can be executed in parallel"""
        # Simple heuristic: tasks can be parallel if they don't have strong dependencies
        # and are not all of the same critical type
        
        if len(tasks) <= 1:
            return False
        
        # Check for explicit dependencies
        for task in tasks:
            if hasattr(task, 'dependencies') and task.dependencies:
                return False
        
        # Check task types - some types work better in parallel
        parallel_friendly_types = [TaskType.RESEARCH, TaskType.ANALYSIS, TaskType.PLANNING]
        parallel_count = sum(1 for task in tasks if task.task_type in parallel_friendly_types)
        
        return parallel_count >= len(tasks) * 0.6  # 60% of tasks should be parallel-friendly
    
    def _determine_coordination_pattern(self, task: FeriqTask) -> str:
        """Determine how teams should coordinate for a specific task"""
        if task.task_type == TaskType.PLANNING:
            return "sequential"  # Planning usually needs sequential input
        elif task.task_type == TaskType.RESEARCH:
            return "parallel"   # Research can often be done in parallel
        elif task.complexity == TaskComplexity.COMPLEX:
            return "hybrid"     # Complex tasks may need both patterns
        else:
            return "parallel"
    
    def _identify_sync_points(self, task: FeriqTask) -> List[str]:
        """Identify synchronization points for collaborative tasks"""
        sync_points = ["task_start", "task_completion"]
        
        if task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            sync_points.extend([
                "requirements_review",
                "intermediate_review",
                "quality_check"
            ])
        
        if task.task_type == TaskType.COORDINATION:
            sync_points.append("stakeholder_alignment")
        
        return sync_points
    
    def _calculate_critical_path(self, 
                               team_assignments: List[TeamTaskAssignment], 
                               collaborations: List[TaskCollaboration]) -> List[str]:
        """Calculate the critical path for team task execution"""
        # Simplified critical path calculation
        critical_path = []
        
        # Find the team with the highest workload
        max_effort_assignment = max(team_assignments, key=lambda x: x.estimated_effort, default=None)
        
        if max_effort_assignment:
            critical_path.extend(max_effort_assignment.assigned_tasks)
        
        # Add collaboration dependencies
        for collaboration in collaborations:
            if collaboration.coordination_pattern == "sequential":
                critical_path.extend(collaboration.shared_tasks)
        
        return critical_path
    
    def _identify_team_dependencies(self, 
                                  team_assignments: List[TeamTaskAssignment], 
                                  collaborations: List[TaskCollaboration]) -> List[Dict[str, Any]]:
        """Identify dependencies between teams"""
        dependencies = []
        
        for collaboration in collaborations:
            for i, team_id in enumerate(collaboration.participating_teams):
                for j, other_team_id in enumerate(collaboration.participating_teams):
                    if i != j:
                        dependencies.append({
                            "dependent_team": team_id,
                            "dependency_on": other_team_id,
                            "type": "collaboration",
                            "shared_tasks": collaboration.shared_tasks,
                            "coordination_pattern": collaboration.coordination_pattern
                        })
        
        return dependencies
    
    def _identify_coordination_risks(self, 
                                   team_assignments: List[TeamTaskAssignment], 
                                   collaborations: List[TaskCollaboration]) -> List[str]:
        """Identify potential risks in team coordination"""
        risks = []
        
        # Check for overloaded teams
        avg_effort = sum(a.estimated_effort for a in team_assignments) / len(team_assignments) if team_assignments else 0
        for assignment in team_assignments:
            if assignment.estimated_effort > avg_effort * 1.5:
                risks.append(f"Team {assignment.team_name} may be overloaded")
        
        # Check for complex collaborations
        if len(collaborations) > len(team_assignments):
            risks.append("High number of collaborations may create coordination overhead")
        
        # Check for sequential bottlenecks
        sequential_collabs = [c for c in collaborations if c.coordination_pattern == "sequential"]
        if len(sequential_collabs) > 2:
            risks.append("Multiple sequential collaborations may create bottlenecks")
        
        return risks
    
    def _create_cross_functional_task(self, need: Dict[str, Any], teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a cross-functional task based on identified needs"""
        required_disciplines = need["required_disciplines"]
        
        return {
            "name": f"Cross-functional {need['type']} task",
            "description": f"Collaborative task requiring {', '.join(required_disciplines)} expertise",
            "required_capabilities": [cap for team in need["available_teams"] 
                                    for cap in team.get("capabilities", [])],
            "estimated_duration": len(required_disciplines) * 10,  # 10 hours per discipline
            "collaboration_requirements": {
                "required_disciplines": required_disciplines,
                "coordination_intensity": need["collaboration_intensity"],
                "team_count": len(need["available_teams"])
            }
        }
    
    def _assign_cross_functional_teams(self, tasks: List[FeriqTask], teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign teams to cross-functional tasks"""
        assignments = []
        
        for task in tasks:
            if hasattr(task, 'collaboration_requirements'):
                required_disciplines = task.collaboration_requirements.get("required_disciplines", [])
                
                # Find teams that match required disciplines
                matching_teams = []
                for discipline in required_disciplines:
                    for team in teams:
                        if team.get("discipline", "").lower() == discipline.lower():
                            matching_teams.append(team)
                            break
                
                assignments.append({
                    "task_id": task.id,
                    "task_name": task.name,
                    "assigned_teams": matching_teams,
                    "coordination_requirements": task.collaboration_requirements
                })
        
        return assignments
    
    def _create_collaboration_framework(self, team_assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create framework for cross-functional collaboration"""
        return {
            "communication_protocols": [
                "Daily stand-ups between team leads",
                "Weekly cross-functional reviews",
                "Shared documentation workspace",
                "Real-time collaboration tools"
            ],
            "decision_making_process": {
                "individual_team_decisions": "autonomous within team scope",
                "cross_team_decisions": "consensus between team leads",
                "escalation_path": "project coordinator or senior stakeholder"
            },
            "knowledge_sharing": {
                "documentation_standards": "shared templates and formats",
                "knowledge_transfer_sessions": "bi-weekly cross-team sessions",
                "expertise_sharing": "peer mentoring and skill exchange"
            },
            "conflict_resolution": {
                "level_1": "direct team-to-team discussion",
                "level_2": "facilitated mediation",
                "level_3": "senior management escalation"
            }
        }
    
    def _define_cross_functional_success_metrics(self, tasks: List[FeriqTask]) -> Dict[str, Any]:
        """Define success metrics for cross-functional work"""
        return {
            "collaboration_metrics": {
                "communication_frequency": "number of inter-team communications per week",
                "knowledge_sharing_events": "number of cross-team knowledge sessions",
                "conflict_resolution_time": "average time to resolve cross-team conflicts"
            },
            "delivery_metrics": {
                "task_completion_rate": "percentage of tasks completed on time",
                "quality_metrics": "defect rate and rework percentage",
                "stakeholder_satisfaction": "feedback from stakeholders on collaboration"
            },
            "innovation_metrics": {
                "cross_pollination_ideas": "number of ideas generated through collaboration",
                "process_improvements": "improvements suggested through cross-team work",
                "capability_development": "new skills developed through collaboration"
            }
        }
    
    def _create_team_communication_plan(self, team_assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create communication plan for teams"""
        return {
            "regular_meetings": {
                "frequency": "daily",
                "duration": "15 minutes",
                "participants": "team leads + rotating team members",
                "agenda": ["progress updates", "blockers", "coordination needs"]
            },
            "milestone_reviews": {
                "frequency": "weekly",
                "duration": "60 minutes",
                "participants": "all team members",
                "agenda": ["milestone progress", "quality review", "next week planning"]
            },
            "communication_channels": {
                "synchronous": ["video calls", "in-person meetings"],
                "asynchronous": ["shared documentation", "project management tools", "chat platforms"],
                "escalation": ["direct manager contact", "project coordinator", "senior stakeholder"]
            },
            "documentation_requirements": {
                "meeting_notes": "documented and shared within 24 hours",
                "decision_log": "all cross-team decisions recorded",
                "progress_reports": "weekly progress updates to all stakeholders"
            }
        }
    
    def _analyze_team_capabilities(self, teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze team capabilities for optimization"""
        analysis = {
            "team_profiles": [],
            "capability_matrix": {},
            "capacity_analysis": {},
            "specialization_areas": {}
        }
        
        for team in teams:
            # Team profile
            profile = {
                "team_id": team["id"],
                "name": team["name"],
                "discipline": team.get("discipline", "general"),
                "capabilities": team.get("capabilities", []),
                "capacity": team.get("capacity", 40),  # Default 40 hours/week
                "efficiency_rating": team.get("efficiency", 0.8)
            }
            analysis["team_profiles"].append(profile)
            
            # Capability matrix
            for capability in team.get("capabilities", []):
                if capability not in analysis["capability_matrix"]:
                    analysis["capability_matrix"][capability] = []
                analysis["capability_matrix"][capability].append(team["id"])
            
            # Capacity analysis
            analysis["capacity_analysis"][team["id"]] = {
                "total_capacity": profile["capacity"],
                "adjusted_capacity": profile["capacity"] * profile["efficiency_rating"],
                "specialization": profile["discipline"]
            }
        
        return analysis
    
    def _optimize_task_assignments(self, tasks: List[FeriqTask], team_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize task assignments using multiple criteria"""
        assignments = {}
        metrics = {}
        recommendations = []
        alternatives = []
        
        # Simple optimization algorithm
        for task in tasks:
            best_team = None
            best_score = 0.0
            
            for team_profile in team_analysis["team_profiles"]:
                score = self._calculate_assignment_score(task, team_profile, team_analysis)
                
                if score > best_score:
                    best_score = score
                    best_team = team_profile["team_id"]
            
            if best_team:
                if best_team not in assignments:
                    assignments[best_team] = []
                assignments[best_team].append({
                    "task_id": task.id,
                    "task_name": task.name,
                    "estimated_effort": task.estimated_duration,
                    "assignment_score": best_score
                })
        
        # Calculate metrics
        total_effort = sum(task.estimated_duration for task in tasks)
        team_workloads = {team_id: sum(task["estimated_effort"] for task in team_tasks) 
                         for team_id, team_tasks in assignments.items()}
        
        metrics = {
            "total_effort": total_effort,
            "team_workloads": team_workloads,
            "load_balance_score": self._calculate_load_balance_score(team_workloads, team_analysis),
            "capability_utilization": self._calculate_capability_utilization(assignments, tasks, team_analysis)
        }
        
        return {
            "assignments": assignments,
            "metrics": metrics,
            "recommendations": recommendations,
            "alternatives": alternatives
        }
    
    def _calculate_assignment_score(self, task: FeriqTask, team_profile: Dict[str, Any], team_analysis: Dict[str, Any]) -> float:
        """Calculate assignment score for task-team combination"""
        score = 0.0
        
        # Capability match (40% weight)
        team_capabilities = set(team_profile["capabilities"])
        task_requirements = set(task.required_capabilities)
        if task_requirements:
            capability_match = len(team_capabilities & task_requirements) / len(task_requirements)
            score += capability_match * 0.4
        
        # Efficiency factor (20% weight)
        score += team_profile["efficiency_rating"] * 0.2
        
        # Workload balance (20% weight)
        current_workload = team_analysis["capacity_analysis"][team_profile["team_id"]]["adjusted_capacity"]
        if current_workload > 0:
            workload_factor = min(1.0, (current_workload - task.estimated_duration) / current_workload)
            score += workload_factor * 0.2
        
        # Discipline alignment (20% weight)
        if team_profile["discipline"].lower() in task.description.lower():
            score += 0.2
        
        return score
    
    def _calculate_load_balance_score(self, team_workloads: Dict[str, float], team_analysis: Dict[str, Any]) -> float:
        """Calculate load balance score"""
        if not team_workloads:
            return 1.0
        
        workload_values = list(team_workloads.values())
        avg_workload = sum(workload_values) / len(workload_values)
        
        if avg_workload == 0:
            return 1.0
        
        variance = sum((workload - avg_workload) ** 2 for workload in workload_values) / len(workload_values)
        coefficient_of_variation = (variance ** 0.5) / avg_workload
        
        # Lower coefficient of variation = better balance
        return max(0.0, 1.0 - coefficient_of_variation)
    
    def _calculate_capability_utilization(self, assignments: Dict[str, List], tasks: List[FeriqTask], team_analysis: Dict[str, Any]) -> float:
        """Calculate how well team capabilities are being utilized"""
        total_capabilities = 0
        utilized_capabilities = 0
        
        for team_id, team_tasks in assignments.items():
            team_profile = next((tp for tp in team_analysis["team_profiles"] if tp["team_id"] == team_id), None)
            if team_profile:
                team_capabilities = set(team_profile["capabilities"])
                total_capabilities += len(team_capabilities)
                
                # Check which capabilities are used
                used_capabilities = set()
                for task_info in team_tasks:
                    task = next((t for t in tasks if t.id == task_info["task_id"]), None)
                    if task:
                        used_capabilities.update(task.required_capabilities)
                
                utilized_capabilities += len(team_capabilities & used_capabilities)
        
        return utilized_capabilities / total_capabilities if total_capabilities > 0 else 0.0
    
    def _create_load_balancing_plan(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create load balancing recommendations"""
        assignments = optimization_results["assignments"]
        metrics = optimization_results["metrics"]
        
        # Identify overloaded and underloaded teams
        team_workloads = metrics["team_workloads"]
        avg_workload = sum(team_workloads.values()) / len(team_workloads) if team_workloads else 0
        
        overloaded = {team_id: workload for team_id, workload in team_workloads.items() 
                     if workload > avg_workload * 1.2}
        underloaded = {team_id: workload for team_id, workload in team_workloads.items() 
                      if workload < avg_workload * 0.8}
        
        recommendations = []
        
        # Suggest task redistribution
        for overloaded_team, workload in overloaded.items():
            for underloaded_team, under_workload in underloaded.items():
                if len(assignments.get(overloaded_team, [])) > 1:
                    recommendations.append({
                        "action": "redistribute_task",
                        "from_team": overloaded_team,
                        "to_team": underloaded_team,
                        "reason": f"Balance workload (from {workload:.1f}h to {under_workload:.1f}h)"
                    })
        
        return {
            "overloaded_teams": overloaded,
            "underloaded_teams": underloaded,
            "redistribution_recommendations": recommendations,
            "load_balance_score": metrics.get("load_balance_score", 0.0)
        }
    
    def _identify_task_bottlenecks(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential bottlenecks in task execution"""
        assignments = optimization_results["assignments"]
        
        bottlenecks = []
        
        # Check for teams with too many tasks
        for team_id, team_tasks in assignments.items():
            if len(team_tasks) > 5:  # Arbitrary threshold
                bottlenecks.append({
                    "type": "task_overload",
                    "team_id": team_id,
                    "task_count": len(team_tasks),
                    "mitigation": "Consider task decomposition or team expansion"
                })
        
        # Check for capability bottlenecks
        capability_demand = {}
        for team_tasks in assignments.values():
            for task_info in team_tasks:
                # Would need to access task details for capabilities
                pass  # Simplified for now
        
        return {
            "identified_bottlenecks": bottlenecks,
            "mitigation_strategies": [
                "Task decomposition for overloaded teams",
                "Cross-training team members",
                "Temporary team member reallocation",
                "External resource acquisition"
            ]
        }
    
    def _create_adaptive_task_framework(self, teams: List[Dict[str, Any]], objectives: List[str]) -> Dict[str, Any]:
        """Create framework for adaptive task management"""
        return {
            "adaptation_mechanisms": {
                "goal_refinement": "Teams can refine objectives based on new information",
                "task_reallocation": "Teams can redistribute tasks based on capacity and expertise",
                "priority_adjustment": "Teams can adjust task priorities based on changing conditions",
                "resource_reallocation": "Teams can share resources as needed"
            },
            "decision_authority": {
                "individual_tasks": "full autonomy within team scope",
                "inter_team_coordination": "collaborative decision making",
                "objective_changes": "requires consensus across affected teams",
                "resource_allocation": "team lead authority with transparency requirements"
            },
            "feedback_loops": {
                "task_completion_feedback": "immediate feedback on task outcomes",
                "objective_progress_feedback": "weekly objective progress reviews",
                "team_performance_feedback": "bi-weekly team performance assessments",
                "adaptation_effectiveness": "monthly adaptation strategy reviews"
            }
        }
    
    def _define_team_autonomy_rules(self, teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Define rules for team autonomous operation"""
        return {
            "decision_boundaries": {
                "within_scope": [
                    "task sequencing and scheduling",
                    "internal resource allocation",
                    "work methodology selection",
                    "quality standards implementation"
                ],
                "requires_coordination": [
                    "changes affecting other teams",
                    "resource requests from other teams",
                    "objective modifications",
                    "timeline adjustments affecting dependencies"
                ],
                "requires_approval": [
                    "major objective changes",
                    "significant resource reallocation",
                    "timeline extensions beyond threshold",
                    "quality standard modifications"
                ]
            },
            "communication_requirements": {
                "status_updates": "daily to coordination system",
                "decision_notifications": "immediate for decisions affecting other teams",
                "request_responses": "within 24 hours for inter-team requests",
                "escalation_timeline": "within 48 hours for unresolved conflicts"
            },
            "performance_accountability": {
                "individual_metrics": "team-defined and tracked",
                "collective_metrics": "shared across all teams",
                "review_frequency": "weekly self-assessment, monthly peer review",
                "improvement_actions": "team-driven with shared learning"
            }
        }
    
    def _create_autonomous_goal_extraction(self, teams: List[Dict[str, Any]], objectives: List[str]) -> Dict[str, Any]:
        """Create mechanisms for autonomous goal extraction and refinement"""
        return {
            "extraction_methods": {
                "objective_analysis": "teams analyze high-level objectives to extract specific goals",
                "stakeholder_consultation": "teams engage with stakeholders to clarify objectives",
                "domain_expertise": "teams apply their expertise to interpret objectives",
                "collaborative_refinement": "teams work together to refine shared goals"
            },
            "refinement_processes": {
                "iterative_refinement": "goals are refined through multiple iterations",
                "feedback_integration": "stakeholder and peer feedback integrated into goal refinement",
                "feasibility_assessment": "goals assessed for feasibility and adjusted accordingly",
                "impact_analysis": "goals evaluated for impact on other teams and objectives"
            },
            "validation_mechanisms": {
                "peer_review": "goals reviewed by other teams for consistency and feasibility",
                "stakeholder_validation": "key stakeholders validate extracted goals",
                "alignment_check": "goals checked for alignment with overall objectives",
                "resource_validation": "goals validated against available resources"
            }
        }
    
    def _create_autonomous_coordination_protocols(self, teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create protocols for autonomous inter-team coordination"""
        return {
            "coordination_triggers": {
                "resource_conflicts": "automatic coordination when teams need same resources",
                "dependency_identification": "coordination triggered when task dependencies identified",
                "objective_overlap": "coordination for overlapping or conflicting objectives",
                "capability_gaps": "coordination when teams identify capability gaps"
            },
            "coordination_mechanisms": {
                "direct_negotiation": "teams negotiate directly for resource allocation",
                "mediated_discussion": "facilitated discussions for complex conflicts",
                "consensus_building": "structured consensus building for shared decisions",
                "escalation_protocols": "clear escalation paths for unresolved issues"
            },
            "coordination_tools": {
                "shared_workspace": "collaborative workspace for coordination activities",
                "communication_channels": "dedicated channels for inter-team communication",
                "decision_tracking": "system for tracking and documenting coordination decisions",
                "resource_visibility": "transparent view of resource allocation and availability"
            }
        }
    
    def _create_self_organization_guidelines(self, teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create guidelines for team self-organization"""
        return {
            "organization_principles": {
                "purpose_alignment": "teams organize around shared purpose and objectives",
                "capability_optimization": "teams organize to optimize use of capabilities",
                "efficiency_focus": "teams organize for maximum efficiency and effectiveness",
                "adaptation_readiness": "teams maintain flexibility for rapid adaptation"
            },
            "structural_flexibility": {
                "role_fluidity": "team members can take on different roles as needed",
                "boundary_permeability": "team boundaries can be adjusted based on needs",
                "leadership_rotation": "leadership roles can rotate based on expertise and context",
                "size_adaptation": "team size can be adjusted based on workload and complexity"
            },
            "organization_mechanisms": {
                "self_assessment": "regular team self-assessment of organization effectiveness",
                "restructuring_triggers": "clear triggers that indicate need for reorganization",
                "reorganization_process": "structured process for implementing organizational changes",
                "change_validation": "mechanisms to validate effectiveness of organizational changes"
            }
        }
    
    def _define_adaptation_triggers(self, objectives: List[str]) -> Dict[str, Any]:
        """Define triggers that indicate need for adaptation"""
        return {
            "performance_triggers": {
                "velocity_decline": "significant decrease in task completion velocity",
                "quality_issues": "increase in defects or rework requirements",
                "resource_utilization": "suboptimal resource utilization patterns",
                "stakeholder_satisfaction": "decline in stakeholder satisfaction metrics"
            },
            "environmental_triggers": {
                "objective_changes": "modifications to high-level objectives",
                "resource_availability": "changes in resource availability or constraints",
                "timeline_pressures": "external timeline pressures or deadline changes",
                "technology_changes": "new technologies or tools that could improve performance"
            },
            "team_triggers": {
                "capability_evolution": "teams develop new capabilities or lose existing ones",
                "workload_imbalance": "significant imbalance in workload distribution",
                "collaboration_friction": "difficulties in inter-team collaboration",
                "motivation_changes": "changes in team motivation or engagement levels"
            },
            "trigger_thresholds": {
                "performance_variance": "20% deviation from expected performance",
                "timeline_variance": "15% deviation from planned timeline",
                "quality_variance": "10% increase in defect rates",
                "satisfaction_variance": "significant decrease in satisfaction scores"
            }
        }
    
    def _create_autonomous_performance_monitoring(self, teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create autonomous performance monitoring system"""
        return {
            "monitoring_dimensions": {
                "task_performance": {
                    "completion_velocity": "rate of task completion",
                    "quality_metrics": "defect rates and rework requirements",
                    "effort_accuracy": "accuracy of effort estimates",
                    "dependency_management": "effectiveness of managing task dependencies"
                },
                "team_performance": {
                    "collaboration_effectiveness": "quality of intra-team collaboration",
                    "capability_utilization": "how well team capabilities are utilized",
                    "adaptation_agility": "speed and effectiveness of team adaptations",
                    "innovation_rate": "rate of process and solution innovations"
                },
                "inter_team_performance": {
                    "coordination_efficiency": "effectiveness of inter-team coordination",
                    "knowledge_sharing": "rate and quality of knowledge sharing",
                    "resource_sharing": "effectiveness of resource sharing",
                    "conflict_resolution": "speed and effectiveness of conflict resolution"
                }
            },
            "monitoring_mechanisms": {
                "automated_tracking": "system automatically tracks quantitative metrics",
                "self_reporting": "teams self-report on qualitative metrics",
                "peer_feedback": "teams provide feedback on each other's performance",
                "stakeholder_input": "stakeholders provide performance feedback"
            },
            "analysis_and_action": {
                "trend_analysis": "system identifies performance trends and patterns",
                "anomaly_detection": "system detects performance anomalies",
                "improvement_suggestions": "system suggests performance improvements",
                "action_triggering": "system automatically triggers actions for performance issues"
            }
        }
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get the current status of the task designer component."""
        return {
            "name": self.name,
            "version": self.version,
            "templates_available": len(self.task_templates),
            "decomposition_strategies": len(self.decomposition_strategies),
            "auto_decompose_enabled": self.auto_decompose,
            "max_decomposition_depth": self.max_decomposition_depth,
            "complexity_threshold": self.complexity_threshold.value,
            "tasks_created": len(self.framework.tasks) if self.framework else 0,
            "team_features": {
                "team_task_assignment": True,
                "cross_functional_tasks": True,
                "autonomous_workflows": True,
                "collaborative_optimization": True
            }
        }