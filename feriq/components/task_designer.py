"""Task Designer component for creating and decomposing tasks."""

from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import uuid

from ..core.task import FeriqTask, TaskType, TaskComplexity, TaskPriority, TaskDependency
from ..core.goal import Goal, GoalType, GoalPriority
from ..core.plan import Plan


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
            "tasks_created": len(self.framework.tasks) if self.framework else 0
        }