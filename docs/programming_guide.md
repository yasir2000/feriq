# Feriq Programming Guide

Comprehensive guide to programming with the Feriq Collaborative AI Agents Framework.

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Core Concepts](#core-concepts)
3. [Getting Started with Code](#getting-started-with-code)
4. [Agent Development](#agent-development)
5. [Goal and Task Management](#goal-and-task-management)
6. [Workflow Orchestration](#workflow-orchestration)
7. [Advanced Features](#advanced-features)
8. [Integration Patterns](#integration-patterns)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)

## Framework Overview

Feriq is built on top of CrewAI and extends it with advanced coordination, reasoning, and orchestration capabilities. The framework provides eight core components that work together to enable sophisticated multi-agent workflows.

### Core Architecture

```python
from feriq import FeriqFramework
from feriq.core import (
    DynamicRoleDesigner,
    TaskDesignerAllocator,
    PlanDesigner,
    PlanObserver,
    WorkflowOrchestrator,
    Choreographer,
    Reasoner
)

# Initialize the framework
framework = FeriqFramework()

# Access core components
role_designer = framework.role_designer
task_allocator = framework.task_allocator
plan_designer = framework.plan_designer
plan_observer = framework.plan_observer
orchestrator = framework.orchestrator
choreographer = framework.choreographer
reasoner = framework.reasoner
```

## Core Concepts

### 1. Agents

Feriq agents extend CrewAI agents with additional capabilities:

```python
from feriq import FeriqAgent
from feriq.core import Role, RoleCapability

# Define a role with capabilities
researcher_role = Role(
    name="AI Researcher",
    description="Specialist in AI research and analysis",
    capabilities=[
        RoleCapability("research", 0.9),
        RoleCapability("analysis", 0.8),
        RoleCapability("writing", 0.7)
    ],
    tools=["web_search", "document_analysis", "citation_manager"]
)

# Create an agent
agent = FeriqAgent(
    name="ResearchBot",
    role=researcher_role,
    capabilities=["research", "analysis", "writing"],
    learning_enabled=True,
    collaboration_style="cooperative"
)
```

### 2. Goals and Tasks

Goals represent high-level objectives, while tasks are specific work units:

```python
from feriq import Goal, Task, GoalType
from datetime import timedelta

# Define a goal
goal = Goal(
    name="market_research",
    title="Comprehensive Market Research",
    description="Conduct detailed market analysis for new product launch",
    goal_type=GoalType.RESEARCH,
    required_capabilities=["research", "analysis", "reporting"],
    estimated_duration=timedelta(hours=4),
    priority="high"
)

# Define tasks for the goal
tasks = [
    Task(
        name="competitor_analysis",
        description="Analyze top 10 competitors in the market",
        required_capabilities=["research", "analysis"],
        estimated_duration=timedelta(hours=1)
    ),
    Task(
        name="market_trends",
        description="Identify current market trends and patterns",
        required_capabilities=["research", "data_analysis"],
        estimated_duration=timedelta(hours=1.5)
    ),
    Task(
        name="report_generation",
        description="Generate comprehensive market research report",
        required_capabilities=["writing", "reporting"],
        estimated_duration=timedelta(hours=1.5)
    )
]
```

### 3. Plans and Workflows

Plans define how tasks are organized and executed:

```python
from feriq import Plan, Workflow, ExecutionStrategy

# Create a plan
plan = Plan(
    name="market_research_plan",
    goal=goal,
    tasks=tasks,
    execution_strategy=ExecutionStrategy.SEQUENTIAL,
    resource_allocation={
        "max_concurrent_tasks": 3,
        "memory_limit": "2GB",
        "time_limit": timedelta(hours=6)
    }
)

# Create a workflow
workflow = Workflow(
    name="market_research_workflow",
    description="Automated market research workflow",
    plan=plan,
    agents=[agent],
    coordination_pattern="pipeline"
)
```

## Getting Started with Code

### Basic Example

```python
import asyncio
from feriq import FeriqFramework, FeriqAgent, Goal, Task, GoalType, Role, RoleCapability
from datetime import timedelta

async def main():
    # Initialize framework
    framework = FeriqFramework()
    
    # Create a role
    analyst_role = Role(
        name="Data Analyst",
        description="Expert in data analysis and insights",
        capabilities=[
            RoleCapability("data_analysis", 0.9),
            RoleCapability("visualization", 0.8)
        ]
    )
    
    # Create an agent
    analyst = FeriqAgent(
        name="DataBot",
        role=analyst_role,
        capabilities=["data_analysis", "visualization"]
    )
    
    # Define a goal
    analysis_goal = Goal(
        name="sales_analysis",
        title="Q3 Sales Analysis",
        description="Analyze Q3 sales data and generate insights",
        goal_type=GoalType.ANALYSIS,
        required_capabilities=["data_analysis"]
    )
    
    # Create and execute workflow
    workflow = framework.create_workflow(
        name="sales_analysis_workflow",
        goal=analysis_goal,
        agents=[analyst]
    )
    
    # Execute the workflow
    result = await framework.execute_workflow(workflow)
    print(f"Workflow completed: {result.status}")
    print(f"Results: {result.outputs}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Framework Configuration

```python
from feriq import FeriqFramework
from feriq.utils import Config

# Create configuration
config = Config()
config.set("orchestrator.max_concurrent_workflows", 5)
config.set("reasoner.default_reasoning_type", "probabilistic")
config.set("logging.level", "INFO")

# Initialize framework with configuration
framework = FeriqFramework(config=config)

# Or load from file
framework = FeriqFramework.from_config_file("config.yaml")
```

## Agent Development

### Custom Agent Classes

```python
from feriq import FeriqAgent
from feriq.core import AgentCapability
from typing import Dict, Any, List

class ResearchAgent(FeriqAgent):
    """Specialized research agent with custom capabilities."""
    
    def __init__(self, name: str, research_domains: List[str] = None):
        super().__init__(
            name=name,
            role="Research Specialist",
            capabilities=["research", "analysis", "citation"]
        )
        self.research_domains = research_domains or []
        self.knowledge_base = {}
    
    async def conduct_research(self, topic: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """Conduct research on a specific topic."""
        # Custom research logic
        research_plan = await self.plan_research(topic, depth)
        sources = await self.gather_sources(topic)
        analysis = await self.analyze_sources(sources)
        
        return {
            "topic": topic,
            "plan": research_plan,
            "sources": sources,
            "analysis": analysis,
            "confidence": self.calculate_confidence(analysis)
        }
    
    async def plan_research(self, topic: str, depth: str) -> Dict[str, Any]:
        """Create a research plan."""
        return {
            "approach": self.determine_approach(topic),
            "sources_needed": self.identify_source_types(topic),
            "timeline": self.estimate_timeline(depth),
            "validation_criteria": self.define_validation_criteria(topic)
        }
    
    def add_domain_expertise(self, domain: str, expertise_level: float):
        """Add expertise in a specific domain."""
        self.research_domains.append(domain)
        self.capabilities[f"{domain}_research"] = AgentCapability(
            name=f"{domain}_research",
            level=expertise_level,
            description=f"Research expertise in {domain}"
        )
```

### Agent Collaboration

```python
from feriq.coordination import CoordinationPattern, CollaborationStyle

class CollaborativeTeam:
    """Manages a team of collaborating agents."""
    
    def __init__(self, agents: List[FeriqAgent], coordination_pattern: CoordinationPattern):
        self.agents = agents
        self.coordination_pattern = coordination_pattern
        self.shared_context = {}
    
    async def execute_collaborative_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task using collaborative agents."""
        
        # Determine optimal agent assignment
        assignments = await self.assign_agents_to_subtasks(task)
        
        # Execute subtasks with coordination
        results = []
        if self.coordination_pattern == CoordinationPattern.PIPELINE:
            results = await self.execute_pipeline(assignments)
        elif self.coordination_pattern == CoordinationPattern.SCATTER_GATHER:
            results = await self.execute_scatter_gather(assignments)
        elif self.coordination_pattern == CoordinationPattern.CONSENSUS:
            results = await self.execute_consensus(assignments)
        
        # Aggregate results
        final_result = await self.aggregate_results(results)
        
        return final_result
    
    async def assign_agents_to_subtasks(self, task: Task) -> Dict[str, List[FeriqAgent]]:
        """Intelligently assign agents to subtasks."""
        subtasks = await self.decompose_task(task)
        assignments = {}
        
        for subtask in subtasks:
            best_agents = self.select_best_agents(subtask, max_agents=2)
            assignments[subtask.name] = best_agents
        
        return assignments
    
    def select_best_agents(self, subtask: Task, max_agents: int = 1) -> List[FeriqAgent]:
        """Select the best agents for a subtask based on capabilities."""
        scores = []
        
        for agent in self.agents:
            score = self.calculate_agent_suitability(agent, subtask)
            scores.append((agent, score))
        
        # Sort by score and return top agents
        scores.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, score in scores[:max_agents]]
```

### Agent Learning and Adaptation

```python
from feriq.learning import LearningManager, ExperienceBuffer

class LearningAgent(FeriqAgent):
    """Agent with learning and adaptation capabilities."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.learning_manager = LearningManager()
        self.experience_buffer = ExperienceBuffer(max_size=1000)
        self.performance_history = []
    
    async def execute_task_with_learning(self, task: Task) -> Dict[str, Any]:
        """Execute task and learn from the experience."""
        
        # Record initial state
        initial_state = self.capture_state()
        
        # Execute task
        start_time = time.time()
        result = await self.execute_task(task)
        execution_time = time.time() - start_time
        
        # Evaluate performance
        performance = self.evaluate_performance(task, result, execution_time)
        
        # Store experience
        experience = {
            "task": task,
            "initial_state": initial_state,
            "actions": result.get("actions", []),
            "outcome": result,
            "performance": performance,
            "timestamp": time.time()
        }
        self.experience_buffer.add(experience)
        
        # Learn from experience
        await self.learn_from_experience(experience)
        
        # Update capabilities if needed
        await self.update_capabilities_from_learning()
        
        return result
    
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from a single experience."""
        
        # Identify patterns in successful/unsuccessful approaches
        patterns = self.learning_manager.identify_patterns([experience])
        
        # Update internal models
        for pattern in patterns:
            await self.update_internal_model(pattern)
        
        # Adjust strategies
        if experience["performance"]["success_rate"] < 0.7:
            await self.adjust_strategy(experience["task"].category)
    
    def update_capabilities_from_learning(self):
        """Update agent capabilities based on learning."""
        
        # Analyze recent performance
        recent_experiences = self.experience_buffer.get_recent(limit=50)
        
        # Calculate capability improvements
        capability_updates = self.learning_manager.calculate_capability_updates(
            recent_experiences
        )
        
        # Apply updates
        for capability, improvement in capability_updates.items():
            if capability in self.capabilities:
                current_level = self.capabilities[capability].level
                new_level = min(1.0, current_level + improvement)
                self.capabilities[capability].level = new_level
```

## Goal and Task Management

### Dynamic Goal Creation

```python
from feriq.planning import GoalGenerator, TaskDecomposer

class DynamicGoalManager:
    """Manages dynamic goal creation and adaptation."""
    
    def __init__(self, framework: FeriqFramework):
        self.framework = framework
        self.goal_generator = GoalGenerator()
        self.task_decomposer = TaskDecomposer()
    
    async def create_goal_from_description(self, description: str, context: Dict[str, Any] = None) -> Goal:
        """Create a goal from natural language description."""
        
        # Analyze description using NLP
        analysis = await self.analyze_goal_description(description)
        
        # Generate goal structure
        goal = Goal(
            name=analysis["name"],
            title=analysis["title"],
            description=description,
            goal_type=analysis["type"],
            required_capabilities=analysis["capabilities"],
            estimated_duration=analysis["duration"],
            priority=analysis["priority"],
            context=context or {}
        )
        
        # Decompose into tasks
        tasks = await self.task_decomposer.decompose_goal(goal)
        goal.tasks = tasks
        
        return goal
    
    async def adapt_goal_based_on_progress(self, goal: Goal, progress_data: Dict[str, Any]) -> Goal:
        """Adapt a goal based on execution progress."""
        
        # Analyze current progress
        progress_analysis = self.analyze_progress(progress_data)
        
        # Identify needed adaptations
        adaptations = []
        
        if progress_analysis["behind_schedule"]:
            adaptations.append("prioritize_critical_tasks")
        
        if progress_analysis["resource_constraints"]:
            adaptations.append("optimize_resource_usage")
        
        if progress_analysis["scope_expansion_needed"]:
            adaptations.append("expand_scope")
        
        # Apply adaptations
        adapted_goal = await self.apply_adaptations(goal, adaptations)
        
        return adapted_goal
```

### Task Prioritization and Scheduling

```python
from feriq.scheduling import TaskScheduler, PriorityCalculator
from typing import List
import heapq

class SmartTaskScheduler:
    """Intelligent task scheduling with priority calculation."""
    
    def __init__(self):
        self.priority_calculator = PriorityCalculator()
        self.task_queue = []
        self.dependencies = {}
    
    def add_task(self, task: Task, dependencies: List[str] = None):
        """Add a task to the scheduler."""
        
        # Calculate priority score
        priority = self.priority_calculator.calculate_priority(task)
        
        # Add to priority queue (negative priority for max-heap behavior)
        heapq.heappush(self.task_queue, (-priority, task.name, task))
        
        # Record dependencies
        if dependencies:
            self.dependencies[task.name] = dependencies
    
    def get_next_task(self, available_agents: List[FeriqAgent]) -> Task:
        """Get the next task that can be executed."""
        
        temp_queue = []
        next_task = None
        
        while self.task_queue and not next_task:
            priority, name, task = heapq.heappop(self.task_queue)
            
            # Check if dependencies are satisfied
            if self.are_dependencies_satisfied(task):
                # Check if we have suitable agents
                if self.has_suitable_agents(task, available_agents):
                    next_task = task
                else:
                    # Put back in queue for later
                    temp_queue.append((priority, name, task))
            else:
                # Dependencies not satisfied, put back
                temp_queue.append((priority, name, task))
        
        # Restore unselected tasks to queue
        for item in temp_queue:
            heapq.heappush(self.task_queue, item)
        
        return next_task
    
    def are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied."""
        if task.name not in self.dependencies:
            return True
        
        for dep in self.dependencies[task.name]:
            if not self.is_task_completed(dep):
                return False
        
        return True
```

## Workflow Orchestration

### Custom Orchestration Patterns

```python
from feriq.orchestration import OrchestrationPattern, ExecutionContext
from enum import Enum

class CustomOrchestrationPattern(OrchestrationPattern):
    """Custom orchestration pattern for specialized workflows."""
    
    async def execute(self, workflow: Workflow, context: ExecutionContext) -> Dict[str, Any]:
        """Execute workflow using custom pattern."""
        
        # Phase 1: Parallel initialization
        initialization_tasks = self.get_initialization_tasks(workflow)
        init_results = await self.execute_parallel(initialization_tasks, context)
        
        # Phase 2: Sequential core execution with feedback loops
        core_tasks = self.get_core_tasks(workflow)
        core_results = await self.execute_with_feedback(core_tasks, context)
        
        # Phase 3: Consensus-based finalization
        finalization_tasks = self.get_finalization_tasks(workflow)
        final_results = await self.execute_consensus(finalization_tasks, context)
        
        # Aggregate all results
        aggregated_results = self.aggregate_results([
            init_results,
            core_results,
            final_results
        ])
        
        return aggregated_results
    
    async def execute_with_feedback(self, tasks: List[Task], context: ExecutionContext) -> Dict[str, Any]:
        """Execute tasks with feedback loops."""
        results = {}
        
        for task in tasks:
            # Execute task
            result = await self.execute_single_task(task, context)
            
            # Evaluate result quality
            quality_score = self.evaluate_quality(result)
            
            # If quality is low, retry with different approach
            if quality_score < 0.7:
                # Analyze failure and adjust approach
                adjusted_task = await self.adjust_task_approach(task, result)
                result = await self.execute_single_task(adjusted_task, context)
            
            results[task.name] = result
            
            # Update context with new information
            context.update_from_result(result)
        
        return results
```

### Workflow Monitoring and Adaptation

```python
from feriq.monitoring import WorkflowMonitor, PerformanceMetrics
import asyncio

class AdaptiveWorkflowManager:
    """Manages workflows with real-time monitoring and adaptation."""
    
    def __init__(self, framework: FeriqFramework):
        self.framework = framework
        self.monitor = WorkflowMonitor()
        self.active_workflows = {}
    
    async def execute_adaptive_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute workflow with adaptive monitoring."""
        
        # Start monitoring
        monitor_task = asyncio.create_task(
            self.monitor_workflow_execution(workflow)
        )
        
        # Execute workflow
        execution_task = asyncio.create_task(
            self.framework.execute_workflow(workflow)
        )
        
        # Wait for completion or intervention
        done, pending = await asyncio.wait(
            [monitor_task, execution_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Handle results
        if execution_task in done:
            # Workflow completed normally
            monitor_task.cancel()
            return execution_task.result()
        else:
            # Monitor detected issue, handle adaptation
            adaptation_needed = monitor_task.result()
            return await self.handle_workflow_adaptation(
                workflow, adaptation_needed, execution_task
            )
    
    async def monitor_workflow_execution(self, workflow: Workflow) -> Dict[str, Any]:
        """Monitor workflow execution for issues."""
        
        start_time = time.time()
        
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            # Collect metrics
            metrics = await self.collect_workflow_metrics(workflow)
            
            # Check for issues
            issues = self.detect_issues(metrics)
            
            if issues:
                return {
                    "adaptation_type": "performance_issue",
                    "issues": issues,
                    "metrics": metrics,
                    "elapsed_time": time.time() - start_time
                }
            
            # Check for timeout
            if time.time() - start_time > workflow.max_execution_time:
                return {
                    "adaptation_type": "timeout",
                    "elapsed_time": time.time() - start_time
                }
    
    async def handle_workflow_adaptation(self, 
                                       workflow: Workflow, 
                                       adaptation_info: Dict[str, Any],
                                       execution_task: asyncio.Task) -> Dict[str, Any]:
        """Handle workflow adaptation based on monitoring results."""
        
        adaptation_type = adaptation_info["adaptation_type"]
        
        if adaptation_type == "performance_issue":
            # Scale resources or optimize tasks
            optimized_workflow = await self.optimize_workflow(workflow, adaptation_info)
            
            # Cancel current execution and restart with optimized workflow
            execution_task.cancel()
            return await self.framework.execute_workflow(optimized_workflow)
        
        elif adaptation_type == "timeout":
            # Implement graceful degradation
            partial_results = await self.collect_partial_results(workflow)
            
            return {
                "status": "partial_completion",
                "results": partial_results,
                "completion_percentage": self.calculate_completion_percentage(workflow)
            }
```

## Advanced Features

### Reasoning Integration

```python
from feriq.reasoning import ReasoningEngine, ReasoningType

class ReasoningAgent(FeriqAgent):
    """Agent with integrated reasoning capabilities."""
    
    def __init__(self, name: str, reasoning_types: List[ReasoningType] = None):
        super().__init__(name=name)
        self.reasoning_engine = ReasoningEngine()
        self.reasoning_types = reasoning_types or [
            ReasoningType.DEDUCTIVE,
            ReasoningType.PROBABILISTIC
        ]
    
    async def solve_problem_with_reasoning(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a problem using multiple reasoning approaches."""
        
        solutions = {}
        
        for reasoning_type in self.reasoning_types:
            solution = await self.reasoning_engine.reason(
                problem=problem,
                reasoning_type=reasoning_type,
                context=self.get_context()
            )
            solutions[reasoning_type.value] = solution
        
        # Combine solutions using meta-reasoning
        final_solution = await self.reasoning_engine.combine_solutions(
            solutions, problem
        )
        
        return final_solution
    
    async def explain_reasoning(self, solution: Dict[str, Any]) -> str:
        """Generate explanation for reasoning process."""
        return await self.reasoning_engine.generate_explanation(
            solution, include_confidence=True
        )
```

### Knowledge Management

```python
from feriq.knowledge import KnowledgeBase, MemoryManager

class KnowledgeAwareAgent(FeriqAgent):
    """Agent with persistent knowledge management."""
    
    def __init__(self, name: str, knowledge_base_path: str = None):
        super().__init__(name=name)
        self.knowledge_base = KnowledgeBase(knowledge_base_path)
        self.memory_manager = MemoryManager()
    
    async def learn_from_interaction(self, interaction: Dict[str, Any]):
        """Learn and store knowledge from interactions."""
        
        # Extract knowledge from interaction
        knowledge_items = await self.extract_knowledge(interaction)
        
        # Store in knowledge base
        for item in knowledge_items:
            await self.knowledge_base.store(item)
        
        # Update working memory
        self.memory_manager.update_working_memory(knowledge_items)
    
    async def retrieve_relevant_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve knowledge relevant to a query."""
        
        # Search knowledge base
        kb_results = await self.knowledge_base.search(query)
        
        # Search working memory
        memory_results = self.memory_manager.search_working_memory(query)
        
        # Combine and rank results
        combined_results = self.combine_knowledge_sources(kb_results, memory_results)
        
        return combined_results
```

## Integration Patterns

### External Tool Integration

```python
from feriq.tools import ToolRegistry, ExternalTool
from typing import Any, Dict

class CustomToolAgent(FeriqAgent):
    """Agent with custom external tool integration."""
    
    def __init__(self, name: str):
        super().__init__(name=name)
        self.tool_registry = ToolRegistry()
        self.setup_tools()
    
    def setup_tools(self):
        """Register custom tools."""
        
        # Web search tool
        self.tool_registry.register_tool(
            name="web_search",
            tool=ExternalTool(
                name="web_search",
                description="Search the web for information",
                function=self.web_search,
                parameters={
                    "query": {"type": "string", "required": True},
                    "max_results": {"type": "integer", "default": 10}
                }
            )
        )
        
        # Database query tool
        self.tool_registry.register_tool(
            name="database_query",
            tool=ExternalTool(
                name="database_query",
                description="Query database for information",
                function=self.database_query,
                parameters={
                    "sql": {"type": "string", "required": True},
                    "database": {"type": "string", "default": "default"}
                }
            )
        )
    
    async def web_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform web search."""
        # Implementation here
        pass
    
    async def database_query(self, sql: str, database: str = "default") -> List[Dict[str, Any]]:
        """Query database."""
        # Implementation here
        pass
```

### API Integration

```python
from feriq.integrations import APIIntegration, HTTPClient

class APIIntegratedWorkflow:
    """Workflow with external API integration."""
    
    def __init__(self):
        self.http_client = HTTPClient()
        self.api_integrations = {}
    
    def register_api(self, name: str, base_url: str, auth_config: Dict[str, Any]):
        """Register an external API."""
        self.api_integrations[name] = APIIntegration(
            name=name,
            base_url=base_url,
            auth_config=auth_config,
            client=self.http_client
        )
    
    async def execute_api_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with API calls."""
        
        results = {}
        
        for step in workflow_config["steps"]:
            if step["type"] == "api_call":
                api_name = step["api"]
                endpoint = step["endpoint"]
                params = step.get("params", {})
                
                # Make API call
                api = self.api_integrations[api_name]
                response = await api.call(endpoint, params)
                
                results[step["name"]] = response
            
            elif step["type"] == "agent_task":
                # Execute agent task using API results
                agent_result = await self.execute_agent_task(
                    step["agent"],
                    step["task"],
                    context=results
                )
                results[step["name"]] = agent_result
        
        return results
```

## Best Practices

### 1. Agent Design Principles

```python
# Good: Single responsibility
class ResearchAgent(FeriqAgent):
    """Focused on research tasks only."""
    pass

class AnalysisAgent(FeriqAgent):
    """Focused on analysis tasks only."""
    pass

# Avoid: Multi-purpose agents
class SuperAgent(FeriqAgent):
    """Handles everything - harder to maintain and optimize."""
    pass
```

### 2. Error Handling and Resilience

```python
from feriq.exceptions import FeriqException, TaskExecutionError
import asyncio
from typing import Optional

class ResilientAgent(FeriqAgent):
    """Agent with robust error handling."""
    
    async def execute_task_with_retry(self, 
                                    task: Task, 
                                    max_retries: int = 3,
                                    backoff_factor: float = 2.0) -> Dict[str, Any]:
        """Execute task with retry logic."""
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self.execute_task(task)
            
            except TaskExecutionError as e:
                last_exception = e
                
                if attempt < max_retries:
                    # Calculate backoff delay
                    delay = backoff_factor ** attempt
                    await asyncio.sleep(delay)
                    
                    # Log retry attempt
                    self.log_retry_attempt(task, attempt + 1, e)
                    
                    # Potentially modify task based on error
                    task = await self.adapt_task_for_retry(task, e)
                else:
                    # Max retries reached
                    raise FeriqException(
                        f"Task {task.name} failed after {max_retries} retries"
                    ) from last_exception
```

### 3. Performance Optimization

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedWorkflow:
    """Workflow optimized for performance."""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.task_cache = {}
    
    async def execute_optimized(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute tasks with performance optimizations."""
        
        # Group tasks by type for batch processing
        task_groups = self.group_tasks_by_type(tasks)
        
        # Execute groups in parallel where possible
        group_results = await asyncio.gather(*[
            self.execute_task_group(group) 
            for group in task_groups
        ])
        
        # Combine results
        return self.combine_group_results(group_results)
    
    async def execute_cpu_intensive_task(self, task: Task) -> Dict[str, Any]:
        """Execute CPU-intensive task in thread pool."""
        
        # Check cache first
        cache_key = self.generate_cache_key(task)
        if cache_key in self.task_cache:
            return self.task_cache[cache_key]
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.thread_pool,
            self.cpu_intensive_function,
            task
        )
        
        # Cache result
        self.task_cache[cache_key] = result
        
        return result
```

### 4. Testing and Validation

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestFeriqAgent:
    """Test cases for Feriq agents."""
    
    @pytest.fixture
    def sample_agent(self):
        return FeriqAgent(
            name="TestAgent",
            capabilities=["test_capability"]
        )
    
    @pytest.mark.asyncio
    async def test_agent_task_execution(self, sample_agent):
        """Test basic task execution."""
        
        # Create test task
        task = Task(
            name="test_task",
            description="Test task description",
            required_capabilities=["test_capability"]
        )
        
        # Mock external dependencies
        sample_agent.external_service = AsyncMock()
        sample_agent.external_service.process.return_value = {"status": "success"}
        
        # Execute task
        result = await sample_agent.execute_task(task)
        
        # Verify results
        assert result["status"] == "completed"
        assert "output" in result
        
        # Verify external service was called
        sample_agent.external_service.process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test complete workflow execution."""
        
        # Setup test environment
        framework = FeriqFramework(config={"testing": True})
        
        # Create test workflow
        workflow = self.create_test_workflow()
        
        # Execute workflow
        result = await framework.execute_workflow(workflow)
        
        # Validate results
        assert result["status"] == "completed"
        assert len(result["task_results"]) == len(workflow.tasks)
```

## API Reference

### Core Classes

#### FeriqFramework

```python
class FeriqFramework:
    """Main framework class for orchestrating multi-agent workflows."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the framework with optional configuration."""
        pass
    
    async def execute_workflow(self, workflow: Workflow) -> WorkflowResult:
        """Execute a complete workflow."""
        pass
    
    def create_agent(self, name: str, role: Role, **kwargs) -> FeriqAgent:
        """Create a new agent."""
        pass
    
    def create_workflow(self, name: str, goal: Goal, agents: List[FeriqAgent], **kwargs) -> Workflow:
        """Create a new workflow."""
        pass
```

#### FeriqAgent

```python
class FeriqAgent:
    """Enhanced agent with collaboration and learning capabilities."""
    
    def __init__(self, 
                 name: str, 
                 role: Union[str, Role], 
                 capabilities: List[str] = None,
                 learning_enabled: bool = False,
                 **kwargs):
        """Initialize agent with role and capabilities."""
        pass
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        pass
    
    async def collaborate_with(self, other_agents: List['FeriqAgent'], task: Task) -> TaskResult:
        """Collaborate with other agents on a task."""
        pass
```

For complete API documentation, see the [API Reference](api_reference.md).

---

*This programming guide provides comprehensive coverage of the Feriq framework. For CLI usage, see the [CLI User Guide](cli_guide.md).*