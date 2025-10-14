"""Task Allocator component for intelligently assigning tasks to agents."""

from typing import Dict, List, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import heapq

from ..core.task import FeriqTask, TaskStatus, TaskPriority, TaskComplexity
from ..core.agent import FeriqAgent, AgentState
from ..core.role import Role


class AllocationScore(BaseModel):
    """Represents the allocation score for an agent-task pair."""
    agent_id: str = Field(..., description="Agent ID")
    task_id: str = Field(..., description="Task ID")
    total_score: float = Field(..., description="Total allocation score")
    capability_score: float = Field(default=0.0, description="Capability match score")
    availability_score: float = Field(default=0.0, description="Availability score")
    efficiency_score: float = Field(default=0.0, description="Efficiency score")
    workload_score: float = Field(default=0.0, description="Workload balance score")
    preference_score: float = Field(default=0.0, description="Preference match score")
    collaboration_score: float = Field(default=0.0, description="Collaboration potential score")
    reasoning: str = Field(default="", description="Reasoning for the score")


class AllocationStrategy(BaseModel):
    """Strategy for task allocation."""
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    weight_capability: float = Field(default=0.4, description="Weight for capability matching")
    weight_availability: float = Field(default=0.2, description="Weight for availability")
    weight_efficiency: float = Field(default=0.15, description="Weight for efficiency")
    weight_workload: float = Field(default=0.15, description="Weight for workload balance")
    weight_preference: float = Field(default=0.05, description="Weight for preferences")
    weight_collaboration: float = Field(default=0.05, description="Weight for collaboration")
    prioritize_specialists: bool = Field(default=True, description="Whether to prioritize specialists")
    balance_workload: bool = Field(default=True, description="Whether to balance workload")
    consider_learning: bool = Field(default=True, description="Whether to consider learning opportunities")


class TaskAllocator(BaseModel):
    """
    Task Allocator component that intelligently assigns tasks to agents
    based on capabilities, availability, workload, and other factors.
    """
    
    # Component identification
    name: str = Field(default="TaskAllocator", description="Component name")
    version: str = Field(default="1.0", description="Component version")
    
    # Allocation strategies
    strategies: Dict[str, AllocationStrategy] = Field(default_factory=dict, description="Available allocation strategies")
    default_strategy: str = Field(default="balanced", description="Default allocation strategy")
    
    # Allocation history and learning
    allocation_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of allocations")
    performance_feedback: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, 
                                                                 description="Performance feedback for allocations")
    
    # Optimization parameters
    max_concurrent_tasks_per_agent: int = Field(default=3, description="Maximum concurrent tasks per agent")
    rebalancing_threshold: float = Field(default=0.8, description="Workload threshold for rebalancing")
    efficiency_learning_rate: float = Field(default=0.1, description="Learning rate for efficiency updates")
    
    # Framework reference
    framework: Optional[Any] = Field(default=None, description="Reference to main framework")
    
    # Real-time allocation
    allocation_queue: List[str] = Field(default_factory=list, description="Queue of tasks awaiting allocation")
    priority_queue: List[Tuple[float, str]] = Field(default_factory=list, description="Priority queue for urgent tasks")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize the task allocator with default strategies."""
        super().__init__(**kwargs)
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default allocation strategies."""
        # Balanced Strategy
        balanced_strategy = AllocationStrategy(
            name="balanced",
            description="Balanced approach considering all factors equally",
            weight_capability=0.3,
            weight_availability=0.2,
            weight_efficiency=0.2,
            weight_workload=0.2,
            weight_preference=0.05,
            weight_collaboration=0.05,
            prioritize_specialists=True,
            balance_workload=True,
            consider_learning=True
        )
        
        # Capability-First Strategy
        capability_first_strategy = AllocationStrategy(
            name="capability_first",
            description="Prioritizes capability matching above all else",
            weight_capability=0.6,
            weight_availability=0.15,
            weight_efficiency=0.15,
            weight_workload=0.05,
            weight_preference=0.025,
            weight_collaboration=0.025,
            prioritize_specialists=True,
            balance_workload=False,
            consider_learning=False
        )
        
        # Efficiency-Focused Strategy
        efficiency_strategy = AllocationStrategy(
            name="efficiency_focused",
            description="Focuses on agent efficiency and past performance",
            weight_capability=0.25,
            weight_availability=0.15,
            weight_efficiency=0.4,
            weight_workload=0.1,
            weight_preference=0.05,
            weight_collaboration=0.05,
            prioritize_specialists=False,
            balance_workload=True,
            consider_learning=True
        )
        
        # Workload Balancing Strategy
        workload_strategy = AllocationStrategy(
            name="workload_balancing",
            description="Prioritizes even distribution of work",
            weight_capability=0.2,
            weight_availability=0.3,
            weight_efficiency=0.1,
            weight_workload=0.35,
            weight_preference=0.025,
            weight_collaboration=0.025,
            prioritize_specialists=False,
            balance_workload=True,
            consider_learning=True
        )
        
        # Learning-Oriented Strategy
        learning_strategy = AllocationStrategy(
            name="learning_oriented",
            description="Assigns tasks to promote agent learning and development",
            weight_capability=0.2,
            weight_availability=0.2,
            weight_efficiency=0.1,
            weight_workload=0.2,
            weight_preference=0.1,
            weight_collaboration=0.2,
            prioritize_specialists=False,
            balance_workload=True,
            consider_learning=True
        )
        
        self.strategies = {
            "balanced": balanced_strategy,
            "capability_first": capability_first_strategy,
            "efficiency_focused": efficiency_strategy,
            "workload_balancing": workload_strategy,
            "learning_oriented": learning_strategy
        }
    
    def allocate_task(self, task_id: str, strategy_name: str = None, 
                     excluded_agents: Set[str] = None) -> bool:
        """Allocate a task to the best available agent."""
        if not self.framework:
            return False
        
        task = self.framework.tasks.get(task_id)
        if not task:
            return False
        
        # Use specified strategy or default
        strategy_name = strategy_name or self.default_strategy
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            strategy = self.strategies[self.default_strategy]
        
        # Get available agents
        available_agents = self._get_available_agents(excluded_agents or set())
        if not available_agents:
            # Add to queue if no agents available
            self.allocation_queue.append(task_id)
            return False
        
        # Calculate allocation scores
        scores = self._calculate_allocation_scores(task, available_agents, strategy)
        
        if not scores:
            return False
        
        # Select best agent
        best_score = max(scores, key=lambda s: s.total_score)
        best_agent = self.framework.get_agent(best_score.agent_id)
        
        if not best_agent:
            return False
        
        # Perform allocation
        success = self._perform_allocation(task, best_agent, best_score)
        
        if success:
            # Record allocation
            self._record_allocation(task_id, best_agent.id, best_score, strategy_name)
        
        return success
    
    def _get_available_agents(self, excluded_agents: Set[str]) -> List[FeriqAgent]:
        """Get list of available agents for task allocation."""
        if not self.framework:
            return []
        
        available = []
        for agent in self.framework.agents.values():
            if (agent.id not in excluded_agents and
                agent.is_active and
                agent.state in [AgentState.IDLE, AgentState.BUSY] and
                len(agent.current_task_ids) < self.max_concurrent_tasks_per_agent):
                available.append(agent)
        
        return available
    
    def _calculate_allocation_scores(self, task: FeriqTask, agents: List[FeriqAgent], 
                                   strategy: AllocationStrategy) -> List[AllocationScore]:
        """Calculate allocation scores for all agent-task pairs."""
        scores = []
        
        for agent in agents:
            score = self._calculate_agent_task_score(task, agent, strategy)
            scores.append(score)
        
        return scores
    
    def _calculate_agent_task_score(self, task: FeriqTask, agent: FeriqAgent, 
                                  strategy: AllocationStrategy) -> AllocationScore:
        """Calculate allocation score for a specific agent-task pair."""
        # Initialize score components
        capability_score = self._calculate_capability_score(task, agent)
        availability_score = self._calculate_availability_score(task, agent)
        efficiency_score = self._calculate_efficiency_score(task, agent)
        workload_score = self._calculate_workload_score(task, agent)
        preference_score = self._calculate_preference_score(task, agent)
        collaboration_score = self._calculate_collaboration_score(task, agent)
        
        # Apply strategy weights
        total_score = (
            capability_score * strategy.weight_capability +
            availability_score * strategy.weight_availability +
            efficiency_score * strategy.weight_efficiency +
            workload_score * strategy.weight_workload +
            preference_score * strategy.weight_preference +
            collaboration_score * strategy.weight_collaboration
        )
        
        # Apply strategy modifiers
        if strategy.prioritize_specialists and self._is_specialist_for_task(agent, task):
            total_score *= 1.2
        
        if strategy.consider_learning and self._has_learning_opportunity(agent, task):
            total_score *= 1.1
        
        # Generate reasoning
        reasoning = self._generate_allocation_reasoning(
            task, agent, capability_score, availability_score, 
            efficiency_score, workload_score
        )
        
        return AllocationScore(
            agent_id=agent.id,
            task_id=task.id,
            total_score=total_score,
            capability_score=capability_score,
            availability_score=availability_score,
            efficiency_score=efficiency_score,
            workload_score=workload_score,
            preference_score=preference_score,
            collaboration_score=collaboration_score,
            reasoning=reasoning
        )
    
    def _calculate_capability_score(self, task: FeriqTask, agent: FeriqAgent) -> float:
        """Calculate capability match score."""
        if not task.required_capabilities:
            return 0.5  # Neutral score for tasks with no specific requirements
        
        total_score = 0.0
        for capability in task.required_capabilities:
            agent_level = agent.get_capability_level(capability)
            
            # Determine required level based on task complexity
            required_level = self._get_required_capability_level(task.complexity)
            
            if agent_level >= required_level:
                # Agent exceeds requirement
                score = 1.0
            else:
                # Agent doesn't fully meet requirement
                score = agent_level / required_level if required_level > 0 else 0.0
            
            total_score += score
        
        return total_score / len(task.required_capabilities)
    
    def _get_required_capability_level(self, complexity: TaskComplexity) -> float:
        """Get required capability level based on task complexity."""
        complexity_levels = {
            TaskComplexity.TRIVIAL: 0.3,
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MODERATE: 0.7,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.EXPERT: 0.9
        }
        return complexity_levels.get(complexity, 0.7)
    
    def _calculate_availability_score(self, task: FeriqTask, agent: FeriqAgent) -> float:
        """Calculate availability score based on current workload and deadlines."""
        # Base availability score
        current_workload = len(agent.current_task_ids) / self.max_concurrent_tasks_per_agent
        availability_score = 1.0 - current_workload
        
        # Consider task deadline urgency
        if task.deadline:
            time_to_deadline = (task.deadline - datetime.now()).total_seconds() / 3600
            if time_to_deadline < 24:  # Very urgent
                availability_score *= 1.5 if agent.state == AgentState.IDLE else 0.8
            elif time_to_deadline < 72:  # Somewhat urgent
                availability_score *= 1.2 if agent.state == AgentState.IDLE else 0.9
        
        # Consider agent's queue length
        queue_factor = 1.0 / (1.0 + len(agent.task_queue) * 0.1)
        availability_score *= queue_factor
        
        return min(1.0, availability_score)
    
    def _calculate_efficiency_score(self, task: FeriqTask, agent: FeriqAgent) -> float:
        """Calculate efficiency score based on agent's past performance."""
        base_efficiency = agent.efficiency_score
        
        # Adjust based on task type match with agent's successful tasks
        if hasattr(agent, 'performance_metrics') and 'task_type_performance' in agent.performance_metrics:
            task_type_performance = agent.performance_metrics['task_type_performance']
            if task.task_type.value in task_type_performance:
                type_efficiency = task_type_performance[task.task_type.value]
                base_efficiency = (base_efficiency + type_efficiency) / 2
        
        # Consider recent performance trend
        if hasattr(agent, 'performance_metrics') and 'efficiency_history' in agent.performance_metrics:
            history = agent.performance_metrics['efficiency_history']
            if len(history) >= 3:
                recent_avg = sum(history[-3:]) / 3
                trend_factor = recent_avg / agent.efficiency_score if agent.efficiency_score > 0 else 1.0
                base_efficiency *= trend_factor
        
        return min(1.0, base_efficiency)
    
    def _calculate_workload_score(self, task: FeriqTask, agent: FeriqAgent) -> float:
        """Calculate workload balance score."""
        current_workload = len(agent.current_task_ids) / self.max_concurrent_tasks_per_agent
        
        # Higher score for agents with lower workload
        workload_score = 1.0 - current_workload
        
        # Consider task priority - high priority tasks can go to busier agents
        if task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
            workload_score = max(0.5, workload_score)  # Don't penalize too much for busy agents
        
        # Consider if agent can handle the additional workload
        if current_workload > 0.8:  # Agent is very busy
            workload_score *= 0.5
        
        return workload_score
    
    def _calculate_preference_score(self, task: FeriqTask, agent: FeriqAgent) -> float:
        """Calculate preference score based on agent and task characteristics."""
        score = 0.5  # Base neutral score
        
        # Check role compatibility
        if agent.current_role:
            role_suitability = agent.current_role.calculate_suitability_score(
                {cap: 0.7 for cap in task.required_capabilities}
            )
            score = (score + role_suitability) / 2
        
        # Consider task type preferences (if available in agent metadata)
        if 'task_type_preferences' in agent.metadata:
            preferences = agent.metadata['task_type_preferences']
            if task.task_type.value in preferences:
                preference_weight = preferences[task.task_type.value]
                score = (score + preference_weight) / 2
        
        # Consider collaboration preferences
        if task.goal_id and len(task.required_capabilities) > 3:  # Complex collaborative task
            collaboration_factor = agent.collaboration_score
            score = (score + collaboration_factor) / 2
        
        return score
    
    def _calculate_collaboration_score(self, task: FeriqTask, agent: FeriqAgent) -> float:
        """Calculate collaboration potential score."""
        if not task.goal_id:
            return 0.5  # Neutral for non-goal tasks
        
        # Check if other agents are working on the same goal
        goal_agents = []
        if self.framework:
            for other_agent in self.framework.agents.values():
                if other_agent.id != agent.id:
                    for task_id in other_agent.current_task_ids:
                        other_task = self.framework.tasks.get(task_id)
                        if other_task and other_task.goal_id == task.goal_id:
                            goal_agents.append(other_agent)
                            break
        
        if not goal_agents:
            return 0.5  # No collaboration needed
        
        # Calculate collaboration compatibility
        collaboration_score = 0.0
        for other_agent in goal_agents:
            # Check if agents have worked together before
            if other_agent.id in agent.trusted_agents:
                collaboration_score += 0.3
            
            # Check collaboration history
            for collab in agent.collaboration_history:
                if collab.get('agent_id') == other_agent.id:
                    collaboration_score += 0.2
                    break
            
            # Check complementary capabilities
            agent_caps = set(agent.base_capabilities.keys())
            other_caps = set(other_agent.base_capabilities.keys())
            if agent_caps.intersection(other_caps):
                collaboration_score += 0.2
            if agent_caps.difference(other_caps):  # Complementary skills
                collaboration_score += 0.3
        
        return min(1.0, collaboration_score / len(goal_agents))
    
    def _is_specialist_for_task(self, agent: FeriqAgent, task: FeriqTask) -> bool:
        """Check if agent is a specialist for the task."""
        if not agent.current_role or not task.required_capabilities:
            return False
        
        # Check if agent has high proficiency in most required capabilities
        high_proficiency_count = 0
        for capability in task.required_capabilities:
            if agent.get_capability_level(capability) >= 0.8:
                high_proficiency_count += 1
        
        return high_proficiency_count >= len(task.required_capabilities) * 0.7
    
    def _has_learning_opportunity(self, agent: FeriqAgent, task: FeriqTask) -> bool:
        """Check if task provides learning opportunity for agent."""
        if not agent.learning_config.learning_enabled:
            return False
        
        # Check if task has capabilities agent can improve
        for capability in task.required_capabilities:
            current_level = agent.get_capability_level(capability)
            if 0.3 <= current_level <= 0.8:  # Room for improvement
                return True
        
        return False
    
    def _generate_allocation_reasoning(self, task: FeriqTask, agent: FeriqAgent,
                                     capability_score: float, availability_score: float,
                                     efficiency_score: float, workload_score: float) -> str:
        """Generate human-readable reasoning for allocation decision."""
        reasons = []
        
        if capability_score > 0.8:
            reasons.append("high capability match")
        elif capability_score > 0.6:
            reasons.append("good capability match")
        else:
            reasons.append("adequate capability match")
        
        if availability_score > 0.8:
            reasons.append("highly available")
        elif availability_score > 0.5:
            reasons.append("available")
        else:
            reasons.append("limited availability")
        
        if efficiency_score > 0.8:
            reasons.append("high efficiency")
        elif efficiency_score > 0.6:
            reasons.append("good efficiency")
        
        if workload_score > 0.7:
            reasons.append("low current workload")
        elif workload_score < 0.3:
            reasons.append("high current workload")
        
        return ", ".join(reasons)
    
    def _perform_allocation(self, task: FeriqTask, agent: FeriqAgent, 
                          score: AllocationScore) -> bool:
        """Perform the actual task allocation."""
        # Assign task to agent
        success = agent.assign_task(task.id)
        if not success:
            return False
        
        # Update task
        task.assign_to_agent(agent.id, agent.current_role.name if agent.current_role else None)
        
        # Update framework tracking
        if self.framework:
            self.framework.last_activity = datetime.now()
        
        return True
    
    def _record_allocation(self, task_id: str, agent_id: str, score: AllocationScore, 
                          strategy_name: str) -> None:
        """Record allocation for learning and analysis."""
        allocation_record = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "agent_id": agent_id,
            "strategy": strategy_name,
            "score": score.total_score,
            "score_breakdown": {
                "capability": score.capability_score,
                "availability": score.availability_score,
                "efficiency": score.efficiency_score,
                "workload": score.workload_score,
                "preference": score.preference_score,
                "collaboration": score.collaboration_score
            },
            "reasoning": score.reasoning
        }
        
        self.allocation_history.append(allocation_record)
        
        # Keep only recent history
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-1000:]
    
    def batch_allocate_tasks(self, task_ids: List[str], strategy_name: str = None) -> Dict[str, bool]:
        """Allocate multiple tasks optimally."""
        results = {}
        
        if not task_ids:
            return results
        
        # Sort tasks by priority and deadline
        sorted_tasks = self._sort_tasks_for_allocation(task_ids)
        
        # Allocate tasks in order
        for task_id in sorted_tasks:
            success = self.allocate_task(task_id, strategy_name)
            results[task_id] = success
        
        return results
    
    def _sort_tasks_for_allocation(self, task_ids: List[str]) -> List[str]:
        """Sort tasks by priority and urgency for optimal allocation."""
        if not self.framework:
            return task_ids
        
        task_priority_scores = []
        
        for task_id in task_ids:
            task = self.framework.tasks.get(task_id)
            if not task:
                continue
            
            priority_score = task.calculate_priority_score()
            
            # Add urgency factor
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.now()).total_seconds() / 3600
                urgency_factor = max(0.1, 1.0 / (time_to_deadline + 1))
                priority_score *= (1 + urgency_factor)
            
            task_priority_scores.append((priority_score, task_id))
        
        # Sort by priority score (highest first)
        task_priority_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [task_id for _, task_id in task_priority_scores]
    
    def reallocate_tasks(self, agent_id: str = None, reason: str = "manual") -> Dict[str, Any]:
        """Reallocate tasks for better optimization."""
        if not self.framework:
            return {"status": "error", "message": "No framework available"}
        
        reallocated = []
        
        if agent_id:
            # Reallocate tasks for specific agent
            agent = self.framework.get_agent(agent_id)
            if not agent:
                return {"status": "error", "message": "Agent not found"}
            
            # Check if reallocation is needed
            current_workload = len(agent.current_task_ids) / self.max_concurrent_tasks_per_agent
            if current_workload <= self.rebalancing_threshold:
                return {"status": "no_action", "message": "Reallocation not needed"}
            
            # Find tasks that can be moved
            movable_tasks = []
            for task_id in agent.current_task_ids:
                task = self.framework.tasks.get(task_id)
                if task and task.status == TaskStatus.ASSIGNED:  # Not yet started
                    movable_tasks.append(task)
            
            # Try to reallocate some tasks
            for task in movable_tasks[:len(movable_tasks)//2]:  # Move half
                # Remove from current agent
                agent.current_task_ids.remove(task.id)
                task.assigned_agent_id = None
                task.status = TaskStatus.PENDING
                
                # Try to allocate to another agent
                success = self.allocate_task(task.id, excluded_agents={agent_id})
                if success:
                    reallocated.append(task.id)
                else:
                    # Return to original agent if can't allocate elsewhere
                    agent.assign_task(task.id)
                    task.assign_to_agent(agent_id)
        
        else:
            # Global rebalancing
            overloaded_agents = []
            for agent in self.framework.agents.values():
                current_workload = len(agent.current_task_ids) / self.max_concurrent_tasks_per_agent
                if current_workload > self.rebalancing_threshold:
                    overloaded_agents.append(agent)
            
            # Reallocate from overloaded agents
            for agent in overloaded_agents:
                result = self.reallocate_tasks(agent.id, reason)
                if result.get("reallocated"):
                    reallocated.extend(result["reallocated"])
        
        return {
            "status": "success",
            "reallocated": reallocated,
            "count": len(reallocated),
            "reason": reason
        }
    
    def process_allocation_queue(self) -> int:
        """Process queued task allocations."""
        allocated_count = 0
        
        # Process regular queue
        while self.allocation_queue:
            task_id = self.allocation_queue.pop(0)
            success = self.allocate_task(task_id)
            if success:
                allocated_count += 1
            else:
                # Put back in queue if still can't allocate
                self.allocation_queue.append(task_id)
                break  # Avoid infinite loop
        
        # Process priority queue
        while self.priority_queue:
            _, task_id = heapq.heappop(self.priority_queue)
            success = self.allocate_task(task_id)
            if success:
                allocated_count += 1
            else:
                # Put back in priority queue
                priority_score = self._calculate_task_priority(task_id)
                heapq.heappush(self.priority_queue, (-priority_score, task_id))
                break
        
        return allocated_count
    
    def _calculate_task_priority(self, task_id: str) -> float:
        """Calculate priority score for task queueing."""
        if not self.framework:
            return 0.0
        
        task = self.framework.tasks.get(task_id)
        if not task:
            return 0.0
        
        return task.calculate_priority_score()
    
    def add_to_priority_queue(self, task_id: str) -> None:
        """Add task to priority queue for urgent allocation."""
        priority_score = self._calculate_task_priority(task_id)
        heapq.heappush(self.priority_queue, (-priority_score, task_id))
    
    def update_allocation_feedback(self, task_id: str, agent_id: str, 
                                 performance_data: Dict[str, Any]) -> None:
        """Update allocation feedback for learning."""
        key = f"{agent_id}:{task_id}"
        
        if key not in self.performance_feedback:
            self.performance_feedback[key] = []
        
        feedback_record = {
            "timestamp": datetime.now().isoformat(),
            "performance": performance_data,
            "task_completed": performance_data.get("completed", False),
            "efficiency": performance_data.get("efficiency", 0.5),
            "quality": performance_data.get("quality", 0.5)
        }
        
        self.performance_feedback[key].append(feedback_record)
        
        # Update agent efficiency if task completed
        if performance_data.get("completed", False):
            agent = self.framework.get_agent(agent_id) if self.framework else None
            if agent:
                efficiency = performance_data.get("efficiency", 0.5)
                # Simple learning update
                current_efficiency = agent.efficiency_score
                new_efficiency = (current_efficiency * (1 - self.efficiency_learning_rate) + 
                                efficiency * self.efficiency_learning_rate)
                agent.efficiency_score = new_efficiency
    
    def get_allocation_analytics(self) -> Dict[str, Any]:
        """Get analytics on allocation performance."""
        if not self.allocation_history:
            return {"status": "no_data"}
        
        # Calculate success rates by strategy
        strategy_performance = {}
        for record in self.allocation_history:
            strategy = record["strategy"]
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {"count": 0, "total_score": 0.0}
            
            strategy_performance[strategy]["count"] += 1
            strategy_performance[strategy]["total_score"] += record["score"]
        
        # Calculate averages
        for strategy in strategy_performance:
            data = strategy_performance[strategy]
            data["average_score"] = data["total_score"] / data["count"]
        
        # Recent allocation trends
        recent_allocations = self.allocation_history[-100:] if len(self.allocation_history) >= 100 else self.allocation_history
        recent_avg_score = sum(r["score"] for r in recent_allocations) / len(recent_allocations) if recent_allocations else 0.0
        
        return {
            "total_allocations": len(self.allocation_history),
            "strategy_performance": strategy_performance,
            "recent_average_score": recent_avg_score,
            "queue_length": len(self.allocation_queue),
            "priority_queue_length": len(self.priority_queue),
            "feedback_records": len(self.performance_feedback)
        }
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get the current status of the task allocator component."""
        return {
            "name": self.name,
            "version": self.version,
            "strategies_available": len(self.strategies),
            "default_strategy": self.default_strategy,
            "allocation_queue_length": len(self.allocation_queue),
            "priority_queue_length": len(self.priority_queue),
            "total_allocations": len(self.allocation_history),
            "max_concurrent_tasks": self.max_concurrent_tasks_per_agent,
            "rebalancing_threshold": self.rebalancing_threshold
        }