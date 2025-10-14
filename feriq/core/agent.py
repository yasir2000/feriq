"""Enhanced Agent class for the Feriq framework, extending CrewAI capabilities."""

from typing import Dict, List, Any, Optional, Set, Callable
from pydantic import BaseModel, Field
from crewai import Agent
from enum import Enum
import uuid
from datetime import datetime

from .role import Role
from .task import FeriqTask
from .goal import Goal


class AgentState(str, Enum):
    """States that an agent can be in."""
    IDLE = "idle"
    BUSY = "busy"
    LEARNING = "learning"
    COLLABORATING = "collaborating"
    REASONING = "reasoning"
    OFFLINE = "offline"
    ERROR = "error"


class AgentStatus(str, Enum):
    """Status of an agent."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    INITIALIZING = "initializing"
    READY = "ready"


class AgentCapabilityLevel(str, Enum):
    """Capability proficiency levels."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class AgentMemory(BaseModel):
    """Represents agent memory for learning and adaptation."""
    short_term: Dict[str, Any] = Field(default_factory=dict, 
                                      description="Short-term memory")
    long_term: Dict[str, Any] = Field(default_factory=dict, 
                                     description="Long-term memory")
    episodic: List[Dict[str, Any]] = Field(default_factory=list, 
                                          description="Episodic memories")
    semantic: Dict[str, Any] = Field(default_factory=dict, 
                                    description="Semantic knowledge")


class AgentLearning(BaseModel):
    """Learning and adaptation configuration."""
    learning_enabled: bool = Field(default=True, description="Whether learning is enabled")
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0, 
                                description="Rate of learning")
    adaptation_threshold: float = Field(default=0.7, ge=0.0, le=1.0,
                                       description="Threshold for adaptation")
    experience_buffer_size: int = Field(default=1000, ge=1,
                                       description="Size of experience buffer")


class FeriqAgent(BaseModel):
    """
    Enhanced agent class for the Feriq framework that extends CrewAI Agent
    with additional capabilities for dynamic role assignment, learning,
    and collaborative behavior.
    """
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
                   description="Unique identifier for this agent")
    name: str = Field(..., description="Human-readable name for the agent")
    description: str = Field(default="", description="Description of the agent")
    
    # Agent state and status
    state: AgentState = Field(default=AgentState.IDLE, 
                             description="Current state of the agent")
    is_active: bool = Field(default=True, description="Whether agent is active")
    created_at: datetime = Field(default_factory=datetime.now, 
                                description="When agent was created")
    last_activity: datetime = Field(default_factory=datetime.now, 
                                   description="Last activity timestamp")
    
    # Role and capabilities
    current_role: Optional[Role] = Field(default=None, 
                                        description="Currently assigned role")
    available_roles: List[Role] = Field(default_factory=list,
                                       description="Roles this agent can assume")
    base_capabilities: Dict[str, float] = Field(default_factory=dict,
                                               description="Base capabilities (name -> level)")
    learned_capabilities: Dict[str, float] = Field(default_factory=dict,
                                                  description="Learned capabilities")
    
    # Task management
    current_task_ids: List[str] = Field(default_factory=list,
                                       description="Currently executing task IDs")
    completed_task_ids: List[str] = Field(default_factory=list,
                                         description="Completed task IDs")
    failed_task_ids: List[str] = Field(default_factory=list,
                                      description="Failed task IDs")
    task_queue: List[str] = Field(default_factory=list,
                                 description="Queued task IDs")
    max_concurrent_tasks: int = Field(default=3, ge=1,
                                     description="Maximum concurrent tasks")
    
    # Performance and learning
    performance_metrics: Dict[str, float] = Field(default_factory=dict,
                                                 description="Performance metrics")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0,
                               description="Task success rate")
    efficiency_score: float = Field(default=0.5, ge=0.0, le=1.0,
                                   description="Efficiency score")
    learning_config: AgentLearning = Field(default_factory=AgentLearning,
                                          description="Learning configuration")
    memory: AgentMemory = Field(default_factory=AgentMemory,
                               description="Agent memory system")
    
    # Collaboration and communication
    collaboration_score: float = Field(default=0.5, ge=0.0, le=1.0,
                                      description="Collaboration effectiveness")
    communication_style: str = Field(default="direct", 
                                    description="Communication style preference")
    trusted_agents: Set[str] = Field(default_factory=set,
                                    description="IDs of trusted agents")
    collaboration_history: List[Dict[str, Any]] = Field(default_factory=list,
                                                       description="Collaboration history")
    
    # Reasoning and decision making
    reasoning_enabled: bool = Field(default=True, 
                                   description="Whether reasoning is enabled")
    decision_confidence: float = Field(default=0.5, ge=0.0, le=1.0,
                                      description="Confidence in decisions")
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0,
                                 description="Risk tolerance level")
    
    # CrewAI integration
    crew_agent: Optional[Agent] = Field(default=None, 
                                       description="Underlying CrewAI agent")
    llm_config: Dict[str, Any] = Field(default_factory=dict,
                                      description="LLM configuration")
    tools: List[Any] = Field(default_factory=list, description="Available tools")
    
    # Workflow and orchestration
    workflow_context: Dict[str, Any] = Field(default_factory=dict,
                                            description="Current workflow context")
    coordination_rules: List[str] = Field(default_factory=list,
                                         description="Coordination rules")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                   description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        use_enum_values = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v)
        }
    
    def assign_role(self, role: Role) -> bool:
        """Assign a role to this agent."""
        # Check if agent can handle this role
        if not self._can_assume_role(role):
            return False
        
        # Update current role
        previous_role = self.current_role
        self.current_role = role
        
        # Add to available roles if not already there
        if role not in self.available_roles:
            self.available_roles.append(role)
        
        # Update agent state
        self.last_activity = datetime.now()
        
        # Store role change in memory
        self.memory.episodic.append({
            "event": "role_change",
            "previous_role": previous_role.name if previous_role else None,
            "new_role": role.name,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def _can_assume_role(self, role: Role) -> bool:
        """Check if agent can assume the given role."""
        # Check if agent has required capabilities
        for capability in role.capabilities:
            agent_level = self.get_capability_level(capability.name)
            if agent_level < capability.proficiency_level:
                return False
        
        return True
    
    def get_capability_level(self, capability_name: str) -> float:
        """Get the agent's proficiency level for a capability."""
        # Combine base and learned capabilities
        base_level = self.base_capabilities.get(capability_name, 0.0)
        learned_level = self.learned_capabilities.get(capability_name, 0.0)
        
        # Return the maximum of base and learned
        return max(base_level, learned_level)
    
    def learn_capability(self, capability_name: str, improvement: float) -> None:
        """Improve a capability through learning."""
        if not self.learning_config.learning_enabled:
            return
        
        current_level = self.get_capability_level(capability_name)
        learning_rate = self.learning_config.learning_rate
        
        # Apply learning with diminishing returns
        new_level = current_level + (improvement * learning_rate * (1 - current_level))
        self.learned_capabilities[capability_name] = min(1.0, new_level)
        
        # Update memory
        self.memory.long_term[f"capability_{capability_name}"] = new_level
    
    def assign_task(self, task_id: str) -> bool:
        """Assign a task to this agent."""
        # Check if agent can take more tasks
        if len(self.current_task_ids) >= self.max_concurrent_tasks:
            # Add to queue
            if task_id not in self.task_queue:
                self.task_queue.append(task_id)
            return False
        
        # Assign task
        if task_id not in self.current_task_ids:
            self.current_task_ids.append(task_id)
            self.state = AgentState.BUSY
            self.last_activity = datetime.now()
            
            # Remove from queue if it was there
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            
            return True
        
        return False
    
    def complete_task(self, task_id: str, success: bool = True, 
                     performance_score: float = 1.0) -> None:
        """Mark a task as completed."""
        if task_id in self.current_task_ids:
            self.current_task_ids.remove(task_id)
            
            if success:
                self.completed_task_ids.append(task_id)
            else:
                self.failed_task_ids.append(task_id)
            
            # Update performance metrics
            self._update_performance_metrics(success, performance_score)
            
            # Update state
            if not self.current_task_ids:
                self.state = AgentState.IDLE
                self._process_task_queue()
            
            self.last_activity = datetime.now()
    
    def _update_performance_metrics(self, success: bool, score: float) -> None:
        """Update agent performance metrics."""
        total_tasks = len(self.completed_task_ids) + len(self.failed_task_ids)
        
        if total_tasks > 0:
            self.success_rate = len(self.completed_task_ids) / total_tasks
        
        # Update efficiency score with weighted average
        if "efficiency_history" not in self.performance_metrics:
            self.performance_metrics["efficiency_history"] = []
        
        self.performance_metrics["efficiency_history"].append(score)
        
        # Keep only recent history
        if len(self.performance_metrics["efficiency_history"]) > 100:
            self.performance_metrics["efficiency_history"] = \
                self.performance_metrics["efficiency_history"][-100:]
        
        # Calculate weighted average (recent scores have more weight)
        history = self.performance_metrics["efficiency_history"]
        weights = [i / len(history) for i in range(1, len(history) + 1)]
        weighted_sum = sum(s * w for s, w in zip(history, weights))
        weight_sum = sum(weights)
        
        self.efficiency_score = weighted_sum / weight_sum if weight_sum > 0 else score
    
    def _process_task_queue(self) -> None:
        """Process queued tasks if agent becomes available."""
        while (self.task_queue and 
               len(self.current_task_ids) < self.max_concurrent_tasks):
            next_task_id = self.task_queue.pop(0)
            self.assign_task(next_task_id)
    
    def collaborate_with(self, other_agent_id: str, task_id: str, 
                        collaboration_type: str = "joint") -> bool:
        """Start collaboration with another agent."""
        self.state = AgentState.COLLABORATING
        
        # Record collaboration
        collaboration_record = {
            "agent_id": other_agent_id,
            "task_id": task_id,
            "collaboration_type": collaboration_type,
            "started_at": datetime.now().isoformat()
        }
        
        self.collaboration_history.append(collaboration_record)
        self.last_activity = datetime.now()
        
        return True
    
    def add_trusted_agent(self, agent_id: str) -> None:
        """Add an agent to the trusted list."""
        self.trusted_agents.add(agent_id)
    
    def is_trusted(self, agent_id: str) -> bool:
        """Check if an agent is trusted."""
        return agent_id in self.trusted_agents
    
    def make_decision(self, options: List[Dict[str, Any]], 
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a decision given options and context."""
        if not self.reasoning_enabled or not options:
            return options[0] if options else {}
        
        # Simple decision making based on agent preferences and capabilities
        scored_options = []
        
        for option in options:
            score = self._score_option(option, context or {})
            scored_options.append((score, option))
        
        # Sort by score and return best option
        scored_options.sort(key=lambda x: x[0], reverse=True)
        best_option = scored_options[0][1]
        
        # Update decision confidence based on score spread
        scores = [s[0] for s in scored_options]
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            self.decision_confidence = min(1.0, score_range)
        
        return best_option
    
    def _score_option(self, option: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Score an option for decision making."""
        score = 0.5  # Base score
        
        # Consider required capabilities
        if "required_capabilities" in option:
            capability_match = 0.0
            for cap in option["required_capabilities"]:
                capability_match += self.get_capability_level(cap)
            
            if option["required_capabilities"]:
                capability_match /= len(option["required_capabilities"])
            
            score += capability_match * 0.3
        
        # Consider risk
        if "risk_level" in option:
            risk_factor = 1.0 - abs(option["risk_level"] - self.risk_tolerance)
            score += risk_factor * 0.2
        
        # Consider efficiency
        if "complexity" in option:
            efficiency_factor = min(1.0, self.efficiency_score / option["complexity"])
            score += efficiency_factor * 0.3
        
        # Random factor for exploration
        import random
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def update_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Update agent based on feedback."""
        if not self.learning_config.learning_enabled:
            return
        
        # Extract learning from feedback
        if "capability_feedback" in feedback:
            for cap, improvement in feedback["capability_feedback"].items():
                self.learn_capability(cap, improvement)
        
        # Update collaboration score
        if "collaboration_score" in feedback:
            new_score = feedback["collaboration_score"]
            # Weighted average with current score
            self.collaboration_score = (self.collaboration_score * 0.7 + 
                                      new_score * 0.3)
        
        # Store feedback in memory
        self.memory.episodic.append({
            "event": "feedback_received",
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of agent status."""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "current_role": self.current_role.name if self.current_role else None,
            "active_tasks": len(self.current_task_ids),
            "queued_tasks": len(self.task_queue),
            "success_rate": self.success_rate,
            "efficiency_score": self.efficiency_score,
            "collaboration_score": self.collaboration_score,
            "last_activity": self.last_activity.isoformat()
        }
    
    def create_crew_agent(self, **kwargs) -> Agent:
        """Create and configure the underlying CrewAI agent."""
        agent_config = {
            "role": self.current_role.name if self.current_role else "Assistant",
            "goal": self.current_role.description if self.current_role else "Help accomplish tasks",
            "backstory": self.description,
            "tools": self.tools,
            **kwargs
        }
        
        self.crew_agent = Agent(**agent_config)
        return self.crew_agent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        data = self.model_dump()
        # Remove non-serializable crew_agent
        if "crew_agent" in data:
            del data["crew_agent"]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeriqAgent":
        """Create agent from dictionary representation."""
        # Remove crew_agent if present (will be recreated)
        if "crew_agent" in data:
            del data["crew_agent"]
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of the agent."""
        role_name = self.current_role.name if self.current_role else "No role"
        return f"Agent(id='{self.id[:8]}...', name='{self.name}', role='{role_name}', state='{self.state}')"
    
    def __repr__(self) -> str:
        """Developer representation of the agent."""
        return self.__str__()