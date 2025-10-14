"""
Choreographer Component

This module implements the choreographer for managing agent interactions,
coordination patterns, and communication orchestration in multi-agent workflows.
"""

from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import asyncio
from collections import defaultdict, deque
import threading
import time
from pydantic import BaseModel, Field
import networkx as nx

from ..core.agent import FeriqAgent, AgentStatus
from ..core.task import FeriqTask, TaskStatus
from ..core.role import Role
from ..utils.logger import FeriqLogger
from ..utils.config import Config


class CoordinationPattern(str, Enum):
    """Types of coordination patterns for agent interactions."""
    PIPELINE = "pipeline"
    BROADCAST = "broadcast"
    SCATTER_GATHER = "scatter_gather"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    CONSENSUS = "consensus"
    COMPETITION = "competition"
    COLLABORATION = "collaboration"


class InteractionType(str, Enum):
    """Types of agent interactions."""
    COMMUNICATION = "communication"
    COORDINATION = "coordination"
    COLLABORATION = "collaboration"
    NEGOTIATION = "negotiation"
    INFORMATION_SHARING = "information_sharing"
    RESOURCE_SHARING = "resource_sharing"
    SYNCHRONIZATION = "synchronization"
    CONFLICT_RESOLUTION = "conflict_resolution"


class MessageType(str, Enum):
    """Types of messages between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_OFFER = "resource_offer"
    COORDINATION_REQUEST = "coordination_request"
    INFORMATION_SHARE = "information_share"
    NEGOTIATION_PROPOSAL = "negotiation_proposal"
    CONSENSUS_VOTE = "consensus_vote"
    ERROR_NOTIFICATION = "error_notification"


@dataclass
class Message:
    """Represents a message between agents."""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    requires_response: bool = False
    correlation_id: Optional[str] = None
    expiry_time: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        return self.expiry_time is not None and datetime.now() > self.expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
            "expiry_time": self.expiry_time.isoformat() if self.expiry_time else None
        }


@dataclass
class Interaction:
    """Represents an interaction between agents."""
    interaction_id: str
    interaction_type: InteractionType
    participants: Set[str]
    initiator_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    messages: List[Message] = field(default_factory=list)
    outcome: Optional[str] = None
    success: bool = False
    
    def add_message(self, message: Message):
        """Add a message to this interaction."""
        self.messages.append(message)
    
    def is_active(self) -> bool:
        """Check if the interaction is still active."""
        return self.end_time is None
    
    def get_duration(self) -> Optional[timedelta]:
        """Get the duration of the interaction."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


class CommunicationHub:
    """Central hub for managing agent communication."""
    
    def __init__(self, logger: FeriqLogger):
        self.logger = logger
        self.message_queue: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.message_history: deque = deque(maxlen=5000)
        self.subscribers: Dict[MessageType, Set[str]] = defaultdict(set)
        self.message_handlers: Dict[str, Dict[MessageType, Callable]] = defaultdict(dict)
        
    def subscribe(self, agent_id: str, message_type: MessageType, handler: Callable[[Message], None]):
        """Subscribe an agent to a message type."""
        self.subscribers[message_type].add(agent_id)
        self.message_handlers[agent_id][message_type] = handler
        
        self.logger.debug(
            "Agent subscribed to message type",
            agent_id=agent_id,
            message_type=message_type
        )
    
    def unsubscribe(self, agent_id: str, message_type: MessageType):
        """Unsubscribe an agent from a message type."""
        self.subscribers[message_type].discard(agent_id)
        if agent_id in self.message_handlers and message_type in self.message_handlers[agent_id]:
            del self.message_handlers[agent_id][message_type]
    
    def send_message(self, message: Message):
        """Send a message to its recipient."""
        if message.is_expired():
            self.logger.warning("Attempted to send expired message", message_id=message.message_id)
            return
        
        # Add to recipient's queue
        self.message_queue[message.receiver_id].append(message)
        
        # Add to history
        self.message_history.append(message)
        
        # Notify subscribers
        for subscriber_id in self.subscribers[message.message_type]:
            if subscriber_id in self.message_handlers:
                handler = self.message_handlers[subscriber_id].get(message.message_type)
                if handler:
                    try:
                        handler(message)
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {e}")
        
        self.logger.debug(
            "Message sent",
            message_id=message.message_id,
            sender_id=message.sender_id,
            receiver_id=message.receiver_id,
            message_type=message.message_type
        )
    
    def get_messages(self, agent_id: str, limit: Optional[int] = None) -> List[Message]:
        """Get messages for an agent."""
        messages = list(self.message_queue[agent_id])
        if limit:
            messages = messages[-limit:]
        return messages
    
    def acknowledge_message(self, agent_id: str, message_id: str):
        """Acknowledge receipt of a message."""
        queue = self.message_queue[agent_id]
        for i, message in enumerate(queue):
            if message.message_id == message_id:
                del queue[i]
                break
    
    def broadcast_message(self, sender_id: str, message_type: MessageType, content: Dict[str, Any], recipients: Optional[Set[str]] = None):
        """Broadcast a message to multiple recipients."""
        if recipients is None:
            recipients = self.subscribers[message_type]
        
        for recipient_id in recipients:
            if recipient_id != sender_id:  # Don't send to self
                message = Message(
                    message_id=str(uuid.uuid4()),
                    sender_id=sender_id,
                    receiver_id=recipient_id,
                    message_type=message_type,
                    content=content,
                    timestamp=datetime.now()
                )
                self.send_message(message)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "total_agents": len(self.message_queue),
            "total_messages": len(self.message_history),
            "message_types": {
                msg_type.value: len(subscribers) 
                for msg_type, subscribers in self.subscribers.items()
            },
            "pending_messages": sum(len(queue) for queue in self.message_queue.values())
        }


class CoordinationEngine:
    """Engine for managing coordination patterns and strategies."""
    
    def __init__(self, logger: FeriqLogger):
        self.logger = logger
        self.coordination_patterns: Dict[str, CoordinationPattern] = {}
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        
    def register_coordination_pattern(self, pattern_id: str, pattern: CoordinationPattern, config: Dict[str, Any]):
        """Register a coordination pattern."""
        self.coordination_patterns[pattern_id] = pattern
        self.active_coordinations[pattern_id] = {
            "pattern": pattern,
            "config": config,
            "participants": set(),
            "state": "inactive",
            "created_at": datetime.now()
        }
        
        self.logger.info(
            "Coordination pattern registered",
            pattern_id=pattern_id,
            pattern=pattern
        )
    
    def start_coordination(self, pattern_id: str, participants: Set[str], initiator_id: str) -> str:
        """Start a coordination process."""
        if pattern_id not in self.coordination_patterns:
            raise ValueError(f"Unknown coordination pattern: {pattern_id}")
        
        coordination_id = str(uuid.uuid4())
        pattern = self.coordination_patterns[pattern_id]
        
        coordination_state = {
            "coordination_id": coordination_id,
            "pattern_id": pattern_id,
            "pattern": pattern,
            "participants": participants,
            "initiator_id": initiator_id,
            "state": "active",
            "start_time": datetime.now(),
            "context": {}
        }
        
        self.active_coordinations[coordination_id] = coordination_state
        
        # Initialize pattern-specific coordination
        if pattern == CoordinationPattern.PIPELINE:
            self._initialize_pipeline_coordination(coordination_state)
        elif pattern == CoordinationPattern.SCATTER_GATHER:
            self._initialize_scatter_gather_coordination(coordination_state)
        elif pattern == CoordinationPattern.CONSENSUS:
            self._initialize_consensus_coordination(coordination_state)
        
        self.logger.info(
            "Coordination started",
            coordination_id=coordination_id,
            pattern=pattern,
            participants=list(participants),
            initiator_id=initiator_id
        )
        
        return coordination_id
    
    def _initialize_pipeline_coordination(self, coordination_state: Dict[str, Any]):
        """Initialize pipeline coordination pattern."""
        participants = list(coordination_state["participants"])
        coordination_state["context"]["pipeline_order"] = participants
        coordination_state["context"]["current_stage"] = 0
        coordination_state["context"]["completed_stages"] = []
    
    def _initialize_scatter_gather_coordination(self, coordination_state: Dict[str, Any]):
        """Initialize scatter-gather coordination pattern."""
        coordination_state["context"]["scattered_tasks"] = {}
        coordination_state["context"]["gathered_results"] = {}
        coordination_state["context"]["pending_responses"] = set(coordination_state["participants"])
    
    def _initialize_consensus_coordination(self, coordination_state: Dict[str, Any]):
        """Initialize consensus coordination pattern."""
        coordination_state["context"]["votes"] = {}
        coordination_state["context"]["required_majority"] = len(coordination_state["participants"]) // 2 + 1
        coordination_state["context"]["voting_deadline"] = datetime.now() + timedelta(minutes=10)
    
    def update_coordination_state(self, coordination_id: str, update: Dict[str, Any]):
        """Update the state of a coordination process."""
        if coordination_id not in self.active_coordinations:
            return False
        
        coordination_state = self.active_coordinations[coordination_id]
        coordination_state["context"].update(update)
        
        # Check if coordination is complete
        self._check_coordination_completion(coordination_id)
        
        return True
    
    def _check_coordination_completion(self, coordination_id: str):
        """Check if a coordination process has completed."""
        coordination_state = self.active_coordinations[coordination_id]
        pattern = coordination_state["pattern"]
        
        if pattern == CoordinationPattern.PIPELINE:
            pipeline_order = coordination_state["context"]["pipeline_order"]
            completed_stages = coordination_state["context"]["completed_stages"]
            if len(completed_stages) >= len(pipeline_order):
                self._complete_coordination(coordination_id, "success")
        
        elif pattern == CoordinationPattern.SCATTER_GATHER:
            pending_responses = coordination_state["context"]["pending_responses"]
            if len(pending_responses) == 0:
                self._complete_coordination(coordination_id, "success")
        
        elif pattern == CoordinationPattern.CONSENSUS:
            votes = coordination_state["context"]["votes"]
            required_majority = coordination_state["context"]["required_majority"]
            voting_deadline = coordination_state["context"]["voting_deadline"]
            
            # Check if majority reached
            vote_counts = defaultdict(int)
            for vote in votes.values():
                vote_counts[vote] += 1
            
            max_votes = max(vote_counts.values()) if vote_counts else 0
            if max_votes >= required_majority:
                self._complete_coordination(coordination_id, "consensus_reached")
            elif datetime.now() > voting_deadline:
                self._complete_coordination(coordination_id, "timeout")
    
    def _complete_coordination(self, coordination_id: str, outcome: str):
        """Complete a coordination process."""
        coordination_state = self.active_coordinations[coordination_id]
        coordination_state["state"] = "completed"
        coordination_state["end_time"] = datetime.now()
        coordination_state["outcome"] = outcome
        
        self.logger.info(
            "Coordination completed",
            coordination_id=coordination_id,
            outcome=outcome,
            duration=(coordination_state["end_time"] - coordination_state["start_time"]).total_seconds()
        )
    
    def get_coordination_status(self, coordination_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a coordination process."""
        return self.active_coordinations.get(coordination_id)


class InteractionTracker:
    """Tracks and analyzes agent interactions."""
    
    def __init__(self, logger: FeriqLogger):
        self.logger = logger
        self.interactions: Dict[str, Interaction] = {}
        self.interaction_history: List[Interaction] = []
        self.agent_interaction_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
    def start_interaction(
        self,
        interaction_type: InteractionType,
        participants: Set[str],
        initiator_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start tracking a new interaction."""
        interaction_id = str(uuid.uuid4())
        
        interaction = Interaction(
            interaction_id=interaction_id,
            interaction_type=interaction_type,
            participants=participants,
            initiator_id=initiator_id,
            start_time=datetime.now(),
            context=context or {}
        )
        
        self.interactions[interaction_id] = interaction
        
        # Update interaction counts
        for participant in participants:
            self.agent_interaction_counts[initiator_id][participant] += 1
        
        self.logger.info(
            "Interaction started",
            interaction_id=interaction_id,
            interaction_type=interaction_type,
            participants=list(participants),
            initiator_id=initiator_id
        )
        
        return interaction_id
    
    def add_message_to_interaction(self, interaction_id: str, message: Message):
        """Add a message to an interaction."""
        if interaction_id in self.interactions:
            self.interactions[interaction_id].add_message(message)
    
    def end_interaction(self, interaction_id: str, outcome: str, success: bool = True):
        """End an interaction."""
        if interaction_id not in self.interactions:
            return False
        
        interaction = self.interactions[interaction_id]
        interaction.end_time = datetime.now()
        interaction.outcome = outcome
        interaction.success = success
        
        # Move to history
        self.interaction_history.append(interaction)
        del self.interactions[interaction_id]
        
        self.logger.info(
            "Interaction ended",
            interaction_id=interaction_id,
            outcome=outcome,
            success=success,
            duration=interaction.get_duration().total_seconds() if interaction.get_duration() else 0
        )
        
        return True
    
    def get_agent_interaction_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get interaction statistics for an agent."""
        # Count active interactions
        active_interactions = sum(
            1 for interaction in self.interactions.values()
            if agent_id in interaction.participants
        )
        
        # Count completed interactions
        completed_interactions = sum(
            1 for interaction in self.interaction_history
            if agent_id in interaction.participants
        )
        
        # Count interactions by type
        interaction_type_counts = defaultdict(int)
        for interaction in self.interaction_history:
            if agent_id in interaction.participants:
                interaction_type_counts[interaction.interaction_type.value] += 1
        
        # Calculate success rate
        successful_interactions = sum(
            1 for interaction in self.interaction_history
            if agent_id in interaction.participants and interaction.success
        )
        
        success_rate = (
            successful_interactions / completed_interactions 
            if completed_interactions > 0 else 0
        )
        
        return {
            "agent_id": agent_id,
            "active_interactions": active_interactions,
            "completed_interactions": completed_interactions,
            "success_rate": success_rate,
            "interaction_types": dict(interaction_type_counts),
            "partner_counts": dict(self.agent_interaction_counts[agent_id])
        }


class Choreographer:
    """
    Choreographer component that manages agent interactions and coordination patterns.
    
    This component provides:
    - Communication orchestration between agents
    - Coordination pattern management (pipeline, broadcast, consensus, etc.)
    - Interaction tracking and analysis
    - Conflict resolution and negotiation support
    - Performance optimization for multi-agent workflows
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = FeriqLogger("Choreographer", self.config)
        
        # Core components
        self.communication_hub = CommunicationHub(self.logger)
        self.coordination_engine = CoordinationEngine(self.logger)
        self.interaction_tracker = InteractionTracker(self.logger)
        
        # Agent registry
        self.registered_agents: Dict[str, FeriqAgent] = {}
        
        # Coordination rules and strategies
        self.coordination_rules: Dict[str, Dict[str, Any]] = {}
        self.default_patterns: Dict[str, CoordinationPattern] = {
            "sequential_tasks": CoordinationPattern.PIPELINE,
            "parallel_tasks": CoordinationPattern.SCATTER_GATHER,
            "decision_making": CoordinationPattern.CONSENSUS,
            "information_sharing": CoordinationPattern.BROADCAST
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance metrics
        self.choreography_metrics: Dict[str, Any] = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "coordination_patterns_used": defaultdict(int),
            "average_interaction_duration": 0.0
        }
        
        self.logger.info("Choreographer initialized successfully")
    
    def register_agent(self, agent: FeriqAgent):
        """Register an agent for choreographed interactions."""
        self.registered_agents[agent.agent_id] = agent
        
        # Set up default message handlers
        self._setup_default_handlers(agent)
        
        self.logger.info("Agent registered for choreography", agent_id=agent.agent_id)
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from choreography."""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
            
            # Clean up message subscriptions
            for message_type in MessageType:
                self.communication_hub.unsubscribe(agent_id, message_type)
            
            self.logger.info("Agent unregistered from choreography", agent_id=agent_id)
    
    def _setup_default_handlers(self, agent: FeriqAgent):
        """Set up default message handlers for an agent."""
        # Status update handler
        def handle_status_update(message: Message):
            self.logger.debug(f"Agent {agent.agent_id} received status update from {message.sender_id}")
        
        # Task request handler
        def handle_task_request(message: Message):
            self.logger.debug(f"Agent {agent.agent_id} received task request from {message.sender_id}")
        
        self.communication_hub.subscribe(agent.agent_id, MessageType.STATUS_UPDATE, handle_status_update)
        self.communication_hub.subscribe(agent.agent_id, MessageType.TASK_REQUEST, handle_task_request)
    
    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: int = 1,
        requires_response: bool = False,
        expiry_minutes: Optional[int] = None
    ) -> str:
        """Send a message between agents."""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            priority=priority,
            requires_response=requires_response,
            expiry_time=datetime.now() + timedelta(minutes=expiry_minutes) if expiry_minutes else None
        )
        
        self.communication_hub.send_message(message)
        return message.message_id
    
    def broadcast_message(
        self,
        sender_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        recipients: Optional[Set[str]] = None
    ):
        """Broadcast a message to multiple agents."""
        if recipients is None:
            recipients = set(self.registered_agents.keys())
        
        self.communication_hub.broadcast_message(sender_id, message_type, content, recipients)
    
    def coordinate_agents(
        self,
        pattern: CoordinationPattern,
        participants: List[str],
        initiator_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Coordinate a group of agents using a specific pattern."""
        # Register coordination pattern if not exists
        pattern_id = f"{pattern.value}_{uuid.uuid4().hex[:8]}"
        self.coordination_engine.register_coordination_pattern(
            pattern_id,
            pattern,
            context or {}
        )
        
        # Start coordination
        coordination_id = self.coordination_engine.start_coordination(
            pattern_id,
            set(participants),
            initiator_id
        )
        
        # Track as interaction
        interaction_id = self.interaction_tracker.start_interaction(
            InteractionType.COORDINATION,
            set(participants),
            initiator_id,
            {"coordination_id": coordination_id, "pattern": pattern.value}
        )
        
        # Execute pattern-specific coordination
        if pattern == CoordinationPattern.PIPELINE:
            self._execute_pipeline_coordination(participants, coordination_id, interaction_id)
        elif pattern == CoordinationPattern.SCATTER_GATHER:
            self._execute_scatter_gather_coordination(participants, coordination_id, interaction_id, context or {})
        elif pattern == CoordinationPattern.BROADCAST:
            self._execute_broadcast_coordination(participants, initiator_id, interaction_id, context or {})
        elif pattern == CoordinationPattern.CONSENSUS:
            self._execute_consensus_coordination(participants, coordination_id, interaction_id, context or {})
        
        # Update metrics
        self.choreography_metrics["coordination_patterns_used"][pattern.value] += 1
        
        return coordination_id
    
    def _execute_pipeline_coordination(self, participants: List[str], coordination_id: str, interaction_id: str):
        """Execute pipeline coordination pattern."""
        if len(participants) < 2:
            return
        
        # Send coordination messages in sequence
        for i in range(len(participants) - 1):
            current_agent = participants[i]
            next_agent = participants[i + 1]
            
            message_id = self.send_message(
                sender_id="choreographer",
                receiver_id=current_agent,
                message_type=MessageType.COORDINATION_REQUEST,
                content={
                    "coordination_id": coordination_id,
                    "pattern": "pipeline",
                    "next_agent": next_agent,
                    "stage": i
                },
                requires_response=True
            )
    
    def _execute_scatter_gather_coordination(
        self,
        participants: List[str],
        coordination_id: str,
        interaction_id: str,
        context: Dict[str, Any]
    ):
        """Execute scatter-gather coordination pattern."""
        # Scatter phase: send tasks to all participants
        for participant in participants:
            message_id = self.send_message(
                sender_id="choreographer",
                receiver_id=participant,
                message_type=MessageType.TASK_REQUEST,
                content={
                    "coordination_id": coordination_id,
                    "pattern": "scatter_gather",
                    "task_data": context.get("task_data", {}),
                    "gather_point": context.get("gather_point", "choreographer")
                },
                requires_response=True,
                expiry_minutes=30
            )
    
    def _execute_broadcast_coordination(
        self,
        participants: List[str],
        initiator_id: str,
        interaction_id: str,
        context: Dict[str, Any]
    ):
        """Execute broadcast coordination pattern."""
        self.broadcast_message(
            sender_id=initiator_id,
            message_type=MessageType.INFORMATION_SHARE,
            content={
                "pattern": "broadcast",
                "data": context.get("broadcast_data", {}),
                "interaction_id": interaction_id
            },
            recipients=set(participants)
        )
    
    def _execute_consensus_coordination(
        self,
        participants: List[str],
        coordination_id: str,
        interaction_id: str,
        context: Dict[str, Any]
    ):
        """Execute consensus coordination pattern."""
        # Send voting request to all participants
        for participant in participants:
            message_id = self.send_message(
                sender_id="choreographer",
                receiver_id=participant,
                message_type=MessageType.CONSENSUS_VOTE,
                content={
                    "coordination_id": coordination_id,
                    "pattern": "consensus",
                    "proposal": context.get("proposal", {}),
                    "voting_deadline": (datetime.now() + timedelta(minutes=10)).isoformat()
                },
                requires_response=True,
                expiry_minutes=10
            )
    
    def handle_coordination_response(self, message: Message):
        """Handle responses from coordination processes."""
        content = message.content
        coordination_id = content.get("coordination_id")
        
        if not coordination_id:
            return
        
        coordination_state = self.coordination_engine.get_coordination_status(coordination_id)
        if not coordination_state:
            return
        
        pattern = coordination_state["pattern"]
        
        if pattern == CoordinationPattern.SCATTER_GATHER:
            # Handle gather phase
            update = {
                f"result_{message.sender_id}": content.get("result", {}),
                "pending_responses": coordination_state["context"]["pending_responses"] - {message.sender_id}
            }
            self.coordination_engine.update_coordination_state(coordination_id, update)
        
        elif pattern == CoordinationPattern.CONSENSUS:
            # Handle vote
            votes = coordination_state["context"]["votes"]
            votes[message.sender_id] = content.get("vote")
            update = {"votes": votes}
            self.coordination_engine.update_coordination_state(coordination_id, update)
    
    def resolve_conflict(
        self,
        conflicting_agents: List[str],
        conflict_type: str,
        context: Dict[str, Any]
    ) -> str:
        """Resolve conflicts between agents."""
        resolution_id = str(uuid.uuid4())
        
        # Start conflict resolution interaction
        interaction_id = self.interaction_tracker.start_interaction(
            InteractionType.CONFLICT_RESOLUTION,
            set(conflicting_agents),
            "choreographer",
            {"conflict_type": conflict_type, "resolution_id": resolution_id}
        )
        
        # Send conflict resolution request
        for agent_id in conflicting_agents:
            self.send_message(
                sender_id="choreographer",
                receiver_id=agent_id,
                message_type=MessageType.NEGOTIATION_PROPOSAL,
                content={
                    "resolution_id": resolution_id,
                    "conflict_type": conflict_type,
                    "context": context,
                    "participants": conflicting_agents
                },
                requires_response=True,
                expiry_minutes=15
            )
        
        self.logger.info(
            "Conflict resolution initiated",
            resolution_id=resolution_id,
            conflict_type=conflict_type,
            participants=conflicting_agents
        )
        
        return resolution_id
    
    def get_agent_messages(self, agent_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages for an agent."""
        messages = self.communication_hub.get_messages(agent_id, limit)
        return [msg.to_dict() for msg in messages]
    
    def acknowledge_message(self, agent_id: str, message_id: str):
        """Acknowledge receipt of a message."""
        self.communication_hub.acknowledge_message(agent_id, message_id)
    
    def get_coordination_status(self, coordination_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a coordination process."""
        return self.coordination_engine.get_coordination_status(coordination_id)
    
    def get_interaction_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get interaction statistics for an agent."""
        return self.interaction_tracker.get_agent_interaction_stats(agent_id)
    
    def add_coordination_rule(self, rule_name: str, condition: Callable, pattern: CoordinationPattern, config: Dict[str, Any]):
        """Add a coordination rule for automatic pattern selection."""
        self.coordination_rules[rule_name] = {
            "condition": condition,
            "pattern": pattern,
            "config": config
        }
    
    def suggest_coordination_pattern(self, agents: List[str], task_context: Dict[str, Any]) -> CoordinationPattern:
        """Suggest the best coordination pattern for a given context."""
        # Check rules first
        for rule_name, rule in self.coordination_rules.items():
            try:
                if rule["condition"](agents, task_context):
                    return rule["pattern"]
            except Exception as e:
                self.logger.error(f"Error in coordination rule {rule_name}: {e}")
        
        # Default suggestions based on context
        agent_count = len(agents)
        
        if agent_count == 1:
            return CoordinationPattern.PEER_TO_PEER
        elif agent_count == 2:
            return CoordinationPattern.PEER_TO_PEER
        elif task_context.get("requires_consensus"):
            return CoordinationPattern.CONSENSUS
        elif task_context.get("sequential_execution"):
            return CoordinationPattern.PIPELINE
        elif task_context.get("parallel_execution"):
            return CoordinationPattern.SCATTER_GATHER
        elif task_context.get("information_sharing"):
            return CoordinationPattern.BROADCAST
        else:
            return CoordinationPattern.COLLABORATION
    
    def get_choreographer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive choreographer statistics."""
        comm_stats = self.communication_hub.get_communication_stats()
        
        # Calculate average interaction duration
        completed_interactions = [
            interaction for interaction in self.interaction_tracker.interaction_history
            if interaction.get_duration() is not None
        ]
        
        avg_duration = 0.0
        if completed_interactions:
            total_duration = sum(interaction.get_duration().total_seconds() for interaction in completed_interactions)
            avg_duration = total_duration / len(completed_interactions)
        
        return {
            "registered_agents": len(self.registered_agents),
            "communication_stats": comm_stats,
            "active_interactions": len(self.interaction_tracker.interactions),
            "completed_interactions": len(self.interaction_tracker.interaction_history),
            "average_interaction_duration": avg_duration,
            "coordination_patterns_used": dict(self.choreography_metrics["coordination_patterns_used"]),
            "coordination_rules": len(self.coordination_rules),
            "success_rate": (
                self.choreography_metrics["successful_interactions"] / 
                max(self.choreography_metrics["total_interactions"], 1)
            )
        }