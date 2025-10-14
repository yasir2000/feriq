"""
Reasoner Component

This module implements the reasoning engine for intelligent decision-making,
problem-solving, context analysis, and strategic planning in multi-agent workflows.
"""

from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import math

from ..core.goal import Goal, GoalType, GoalStatus
from ..core.task import FeriqTask, TaskStatus, TaskComplexity
from ..core.plan import Plan, PlanStatus
from ..core.agent import FeriqAgent, AgentStatus
from ..core.role import Role
from ..utils.logger import FeriqLogger
from ..utils.config import Config


class ReasoningType(str, Enum):
    """Types of reasoning processes."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    STRATEGIC = "strategic"
    TEMPORAL = "temporal"


class DecisionType(str, Enum):
    """Types of decisions that can be made."""
    TASK_ASSIGNMENT = "task_assignment"
    RESOURCE_ALLOCATION = "resource_allocation"
    PLAN_MODIFICATION = "plan_modification"
    STRATEGY_SELECTION = "strategy_selection"
    CONFLICT_RESOLUTION = "conflict_resolution"
    OPTIMIZATION = "optimization"
    RISK_ASSESSMENT = "risk_assessment"
    ADAPTATION = "adaptation"


class ContextType(str, Enum):
    """Types of context for reasoning."""
    CURRENT_STATE = "current_state"
    HISTORICAL = "historical"
    ENVIRONMENTAL = "environmental"
    AGENT_SPECIFIC = "agent_specific"
    TASK_SPECIFIC = "task_specific"
    GOAL_SPECIFIC = "goal_specific"
    SYSTEM_WIDE = "system_wide"


@dataclass
class ReasoningContext:
    """Context information for reasoning processes."""
    context_id: str
    context_type: ContextType
    timestamp: datetime
    data: Dict[str, Any]
    relevance_score: float = 1.0
    expiry_time: Optional[datetime] = None
    source: str = "unknown"
    
    def is_expired(self) -> bool:
        """Check if the context has expired."""
        return self.expiry_time is not None and datetime.now() > self.expiry_time
    
    def is_relevant_for_goal(self, goal_id: str) -> bool:
        """Check if context is relevant for a specific goal."""
        return (
            self.context_type in [ContextType.GOAL_SPECIFIC, ContextType.SYSTEM_WIDE] or
            self.data.get("goal_id") == goal_id
        )


@dataclass
class Decision:
    """Represents a decision made by the reasoner."""
    decision_id: str
    decision_type: DecisionType
    context_ids: List[str]
    reasoning_type: ReasoningType
    confidence: float
    rationale: str
    decision_data: Dict[str, Any]
    timestamp: datetime
    made_by: str = "reasoner"
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    impact_assessment: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "reasoning_type": self.reasoning_type,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "decision_data": self.decision_data,
            "timestamp": self.timestamp.isoformat(),
            "made_by": self.made_by,
            "alternatives_count": len(self.alternatives_considered),
            "impact_factors": list(self.impact_assessment.keys())
        }


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies."""
    
    @abstractmethod
    def reason(self, context: List[ReasoningContext], goal: str) -> Decision:
        """Perform reasoning and return a decision."""
        pass
    
    @abstractmethod
    def get_confidence(self, context: List[ReasoningContext]) -> float:
        """Calculate confidence level for the reasoning."""
        pass


class DeductiveReasoning(ReasoningStrategy):
    """Deductive reasoning strategy based on logical rules."""
    
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules
    
    def reason(self, context: List[ReasoningContext], goal: str) -> Decision:
        """Apply deductive reasoning using logical rules."""
        # Extract facts from context
        facts = []
        for ctx in context:
            facts.extend(ctx.data.get("facts", []))
        
        # Apply rules to derive conclusions
        conclusions = []
        applied_rules = []
        
        for rule in self.rules:
            if self._can_apply_rule(rule, facts):
                conclusion = rule["conclusion"]
                conclusions.append(conclusion)
                applied_rules.append(rule["name"])
        
        confidence = self.get_confidence(context)
        
        return Decision(
            decision_id=str(uuid.uuid4()),
            decision_type=DecisionType.STRATEGY_SELECTION,
            context_ids=[ctx.context_id for ctx in context],
            reasoning_type=ReasoningType.DEDUCTIVE,
            confidence=confidence,
            rationale=f"Applied rules: {', '.join(applied_rules)}. Derived conclusions: {conclusions}",
            decision_data={"conclusions": conclusions, "applied_rules": applied_rules},
            timestamp=datetime.now()
        )
    
    def _can_apply_rule(self, rule: Dict[str, Any], facts: List[str]) -> bool:
        """Check if a rule can be applied given the facts."""
        premises = rule.get("premises", [])
        return all(premise in facts for premise in premises)
    
    def get_confidence(self, context: List[ReasoningContext]) -> float:
        """Calculate confidence based on context quality and completeness."""
        if not context:
            return 0.0
        
        avg_relevance = sum(ctx.relevance_score for ctx in context) / len(context)
        completeness = min(len(context) / 5.0, 1.0)  # Assume 5 contexts is complete
        freshness = sum(1.0 if not ctx.is_expired() else 0.5 for ctx in context) / len(context)
        
        return (avg_relevance * 0.4 + completeness * 0.3 + freshness * 0.3)


class ProbabilisticReasoning(ReasoningStrategy):
    """Probabilistic reasoning using Bayesian inference."""
    
    def __init__(self):
        self.priors: Dict[str, float] = {}
        self.likelihoods: Dict[str, Dict[str, float]] = {}
    
    def reason(self, context: List[ReasoningContext], goal: str) -> Decision:
        """Apply probabilistic reasoning using Bayesian inference."""
        # Extract evidence from context
        evidence = []
        for ctx in context:
            evidence.extend(ctx.data.get("observations", []))
        
        # Calculate posterior probabilities for different hypotheses
        hypotheses = self._generate_hypotheses(goal, context)
        posteriors = {}
        
        for hypothesis in hypotheses:
            prior = self.priors.get(hypothesis, 0.5)
            likelihood = self._calculate_likelihood(hypothesis, evidence)
            posteriors[hypothesis] = prior * likelihood
        
        # Normalize probabilities
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {h: p / total for h, p in posteriors.items()}
        
        # Select best hypothesis
        best_hypothesis = max(posteriors.keys(), key=lambda h: posteriors[h]) if posteriors else "default"
        confidence = posteriors.get(best_hypothesis, 0.0)
        
        return Decision(
            decision_id=str(uuid.uuid4()),
            decision_type=DecisionType.STRATEGY_SELECTION,
            context_ids=[ctx.context_id for ctx in context],
            reasoning_type=ReasoningType.PROBABILISTIC,
            confidence=confidence,
            rationale=f"Bayesian inference selected '{best_hypothesis}' with probability {confidence:.3f}",
            decision_data={"hypothesis": best_hypothesis, "posteriors": posteriors},
            timestamp=datetime.now()
        )
    
    def _generate_hypotheses(self, goal: str, context: List[ReasoningContext]) -> List[str]:
        """Generate hypotheses based on goal and context."""
        # Default hypotheses based on common patterns
        hypotheses = ["success", "failure", "partial_success"]
        
        # Add context-specific hypotheses
        for ctx in context:
            if "possible_outcomes" in ctx.data:
                hypotheses.extend(ctx.data["possible_outcomes"])
        
        return list(set(hypotheses))
    
    def _calculate_likelihood(self, hypothesis: str, evidence: List[str]) -> float:
        """Calculate likelihood of evidence given hypothesis."""
        if hypothesis not in self.likelihoods:
            return 0.5  # Default likelihood
        
        likelihood = 1.0
        for ev in evidence:
            likelihood *= self.likelihoods[hypothesis].get(ev, 0.5)
        
        return likelihood
    
    def get_confidence(self, context: List[ReasoningContext]) -> float:
        """Calculate confidence based on evidence quality."""
        if not context:
            return 0.0
        
        evidence_count = sum(len(ctx.data.get("observations", [])) for ctx in context)
        evidence_quality = sum(ctx.relevance_score for ctx in context) / len(context)
        
        # Confidence increases with more evidence but plateaus
        evidence_factor = 1 - math.exp(-evidence_count / 10)
        
        return evidence_factor * evidence_quality


class CausalReasoning(ReasoningStrategy):
    """Causal reasoning for understanding cause-effect relationships."""
    
    def __init__(self):
        self.causal_model: Dict[str, List[str]] = {}  # effect -> [causes]
    
    def reason(self, context: List[ReasoningContext], goal: str) -> Decision:
        """Apply causal reasoning to understand cause-effect relationships."""
        # Extract events and their relationships
        events = []
        relationships = []
        
        for ctx in context:
            events.extend(ctx.data.get("events", []))
            relationships.extend(ctx.data.get("causal_links", []))
        
        # Build causal chains
        causal_chains = self._build_causal_chains(events, relationships)
        
        # Identify root causes for current situation
        root_causes = self._identify_root_causes(goal, causal_chains)
        
        # Predict effects of potential actions
        action_effects = self._predict_action_effects(causal_chains)
        
        confidence = self.get_confidence(context)
        
        return Decision(
            decision_id=str(uuid.uuid4()),
            decision_type=DecisionType.STRATEGY_SELECTION,
            context_ids=[ctx.context_id for ctx in context],
            reasoning_type=ReasoningType.CAUSAL,
            confidence=confidence,
            rationale=f"Identified root causes: {root_causes}. Predicted effects of actions.",
            decision_data={
                "root_causes": root_causes,
                "causal_chains": causal_chains,
                "action_effects": action_effects
            },
            timestamp=datetime.now()
        )
    
    def _build_causal_chains(self, events: List[str], relationships: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Build causal chains from events and relationships."""
        chains = defaultdict(list)
        
        for rel in relationships:
            cause = rel.get("cause")
            effect = rel.get("effect")
            if cause and effect:
                chains[effect].append(cause)
        
        return dict(chains)
    
    def _identify_root_causes(self, goal: str, causal_chains: Dict[str, List[str]]) -> List[str]:
        """Identify root causes for a goal or problem."""
        # Simple root cause identification - causes with no predecessors
        all_effects = set(causal_chains.keys())
        all_causes = set()
        for causes in causal_chains.values():
            all_causes.update(causes)
        
        root_causes = all_causes - all_effects
        return list(root_causes)
    
    def _predict_action_effects(self, causal_chains: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Predict effects of potential actions."""
        # Reverse the causal chains to predict forward
        action_effects = {}
        for effect, causes in causal_chains.items():
            for cause in causes:
                if cause not in action_effects:
                    action_effects[cause] = []
                action_effects[cause].append(effect)
        
        return action_effects
    
    def get_confidence(self, context: List[ReasoningContext]) -> float:
        """Calculate confidence based on causal evidence strength."""
        if not context:
            return 0.0
        
        causal_evidence_count = sum(
            len(ctx.data.get("causal_links", [])) for ctx in context
        )
        
        # Confidence based on amount of causal evidence
        return min(causal_evidence_count / 10.0, 1.0)


class KnowledgeBase:
    """Knowledge base for storing and retrieving reasoning knowledge."""
    
    def __init__(self, logger: FeriqLogger):
        self.logger = logger
        self.facts: Set[str] = set()
        self.rules: List[Dict[str, Any]] = []
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.experiences: List[Dict[str, Any]] = []
        self.analogies: Dict[str, List[str]] = defaultdict(list)
    
    def add_fact(self, fact: str):
        """Add a fact to the knowledge base."""
        self.facts.add(fact)
        self.logger.debug(f"Added fact: {fact}")
    
    def add_rule(self, name: str, premises: List[str], conclusion: str, confidence: float = 1.0):
        """Add a reasoning rule."""
        rule = {
            "name": name,
            "premises": premises,
            "conclusion": conclusion,
            "confidence": confidence
        }
        self.rules.append(rule)
        self.logger.debug(f"Added rule: {name}")
    
    def add_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]):
        """Add a pattern for pattern matching."""
        self.patterns[pattern_name] = pattern_data
        self.logger.debug(f"Added pattern: {pattern_name}")
    
    def add_experience(self, situation: str, actions: List[str], outcome: str, success: bool):
        """Add an experience for case-based reasoning."""
        experience = {
            "situation": situation,
            "actions": actions,
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        self.experiences.append(experience)
        self.logger.debug(f"Added experience: {situation} -> {outcome}")
    
    def find_similar_experiences(self, situation: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar experiences for analogical reasoning."""
        similar = []
        for exp in self.experiences:
            similarity = self._calculate_similarity(situation, exp["situation"])
            if similarity >= similarity_threshold:
                exp_copy = exp.copy()
                exp_copy["similarity"] = similarity
                similar.append(exp_copy)
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings (simple implementation)."""
        # Simple word overlap similarity
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def query_knowledge(self, query: str) -> Dict[str, Any]:
        """Query the knowledge base."""
        results = {
            "facts": [fact for fact in self.facts if query.lower() in fact.lower()],
            "rules": [rule for rule in self.rules if query.lower() in rule["name"].lower()],
            "patterns": {name: pattern for name, pattern in self.patterns.items() if query.lower() in name.lower()},
            "experiences": [exp for exp in self.experiences if query.lower() in exp["situation"].lower()]
        }
        return results


class Reasoner:
    """
    Reasoner component that provides intelligent decision-making and problem-solving.
    
    This component provides:
    - Multiple reasoning strategies (deductive, inductive, probabilistic, causal)
    - Context-aware decision making
    - Knowledge base for storing and retrieving reasoning knowledge
    - Strategic planning and optimization
    - Learning from experience and adaptation
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = FeriqLogger("Reasoner", self.config)
        
        # Knowledge and context management
        self.knowledge_base = KnowledgeBase(self.logger)
        self.context_store: Dict[str, ReasoningContext] = {}
        self.decision_history: List[Decision] = []
        
        # Reasoning strategies
        self.strategies: Dict[ReasoningType, ReasoningStrategy] = {
            ReasoningType.DEDUCTIVE: DeductiveReasoning([]),
            ReasoningType.PROBABILISTIC: ProbabilisticReasoning(),
            ReasoningType.CAUSAL: CausalReasoning()
        }
        
        # Strategy selection rules
        self.strategy_rules: Dict[str, ReasoningType] = {
            "logical_problem": ReasoningType.DEDUCTIVE,
            "uncertain_outcome": ReasoningType.PROBABILISTIC,
            "cause_analysis": ReasoningType.CAUSAL,
            "pattern_matching": ReasoningType.ANALOGICAL
        }
        
        # Performance tracking
        self.reasoning_metrics: Dict[str, Any] = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "strategy_usage": defaultdict(int),
            "average_confidence": 0.0,
            "reasoning_time": 0.0
        }
        
        # Initialize default knowledge
        self._initialize_default_knowledge()
        
        self.logger.info("Reasoner initialized successfully")
    
    def _initialize_default_knowledge(self):
        """Initialize default knowledge base."""
        # Basic facts
        self.knowledge_base.add_fact("agents_can_collaborate")
        self.knowledge_base.add_fact("tasks_have_dependencies")
        self.knowledge_base.add_fact("resources_are_limited")
        
        # Basic rules
        self.knowledge_base.add_rule(
            "collaboration_rule",
            ["task_complexity_high", "multiple_agents_available"],
            "use_collaborative_approach",
            0.9
        )
        
        self.knowledge_base.add_rule(
            "resource_conservation",
            ["resources_low", "task_priority_medium"],
            "defer_task_execution",
            0.8
        )
        
        # Basic patterns
        self.knowledge_base.add_pattern("sequential_tasks", {
            "pattern_type": "temporal",
            "description": "Tasks that must be executed in order",
            "triggers": ["has_dependencies", "order_matters"],
            "strategy": "pipeline_coordination"
        })
    
    def add_context(self, context: ReasoningContext):
        """Add context information for reasoning."""
        self.context_store[context.context_id] = context
        self.logger.debug(f"Added reasoning context: {context.context_id}")
    
    def reason_about_goal(self, goal: Goal, reasoning_type: Optional[ReasoningType] = None) -> Decision:
        """
        Reason about how to achieve a goal.
        
        Args:
            goal: The goal to reason about
            reasoning_type: Specific reasoning type to use (optional)
            
        Returns:
            Decision with reasoning results
        """
        start_time = datetime.now()
        
        # Gather relevant context
        relevant_context = self._gather_relevant_context(goal)
        
        # Select reasoning strategy
        if reasoning_type is None:
            reasoning_type = self._select_reasoning_strategy(goal, relevant_context)
        
        strategy = self.strategies.get(reasoning_type)
        if not strategy:
            raise ValueError(f"Unknown reasoning type: {reasoning_type}")
        
        # Perform reasoning
        decision = strategy.reason(relevant_context, goal.goal_id)
        decision.decision_type = DecisionType.STRATEGY_SELECTION
        
        # Update metrics
        self._update_reasoning_metrics(decision, start_time)
        
        # Store decision
        self.decision_history.append(decision)
        
        self.logger.info(
            "Goal reasoning completed",
            goal_id=goal.goal_id,
            reasoning_type=reasoning_type,
            confidence=decision.confidence,
            decision_id=decision.decision_id
        )
        
        return decision
    
    def reason_about_task_assignment(
        self,
        task: FeriqTask,
        available_agents: List[FeriqAgent]
    ) -> Decision:
        """Reason about the best agent assignment for a task."""
        # Create context for task assignment
        context = ReasoningContext(
            context_id=str(uuid.uuid4()),
            context_type=ContextType.TASK_SPECIFIC,
            timestamp=datetime.now(),
            data={
                "task_id": task.task_id,
                "task_complexity": task.complexity_score,
                "required_capabilities": task.required_capabilities,
                "available_agents": [
                    {
                        "agent_id": agent.agent_id,
                        "capabilities": agent.capabilities,
                        "performance": agent.performance_metrics,
                        "status": agent.status.value
                    }
                    for agent in available_agents
                ]
            }
        )
        
        # Use probabilistic reasoning for assignment
        strategy = self.strategies[ReasoningType.PROBABILISTIC]
        decision = strategy.reason([context], task.task_id)
        decision.decision_type = DecisionType.TASK_ASSIGNMENT
        
        # Calculate assignment scores
        assignment_scores = {}
        for agent in available_agents:
            score = self._calculate_assignment_score(task, agent)
            assignment_scores[agent.agent_id] = score
        
        # Select best agent
        best_agent_id = max(assignment_scores.keys(), key=lambda aid: assignment_scores[aid])
        
        decision.decision_data.update({
            "recommended_agent": best_agent_id,
            "assignment_scores": assignment_scores,
            "reasoning": f"Selected agent {best_agent_id} with score {assignment_scores[best_agent_id]:.3f}"
        })
        
        self.decision_history.append(decision)
        
        self.logger.info(
            "Task assignment reasoning completed",
            task_id=task.task_id,
            recommended_agent=best_agent_id,
            confidence=decision.confidence
        )
        
        return decision
    
    def reason_about_plan_optimization(self, plan: Plan) -> Decision:
        """Reason about how to optimize a plan."""
        # Analyze plan performance
        context = ReasoningContext(
            context_id=str(uuid.uuid4()),
            context_type=ContextType.SYSTEM_WIDE,
            timestamp=datetime.now(),
            data={
                "plan_id": plan.plan_id,
                "total_tasks": len(plan.tasks),
                "completed_tasks": sum(1 for task in plan.tasks if task.status == TaskStatus.COMPLETED),
                "failed_tasks": sum(1 for task in plan.tasks if task.status == TaskStatus.FAILED),
                "bottlenecks": self._identify_bottlenecks(plan),
                "resource_utilization": self._analyze_resource_utilization(plan)
            }
        )
        
        # Use causal reasoning for optimization
        strategy = self.strategies[ReasoningType.CAUSAL]
        decision = strategy.reason([context], plan.plan_id)
        decision.decision_type = DecisionType.OPTIMIZATION
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(plan, context)
        
        decision.decision_data.update({
            "recommendations": recommendations,
            "priority_issues": self._identify_priority_issues(plan),
            "expected_improvement": self._estimate_improvement_potential(plan, recommendations)
        })
        
        self.decision_history.append(decision)
        
        self.logger.info(
            "Plan optimization reasoning completed",
            plan_id=plan.plan_id,
            recommendations_count=len(recommendations),
            confidence=decision.confidence
        )
        
        return decision
    
    def _gather_relevant_context(self, goal: Goal) -> List[ReasoningContext]:
        """Gather context relevant to a goal."""
        relevant = []
        
        for context in self.context_store.values():
            if not context.is_expired() and context.is_relevant_for_goal(goal.goal_id):
                relevant.append(context)
        
        # Sort by relevance and recency
        relevant.sort(key=lambda c: (c.relevance_score, c.timestamp), reverse=True)
        
        return relevant[:10]  # Limit to top 10 most relevant contexts
    
    def _select_reasoning_strategy(self, goal: Goal, context: List[ReasoningContext]) -> ReasoningType:
        """Select the most appropriate reasoning strategy."""
        # Analyze goal and context characteristics
        if goal.goal_type == GoalType.RESEARCH:
            return ReasoningType.INDUCTIVE
        elif goal.goal_type == GoalType.ANALYSIS:
            return ReasoningType.DEDUCTIVE
        elif any("uncertainty" in ctx.data for ctx in context):
            return ReasoningType.PROBABILISTIC
        elif any("cause" in ctx.data or "effect" in ctx.data for ctx in context):
            return ReasoningType.CAUSAL
        else:
            return ReasoningType.DEDUCTIVE  # Default
    
    def _calculate_assignment_score(self, task: FeriqTask, agent: FeriqAgent) -> float:
        """Calculate assignment score for a task-agent pair."""
        score = 0.0
        
        # Capability match
        agent_caps = set(agent.capabilities)
        task_caps = set(task.required_capabilities)
        capability_overlap = len(agent_caps.intersection(task_caps))
        capability_score = capability_overlap / len(task_caps) if task_caps else 1.0
        score += capability_score * 0.4
        
        # Performance history
        success_rate = agent.performance_metrics.get("success_rate", 0.5)
        score += success_rate * 0.3
        
        # Agent availability
        if agent.status == AgentStatus.IDLE:
            score += 0.2
        elif agent.status == AgentStatus.BUSY:
            score += 0.05
        
        # Complexity match
        agent_exp_level = agent.performance_metrics.get("experience_level", 0.5)
        complexity_match = 1 - abs(agent_exp_level - task.complexity_score)
        score += complexity_match * 0.1
        
        return min(score, 1.0)
    
    def _identify_bottlenecks(self, plan: Plan) -> List[str]:
        """Identify bottlenecks in a plan."""
        bottlenecks = []
        
        # Simple bottleneck identification
        for task in plan.tasks:
            if task.status == TaskStatus.IN_PROGRESS:
                # Check how many tasks depend on this one
                dependent_count = sum(1 for t in plan.tasks if task.task_id in t.dependencies)
                if dependent_count > 2:
                    bottlenecks.append(task.task_id)
        
        return bottlenecks
    
    def _analyze_resource_utilization(self, plan: Plan) -> Dict[str, float]:
        """Analyze resource utilization in a plan."""
        resource_usage = defaultdict(float)
        total_resources = defaultdict(float)
        
        for task in plan.tasks:
            for resource, amount in task.resource_requirements.items():
                if task.status in [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]:
                    resource_usage[resource] += amount
                total_resources[resource] += amount
        
        utilization = {}
        for resource in total_resources:
            if total_resources[resource] > 0:
                utilization[resource] = resource_usage[resource] / total_resources[resource]
        
        return utilization
    
    def _generate_optimization_recommendations(
        self,
        plan: Plan,
        context: ReasoningContext
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations for a plan."""
        recommendations = []
        
        # Analyze bottlenecks
        bottlenecks = context.data.get("bottlenecks", [])
        if bottlenecks:
            recommendations.append({
                "type": "bottleneck_resolution",
                "description": f"Address bottlenecks in tasks: {', '.join(bottlenecks)}",
                "priority": "high",
                "actions": ["allocate_additional_resources", "parallelize_dependencies"]
            })
        
        # Analyze resource utilization
        resource_util = context.data.get("resource_utilization", {})
        for resource, utilization in resource_util.items():
            if utilization < 0.3:
                recommendations.append({
                    "type": "resource_optimization",
                    "description": f"Underutilized resource: {resource} ({utilization:.1%})",
                    "priority": "medium",
                    "actions": ["reallocate_resources", "reduce_resource_allocation"]
                })
            elif utilization > 0.9:
                recommendations.append({
                    "type": "resource_scaling",
                    "description": f"Overutilized resource: {resource} ({utilization:.1%})",
                    "priority": "high",
                    "actions": ["scale_up_resources", "load_balancing"]
                })
        
        # Task failure analysis
        failed_tasks = context.data.get("failed_tasks", 0)
        total_tasks = context.data.get("total_tasks", 1)
        if failed_tasks / total_tasks > 0.1:
            recommendations.append({
                "type": "failure_reduction",
                "description": f"High failure rate: {failed_tasks}/{total_tasks} tasks failed",
                "priority": "high",
                "actions": ["improve_error_handling", "review_task_complexity", "agent_training"]
            })
        
        return recommendations
    
    def _identify_priority_issues(self, plan: Plan) -> List[str]:
        """Identify priority issues in a plan."""
        issues = []
        
        # Check for failed critical tasks
        for task in plan.tasks:
            if task.status == TaskStatus.FAILED and task.priority.value >= 3:  # HIGH or URGENT
                issues.append(f"Critical task failed: {task.task_id}")
        
        # Check for overdue tasks
        current_time = datetime.now()
        for task in plan.tasks:
            if (task.status == TaskStatus.IN_PROGRESS and 
                task.started_at and 
                current_time - task.started_at > task.estimated_duration * 1.5):
                issues.append(f"Task overdue: {task.task_id}")
        
        return issues
    
    def _estimate_improvement_potential(
        self,
        plan: Plan,
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Estimate potential improvement from recommendations."""
        improvement = {
            "efficiency_gain": 0.0,
            "time_reduction": 0.0,
            "resource_savings": 0.0,
            "success_rate_improvement": 0.0
        }
        
        # Simple estimation based on recommendation types
        for rec in recommendations:
            rec_type = rec["type"]
            priority = rec["priority"]
            
            impact_multiplier = {"high": 0.2, "medium": 0.1, "low": 0.05}.get(priority, 0.05)
            
            if rec_type == "bottleneck_resolution":
                improvement["time_reduction"] += impact_multiplier
                improvement["efficiency_gain"] += impact_multiplier * 0.8
            elif rec_type == "resource_optimization":
                improvement["resource_savings"] += impact_multiplier
                improvement["efficiency_gain"] += impact_multiplier * 0.6
            elif rec_type == "failure_reduction":
                improvement["success_rate_improvement"] += impact_multiplier
        
        return improvement
    
    def _update_reasoning_metrics(self, decision: Decision, start_time: datetime):
        """Update reasoning performance metrics."""
        self.reasoning_metrics["total_decisions"] += 1
        
        # Update strategy usage
        self.reasoning_metrics["strategy_usage"][decision.reasoning_type.value] += 1
        
        # Update average confidence
        total = self.reasoning_metrics["total_decisions"]
        current_avg = self.reasoning_metrics["average_confidence"]
        new_avg = (current_avg * (total - 1) + decision.confidence) / total
        self.reasoning_metrics["average_confidence"] = new_avg
        
        # Update reasoning time
        reasoning_time = (datetime.now() - start_time).total_seconds()
        current_time = self.reasoning_metrics["reasoning_time"]
        new_time = (current_time * (total - 1) + reasoning_time) / total
        self.reasoning_metrics["reasoning_time"] = new_time
    
    def learn_from_outcome(self, decision_id: str, outcome: str, success: bool):
        """Learn from the outcome of a decision."""
        # Find the decision
        decision = None
        for d in self.decision_history:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if not decision:
            self.logger.warning(f"Decision {decision_id} not found for learning")
            return
        
        # Update success metrics
        if success:
            self.reasoning_metrics["successful_decisions"] += 1
        
        # Add experience to knowledge base
        situation = f"{decision.decision_type.value}_{decision.reasoning_type.value}"
        actions = list(decision.decision_data.keys())
        
        self.knowledge_base.add_experience(situation, actions, outcome, success)
        
        # Update strategy performance (simple learning)
        strategy = self.strategies.get(decision.reasoning_type)
        if hasattr(strategy, 'learn_from_outcome'):
            strategy.learn_from_outcome(decision, outcome, success)
        
        self.logger.info(
            "Learned from decision outcome",
            decision_id=decision_id,
            outcome=outcome,
            success=success
        )
    
    def explain_decision(self, decision_id: str) -> Dict[str, Any]:
        """Provide an explanation for a decision."""
        decision = None
        for d in self.decision_history:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if not decision:
            return {"error": "Decision not found"}
        
        explanation = {
            "decision_id": decision_id,
            "decision_type": decision.decision_type.value,
            "reasoning_type": decision.reasoning_type.value,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "context_used": len(decision.context_ids),
            "alternatives_considered": len(decision.alternatives_considered),
            "key_factors": list(decision.impact_assessment.keys()),
            "decision_data": decision.decision_data
        }
        
        return explanation
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reasoning statistics."""
        success_rate = (
            self.reasoning_metrics["successful_decisions"] / 
            max(self.reasoning_metrics["total_decisions"], 1)
        )
        
        return {
            "total_decisions": self.reasoning_metrics["total_decisions"],
            "success_rate": success_rate,
            "average_confidence": self.reasoning_metrics["average_confidence"],
            "average_reasoning_time": self.reasoning_metrics["reasoning_time"],
            "strategy_usage": dict(self.reasoning_metrics["strategy_usage"]),
            "knowledge_base_size": {
                "facts": len(self.knowledge_base.facts),
                "rules": len(self.knowledge_base.rules),
                "patterns": len(self.knowledge_base.patterns),
                "experiences": len(self.knowledge_base.experiences)
            },
            "active_contexts": len(self.context_store),
            "decision_history_size": len(self.decision_history)
        }