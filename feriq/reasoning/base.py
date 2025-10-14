"""
Base classes for the Feriq reasoning system.

This module provides the foundational classes and interfaces for all reasoning types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Type
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import uuid
from datetime import datetime


class ReasoningType(Enum):
    """Types of reasoning supported by the framework."""
    INDUCTIVE = "inductive"
    DEDUCTIVE = "deductive"
    PROBABILISTIC = "probabilistic"
    CAUSAL = "causal"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    HYBRID = "hybrid"
    COLLABORATIVE = "collaborative"


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning results."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


@dataclass
class Evidence:
    """Represents a piece of evidence used in reasoning."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    source: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class Hypothesis:
    """Represents a hypothesis in reasoning."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    probability: float = 0.5
    evidence_for: List[Evidence] = field(default_factory=list)
    evidence_against: List[Evidence] = field(default_factory=list)
    prior_probability: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        if not 0.0 <= self.prior_probability <= 1.0:
            raise ValueError("Prior probability must be between 0.0 and 1.0")
    
    def update_probability(self, new_evidence: Evidence, method: str = "bayesian"):
        """Update hypothesis probability based on new evidence."""
        if method == "bayesian":
            # Simplified Bayesian update
            likelihood = new_evidence.confidence
            self.probability = (likelihood * self.prior_probability) / (
                likelihood * self.prior_probability + 
                (1 - likelihood) * (1 - self.prior_probability)
            )


@dataclass
class Conclusion:
    """Represents a reasoning conclusion."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    confidence: float = 0.5
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    supporting_evidence: List[Evidence] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    alternatives: List['Conclusion'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class ReasoningContext:
    """Context for reasoning operations."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    task_id: str = ""
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    available_evidence: List[Evidence] = field(default_factory=list)
    prior_knowledge: Dict[str, Any] = field(default_factory=dict)
    reasoning_history: List['ReasoningResult'] = field(default_factory=list)
    collaborative_agents: List[str] = field(default_factory=list)
    timeout: float = 60.0
    max_iterations: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = ""
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    conclusions: List[Conclusion] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    confidence: float = 0.5
    execution_time: float = 0.0
    iterations: int = 0
    success: bool = True
    error_message: str = ""
    reasoning_trace: List[str] = field(default_factory=list)
    used_evidence: List[Evidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_best_conclusion(self) -> Optional[Conclusion]:
        """Get the conclusion with highest confidence."""
        if not self.conclusions:
            return None
        return max(self.conclusions, key=lambda c: c.confidence)
    
    def get_best_hypothesis(self) -> Optional[Hypothesis]:
        """Get the hypothesis with highest probability."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.probability)


class BaseReasoner(ABC):
    """Base class for all reasoning engines."""
    
    def __init__(self, 
                 name: str,
                 reasoning_type: ReasoningType,
                 config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.reasoning_type = reasoning_type
        self.config = config or {}
        self.is_initialized = False
        self.performance_metrics = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'average_confidence': 0.0,
            'average_execution_time': 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the reasoner."""
        self.is_initialized = True
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @abstractmethod
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Main reasoning method to be implemented by subclasses.
        
        Args:
            context: The reasoning context containing goal, evidence, etc.
            
        Returns:
            ReasoningResult containing conclusions and metadata
        """
        pass
    
    async def validate_input(self, context: ReasoningContext) -> bool:
        """Validate input context for reasoning."""
        if not isinstance(context, ReasoningContext):
            return False
        if not context.goal:
            return False
        return True
    
    async def preprocess_evidence(self, evidence: List[Evidence]) -> List[Evidence]:
        """Preprocess evidence before reasoning."""
        # Filter out low-confidence evidence
        min_confidence = self.config.get('min_evidence_confidence', 0.1)
        return [e for e in evidence if e.confidence >= min_confidence]
    
    async def postprocess_result(self, result: ReasoningResult) -> ReasoningResult:
        """Postprocess reasoning result."""
        # Update performance metrics
        self.performance_metrics['total_inferences'] += 1
        if result.success:
            self.performance_metrics['successful_inferences'] += 1
        
        # Update average metrics
        total = self.performance_metrics['total_inferences']
        self.performance_metrics['average_confidence'] = (
            (self.performance_metrics['average_confidence'] * (total - 1) + result.confidence) / total
        )
        self.performance_metrics['average_execution_time'] = (
            (self.performance_metrics['average_execution_time'] * (total - 1) + result.execution_time) / total
        )
        
        return result
    
    async def reason_with_timeout(self, 
                                 context: ReasoningContext,
                                 timeout: Optional[float] = None) -> ReasoningResult:
        """Reason with timeout protection."""
        if not self.is_initialized:
            await self.initialize()
        
        timeout = timeout or context.timeout
        start_time = time.time()
        
        try:
            # Validate input
            if not await self.validate_input(context):
                return ReasoningResult(
                    context_id=context.id,
                    reasoning_type=self.reasoning_type,
                    success=False,
                    error_message="Invalid input context",
                    execution_time=time.time() - start_time
                )
            
            # Preprocess evidence
            processed_evidence = await self.preprocess_evidence(context.available_evidence)
            context.available_evidence = processed_evidence
            
            # Perform reasoning with timeout
            result = await asyncio.wait_for(
                self.reason(context),
                timeout=timeout
            )
            
            result.execution_time = time.time() - start_time
            result.context_id = context.id
            
            # Postprocess result
            result = await self.postprocess_result(result)
            
            return result
            
        except asyncio.TimeoutError:
            return ReasoningResult(
                context_id=context.id,
                reasoning_type=self.reasoning_type,
                success=False,
                error_message=f"Reasoning timed out after {timeout} seconds",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ReasoningResult(
                context_id=context.id,
                reasoning_type=self.reasoning_type,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this reasoner."""
        return self.performance_metrics.copy()
    
    def get_success_rate(self) -> float:
        """Get success rate of reasoning operations."""
        total = self.performance_metrics['total_inferences']
        if total == 0:
            return 0.0
        return self.performance_metrics['successful_inferences'] / total
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'average_confidence': 0.0,
            'average_execution_time': 0.0
        }


class CompositeReasoner(BaseReasoner):
    """A reasoner that combines multiple reasoning approaches."""
    
    def __init__(self, 
                 name: str,
                 reasoners: List[BaseReasoner],
                 combination_strategy: str = "weighted_average",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.HYBRID, config)
        self.reasoners = reasoners
        self.combination_strategy = combination_strategy
        self.weights = config.get('weights', [1.0] * len(reasoners))
    
    async def initialize(self) -> None:
        """Initialize all component reasoners."""
        await super().initialize()
        for reasoner in self.reasoners:
            await reasoner.initialize()
    
    async def cleanup(self) -> None:
        """Clean up all component reasoners."""
        for reasoner in self.reasoners:
            await reasoner.cleanup()
        await super().cleanup()
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Combine results from multiple reasoners."""
        results = []
        
        # Run all reasoners
        for reasoner in self.reasoners:
            try:
                result = await reasoner.reason_with_timeout(context)
                results.append(result)
            except Exception as e:
                # Continue with other reasoners if one fails
                continue
        
        if not results:
            return ReasoningResult(
                reasoning_type=self.reasoning_type,
                success=False,
                error_message="All component reasoners failed"
            )
        
        # Combine results based on strategy
        return await self.combine_results(results, context)
    
    async def combine_results(self, 
                             results: List[ReasoningResult],
                             context: ReasoningContext) -> ReasoningResult:
        """Combine multiple reasoning results."""
        if self.combination_strategy == "weighted_average":
            return await self.weighted_average_combination(results)
        elif self.combination_strategy == "consensus":
            return await self.consensus_combination(results)
        elif self.combination_strategy == "best_confidence":
            return await self.best_confidence_combination(results)
        else:
            # Default to weighted average
            return await self.weighted_average_combination(results)
    
    async def weighted_average_combination(self, results: List[ReasoningResult]) -> ReasoningResult:
        """Combine results using weighted average."""
        if not results:
            return ReasoningResult(
                reasoning_type=self.reasoning_type,
                success=False,
                error_message="No results to combine"
            )
        
        combined_result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        # Combine conclusions
        all_conclusions = []
        for i, result in enumerate(results):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            for conclusion in result.conclusions:
                # Weight the confidence
                weighted_conclusion = Conclusion(
                    statement=conclusion.statement,
                    confidence=conclusion.confidence * weight,
                    reasoning_type=conclusion.reasoning_type,
                    supporting_evidence=conclusion.supporting_evidence,
                    reasoning_chain=conclusion.reasoning_chain
                )
                all_conclusions.append(weighted_conclusion)
        
        combined_result.conclusions = all_conclusions
        
        # Calculate overall confidence
        if results:
            weighted_confidences = [
                result.confidence * self.weights[i] if i < len(self.weights) else result.confidence
                for i, result in enumerate(results)
            ]
            combined_result.confidence = sum(weighted_confidences) / len(weighted_confidences)
        
        combined_result.success = any(result.success for result in results)
        
        return combined_result
    
    async def consensus_combination(self, results: List[ReasoningResult]) -> ReasoningResult:
        """Combine results based on consensus."""
        # Group similar conclusions
        conclusion_groups = {}
        
        for result in results:
            for conclusion in result.conclusions:
                key = conclusion.statement.lower().strip()
                if key not in conclusion_groups:
                    conclusion_groups[key] = []
                conclusion_groups[key].append(conclusion)
        
        # Create consensus conclusions
        consensus_conclusions = []
        for statement, conclusions in conclusion_groups.items():
            if len(conclusions) >= len(results) / 2:  # Majority consensus
                avg_confidence = sum(c.confidence for c in conclusions) / len(conclusions)
                consensus_conclusion = Conclusion(
                    statement=conclusions[0].statement,
                    confidence=avg_confidence,
                    reasoning_type=ReasoningType.COLLABORATIVE
                )
                consensus_conclusions.append(consensus_conclusion)
        
        return ReasoningResult(
            reasoning_type=self.reasoning_type,
            conclusions=consensus_conclusions,
            confidence=sum(c.confidence for c in consensus_conclusions) / len(consensus_conclusions) if consensus_conclusions else 0.0,
            success=len(consensus_conclusions) > 0
        )
    
    async def best_confidence_combination(self, results: List[ReasoningResult]) -> ReasoningResult:
        """Select result with highest confidence."""
        if not results:
            return ReasoningResult(
                reasoning_type=self.reasoning_type,
                success=False,
                error_message="No results to combine"
            )
        
        best_result = max(results, key=lambda r: r.confidence)
        best_result.reasoning_type = self.reasoning_type
        return best_result


# Utility functions for evidence and hypothesis management

def create_evidence(content: Any, 
                   source: str = "",
                   confidence: float = 1.0,
                   metadata: Optional[Dict[str, Any]] = None) -> Evidence:
    """Create evidence with validation."""
    return Evidence(
        content=content,
        source=source,
        confidence=confidence,
        metadata=metadata or {}
    )


def create_hypothesis(statement: str,
                     probability: float = 0.5,
                     prior_probability: float = 0.5,
                     metadata: Optional[Dict[str, Any]] = None) -> Hypothesis:
    """Create hypothesis with validation."""
    return Hypothesis(
        statement=statement,
        probability=probability,
        prior_probability=prior_probability,
        metadata=metadata or {}
    )


def create_conclusion(statement: str,
                     confidence: float = 0.5,
                     reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
                     evidence: Optional[List[Evidence]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Conclusion:
    """Create conclusion with validation."""
    return Conclusion(
        statement=statement,
        confidence=confidence,
        reasoning_type=reasoning_type,
        supporting_evidence=evidence or [],
        metadata=metadata or {}
    )