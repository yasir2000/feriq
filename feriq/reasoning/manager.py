"""
Reasoning Manager and Orchestrator

Coordinates and manages different reasoning types, provides unified reasoning interface,
and orchestrates complex reasoning workflows.
"""

from typing import Any, Dict, List, Optional, Union, Type, Callable
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from .base import BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType, Evidence, Hypothesis, Conclusion
from .inductive import InductiveReasoner
from .deductive import DeductiveReasoner
from .probabilistic import ProbabilisticReasoner
from .causal import CausalReasoner
from .abductive import AbductiveReasoner
from .analogical import AnalogicalReasoner
from .temporal import TemporalReasoner
from .spatial import SpatialReasoner
from .hybrid import HybridReasoner
from .collaborative import CollaborativeReasoner


class ReasoningStrategy(Enum):
    """Strategies for orchestrating multiple reasoning types."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    PIPELINE = "pipeline"


@dataclass
class ReasoningPlan:
    """Represents a plan for orchestrating reasoning."""
    strategy: ReasoningStrategy
    reasoning_types: List[ReasoningType]
    dependencies: Dict[ReasoningType, List[ReasoningType]] = field(default_factory=dict)
    conditions: Dict[ReasoningType, Callable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningSession:
    """Tracks an active reasoning session."""
    session_id: str
    context: ReasoningContext
    plan: ReasoningPlan
    results: Dict[ReasoningType, ReasoningResult] = field(default_factory=dict)
    status: str = "active"
    start_time: float = 0.0
    end_time: Optional[float] = None


class ReasoningManager:
    """Manages different types of reasoners and their coordination."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.reasoners: Dict[ReasoningType, BaseReasoner] = {}
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self._initialize_reasoners()
    
    def _initialize_reasoners(self):
        """Initialize all available reasoners."""
        reasoner_classes = {
            ReasoningType.INDUCTIVE: InductiveReasoner,
            ReasoningType.DEDUCTIVE: DeductiveReasoner,
            ReasoningType.PROBABILISTIC: ProbabilisticReasoner,
            ReasoningType.CAUSAL: CausalReasoner,
            ReasoningType.ABDUCTIVE: AbductiveReasoner,
            ReasoningType.ANALOGICAL: AnalogicalReasoner,
            ReasoningType.TEMPORAL: TemporalReasoner,
            ReasoningType.SPATIAL: SpatialReasoner,
            ReasoningType.HYBRID: HybridReasoner,
            ReasoningType.COLLABORATIVE: CollaborativeReasoner
        }
        
        for reasoning_type, reasoner_class in reasoner_classes.items():
            try:
                self.reasoners[reasoning_type] = reasoner_class(config=self.config)
            except Exception as e:
                print(f"Failed to initialize {reasoning_type.value} reasoner: {e}")
    
    def get_reasoner(self, reasoning_type: ReasoningType) -> Optional[BaseReasoner]:
        """Get a specific reasoner by type."""
        return self.reasoners.get(reasoning_type)
    
    def list_available_reasoners(self) -> List[ReasoningType]:
        """List all available reasoning types."""
        return list(self.reasoners.keys())
    
    async def reason_single(self, reasoning_type: ReasoningType, context: ReasoningContext) -> ReasoningResult:
        """Perform reasoning using a single reasoner."""
        reasoner = self.get_reasoner(reasoning_type)
        if not reasoner:
            result = ReasoningResult(reasoning_type=reasoning_type)
            result.success = False
            result.error_message = f"Reasoner not available: {reasoning_type.value}"
            return result
        
        return await reasoner.reason(context)
    
    async def reason_multiple(self, reasoning_types: List[ReasoningType], context: ReasoningContext, 
                            strategy: ReasoningStrategy = ReasoningStrategy.PARALLEL) -> Dict[ReasoningType, ReasoningResult]:
        """Perform reasoning using multiple reasoners."""
        if strategy == ReasoningStrategy.PARALLEL:
            return await self._reason_parallel(reasoning_types, context)
        elif strategy == ReasoningStrategy.SEQUENTIAL:
            return await self._reason_sequential(reasoning_types, context)
        else:
            # Default to parallel for other strategies
            return await self._reason_parallel(reasoning_types, context)
    
    async def _reason_parallel(self, reasoning_types: List[ReasoningType], context: ReasoningContext) -> Dict[ReasoningType, ReasoningResult]:
        """Run multiple reasoners in parallel."""
        tasks = []
        valid_types = []
        
        for reasoning_type in reasoning_types:
            reasoner = self.get_reasoner(reasoning_type)
            if reasoner:
                tasks.append(reasoner.reason(context))
                valid_types.append(reasoning_type)
        
        if not tasks:
            return {}
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        reasoning_results = {}
        for i, result in enumerate(results):
            reasoning_type = valid_types[i]
            if isinstance(result, Exception):
                error_result = ReasoningResult(reasoning_type=reasoning_type)
                error_result.success = False
                error_result.error_message = str(result)
                reasoning_results[reasoning_type] = error_result
            else:
                reasoning_results[reasoning_type] = result
        
        return reasoning_results
    
    async def _reason_sequential(self, reasoning_types: List[ReasoningType], context: ReasoningContext) -> Dict[ReasoningType, ReasoningResult]:
        """Run multiple reasoners sequentially."""
        results = {}
        current_context = context
        
        for reasoning_type in reasoning_types:
            reasoner = self.get_reasoner(reasoning_type)
            if reasoner:
                result = await reasoner.reason(current_context)
                results[reasoning_type] = result
                
                # Update context with previous results for sequential processing
                if result.success and result.conclusions:
                    current_context.metadata[f'{reasoning_type.value}_result'] = result
        
        return results


class ReasoningOrchestrator:
    """Orchestrates complex reasoning workflows with dependencies and conditions."""
    
    def __init__(self, manager: ReasoningManager, config: Optional[Dict[str, Any]] = None):
        self.manager = manager
        self.config = config or {}
    
    async def execute_plan(self, plan: ReasoningPlan, context: ReasoningContext) -> Dict[ReasoningType, ReasoningResult]:
        """Execute a reasoning plan."""
        if plan.strategy == ReasoningStrategy.SEQUENTIAL:
            return await self._execute_sequential(plan, context)
        elif plan.strategy == ReasoningStrategy.PARALLEL:
            return await self._execute_parallel(plan, context)
        elif plan.strategy == ReasoningStrategy.HIERARCHICAL:
            return await self._execute_hierarchical(plan, context)
        elif plan.strategy == ReasoningStrategy.ADAPTIVE:
            return await self._execute_adaptive(plan, context)
        elif plan.strategy == ReasoningStrategy.PIPELINE:
            return await self._execute_pipeline(plan, context)
        else:
            raise ValueError(f"Unknown reasoning strategy: {plan.strategy}")
    
    async def _execute_sequential(self, plan: ReasoningPlan, context: ReasoningContext) -> Dict[ReasoningType, ReasoningResult]:
        """Execute reasoning types sequentially."""
        return await self.manager._reason_sequential(plan.reasoning_types, context)
    
    async def _execute_parallel(self, plan: ReasoningPlan, context: ReasoningContext) -> Dict[ReasoningType, ReasoningResult]:
        """Execute reasoning types in parallel."""
        return await self.manager._reason_parallel(plan.reasoning_types, context)
    
    async def _execute_hierarchical(self, plan: ReasoningPlan, context: ReasoningContext) -> Dict[ReasoningType, ReasoningResult]:
        """Execute reasoning types in hierarchical order based on dependencies."""
        results = {}
        remaining_types = set(plan.reasoning_types)
        completed_types = set()
        
        while remaining_types:
            # Find types with satisfied dependencies
            ready_types = []
            for reasoning_type in remaining_types:
                dependencies = plan.dependencies.get(reasoning_type, [])
                if all(dep in completed_types for dep in dependencies):
                    ready_types.append(reasoning_type)
            
            if not ready_types:
                # Circular dependency or unsatisfiable dependencies
                break
            
            # Execute ready types in parallel
            ready_results = await self.manager._reason_parallel(ready_types, context)
            results.update(ready_results)
            
            # Update completed types
            for reasoning_type in ready_types:
                if ready_results[reasoning_type].success:
                    completed_types.add(reasoning_type)
                remaining_types.discard(reasoning_type)
        
        return results
    
    async def _execute_adaptive(self, plan: ReasoningPlan, context: ReasoningContext) -> Dict[ReasoningType, ReasoningResult]:
        """Execute reasoning types adaptively based on conditions and results."""
        results = {}
        remaining_types = list(plan.reasoning_types)
        
        while remaining_types:
            # Select next reasoning type based on conditions
            next_type = await self._select_next_adaptive(remaining_types, plan, results, context)
            if not next_type:
                break
            
            # Execute selected reasoning type
            result = await self.manager.reason_single(next_type, context)
            results[next_type] = result
            remaining_types.remove(next_type)
            
            # Update context with new result
            if result.success:
                context.metadata[f'{next_type.value}_result'] = result
        
        return results
    
    async def _execute_pipeline(self, plan: ReasoningPlan, context: ReasoningContext) -> Dict[ReasoningType, ReasoningResult]:
        """Execute reasoning types in a pipeline where output feeds into next stage."""
        results = {}
        current_context = context
        
        for reasoning_type in plan.reasoning_types:
            result = await self.manager.reason_single(reasoning_type, current_context)
            results[reasoning_type] = result
            
            if result.success:
                # Create new context for next stage using current results
                new_evidence = []
                for conclusion in result.conclusions:
                    evidence = Evidence(
                        content=conclusion.statement,
                        source=f"{reasoning_type.value}_reasoner",
                        confidence=conclusion.confidence
                    )
                    new_evidence.append(evidence)
                
                current_context = ReasoningContext(
                    problem=current_context.problem,
                    evidence=current_context.evidence + new_evidence,
                    hypotheses=current_context.hypotheses,
                    metadata={**current_context.metadata, f'{reasoning_type.value}_result': result}
                )
        
        return results
    
    async def _select_next_adaptive(self, remaining_types: List[ReasoningType], plan: ReasoningPlan, 
                                  current_results: Dict[ReasoningType, ReasoningResult], 
                                  context: ReasoningContext) -> Optional[ReasoningType]:
        """Select next reasoning type for adaptive execution."""
        # Simple adaptive selection based on conditions
        for reasoning_type in remaining_types:
            condition = plan.conditions.get(reasoning_type)
            if condition is None or await self._evaluate_condition(condition, current_results, context):
                return reasoning_type
        
        # If no conditions matched, return first remaining type
        return remaining_types[0] if remaining_types else None
    
    async def _evaluate_condition(self, condition: Callable, current_results: Dict[ReasoningType, ReasoningResult], 
                                context: ReasoningContext) -> bool:
        """Evaluate a condition for adaptive reasoning."""
        try:
            return condition(current_results, context)
        except Exception:
            return True  # Default to true if condition evaluation fails


class ReasoningCoordinator:
    """Coordinates complex reasoning workflows and provides unified interface."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.manager = ReasoningManager(config)
        self.orchestrator = ReasoningOrchestrator(self.manager, config)
        self.session_counter = 0
    
    async def reason(self, context: ReasoningContext, 
                   reasoning_types: Optional[List[ReasoningType]] = None,
                   strategy: ReasoningStrategy = ReasoningStrategy.PARALLEL) -> Dict[ReasoningType, ReasoningResult]:
        """Main reasoning interface."""
        if reasoning_types is None:
            reasoning_types = [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE, ReasoningType.PROBABILISTIC]
        
        return await self.manager.reason_multiple(reasoning_types, context, strategy)
    
    async def reason_with_plan(self, context: ReasoningContext, plan: ReasoningPlan) -> Dict[ReasoningType, ReasoningResult]:
        """Reason using a specific plan."""
        return await self.orchestrator.execute_plan(plan, context)
    
    def create_plan(self, strategy: ReasoningStrategy, reasoning_types: List[ReasoningType],
                   dependencies: Optional[Dict[ReasoningType, List[ReasoningType]]] = None,
                   conditions: Optional[Dict[ReasoningType, Callable]] = None) -> ReasoningPlan:
        """Create a reasoning plan."""
        return ReasoningPlan(
            strategy=strategy,
            reasoning_types=reasoning_types,
            dependencies=dependencies or {},
            conditions=conditions or {}
        )
    
    async def start_session(self, context: ReasoningContext, plan: ReasoningPlan) -> str:
        """Start a reasoning session."""
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"
        
        session = ReasoningSession(
            session_id=session_id,
            context=context,
            plan=plan,
            start_time=asyncio.get_event_loop().time()
        )
        
        self.manager.active_sessions[session_id] = session
        return session_id
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a reasoning session."""
        session = self.manager.active_sessions.get(session_id)
        if not session:
            return None
        
        return {
            'session_id': session_id,
            'status': session.status,
            'completed_reasoning_types': list(session.results.keys()),
            'total_reasoning_types': len(session.plan.reasoning_types),
            'start_time': session.start_time,
            'end_time': session.end_time
        }
    
    def get_available_reasoning_types(self) -> List[ReasoningType]:
        """Get list of available reasoning types."""
        return self.manager.list_available_reasoners()
    
    async def analyze_problem(self, problem: str) -> List[ReasoningType]:
        """Analyze a problem and suggest appropriate reasoning types."""
        # Simple problem analysis - could be enhanced with ML
        suggested_types = []
        
        problem_lower = problem.lower()
        
        # Pattern-based suggestions
        if any(word in problem_lower for word in ['pattern', 'example', 'similar', 'generalize']):
            suggested_types.append(ReasoningType.INDUCTIVE)
        
        if any(word in problem_lower for word in ['logic', 'rule', 'if', 'then', 'proof']):
            suggested_types.append(ReasoningType.DEDUCTIVE)
        
        if any(word in problem_lower for word in ['probability', 'uncertain', 'likely', 'chance']):
            suggested_types.append(ReasoningType.PROBABILISTIC)
        
        if any(word in problem_lower for word in ['cause', 'effect', 'because', 'reason']):
            suggested_types.append(ReasoningType.CAUSAL)
        
        if any(word in problem_lower for word in ['explain', 'why', 'hypothesis', 'best']):
            suggested_types.append(ReasoningType.ABDUCTIVE)
        
        if any(word in problem_lower for word in ['like', 'similar', 'analogy', 'compare']):
            suggested_types.append(ReasoningType.ANALOGICAL)
        
        if any(word in problem_lower for word in ['time', 'sequence', 'temporal', 'when', 'before', 'after']):
            suggested_types.append(ReasoningType.TEMPORAL)
        
        if any(word in problem_lower for word in ['space', 'location', 'distance', 'near', 'spatial']):
            suggested_types.append(ReasoningType.SPATIAL)
        
        if any(word in problem_lower for word in ['multiple', 'agents', 'collaborate', 'consensus']):
            suggested_types.append(ReasoningType.COLLABORATIVE)
        
        if any(word in problem_lower for word in ['hybrid', 'combine', 'neural', 'symbolic']):
            suggested_types.append(ReasoningType.HYBRID)
        
        # If no specific patterns found, suggest basic reasoning types
        if not suggested_types:
            suggested_types = [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE, ReasoningType.PROBABILISTIC]
        
        return suggested_types