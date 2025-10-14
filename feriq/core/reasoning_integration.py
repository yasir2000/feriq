"""
Agent Reasoning Capabilities Integration

This module enhances Feriq agents with comprehensive reasoning capabilities,
allowing them to use different reasoning types for intelligent problem-solving.
"""

from typing import Dict, List, Any, Optional, Union
import asyncio
from feriq.reasoning import (
    ReasoningCoordinator, ReasoningManager, ReasoningContext, ReasoningType,
    Evidence, Hypothesis, ReasoningStrategy, ReasoningPlan, ReasoningResult
)


class ReasoningMixin:
    """Mixin class to add reasoning capabilities to any agent."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_coordinator = ReasoningCoordinator()
        self.reasoning_manager = ReasoningManager()
        self.default_reasoning_types = [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE]
        self.reasoning_history = []
    
    async def think(self, problem: str, evidence: List[str] = None, 
                   reasoning_types: List[ReasoningType] = None,
                   strategy: ReasoningStrategy = ReasoningStrategy.PARALLEL) -> Dict[str, Any]:
        """
        Core thinking method that uses reasoning system to analyze problems.
        
        Args:
            problem: The problem to analyze
            evidence: List of evidence strings
            reasoning_types: Specific reasoning types to use
            strategy: Reasoning strategy (parallel, sequential, etc.)
        
        Returns:
            Dictionary containing reasoning results and conclusions
        """
        
        # Create evidence objects
        evidence_objects = []
        if evidence:
            for i, ev in enumerate(evidence):
                evidence_objects.append(Evidence(
                    content=ev,
                    source=f"agent_{getattr(self, 'name', 'unknown')}",
                    confidence=0.8
                ))
        
        # Create reasoning context
        context = ReasoningContext(
            problem=problem,
            evidence=evidence_objects,
            metadata={
                'agent_name': getattr(self, 'name', 'unknown'),
                'agent_role': getattr(self, 'role', 'unknown'),
                'timestamp': asyncio.get_event_loop().time()
            }
        )
        
        # Auto-suggest reasoning types if not provided
        if reasoning_types is None:
            reasoning_types = await self.reasoning_coordinator.analyze_problem(problem)
            if not reasoning_types:
                reasoning_types = self.default_reasoning_types
        
        # Perform reasoning
        results = await self.reasoning_coordinator.reason(
            context,
            reasoning_types=reasoning_types,
            strategy=strategy
        )
        
        # Store in reasoning history
        reasoning_session = {
            'problem': problem,
            'evidence': evidence,
            'reasoning_types': [rt.value for rt in reasoning_types],
            'strategy': strategy.value,
            'results': results,
            'timestamp': context.metadata['timestamp']
        }
        self.reasoning_history.append(reasoning_session)
        
        # Extract key insights
        insights = self._extract_insights(results)
        
        return {
            'problem': problem,
            'reasoning_results': results,
            'insights': insights,
            'success': any(result.success for result in results.values()),
            'confidence': self._calculate_overall_confidence(results)
        }
    
    async def reason_inductively(self, examples: List[str], pattern_to_find: str = None) -> Dict[str, Any]:
        """Use inductive reasoning to find patterns in examples."""
        problem = f"Find patterns in the given examples: {pattern_to_find or 'general patterns'}"
        return await self.think(
            problem=problem,
            evidence=examples,
            reasoning_types=[ReasoningType.INDUCTIVE]
        )
    
    async def reason_deductively(self, rules: List[str], facts: List[str], conclusion_to_prove: str) -> Dict[str, Any]:
        """Use deductive reasoning to prove conclusions from rules and facts."""
        problem = f"Prove or disprove: {conclusion_to_prove}"
        evidence = rules + facts
        return await self.think(
            problem=problem,
            evidence=evidence,
            reasoning_types=[ReasoningType.DEDUCTIVE]
        )
    
    async def reason_probabilistically(self, uncertain_info: List[str], query: str) -> Dict[str, Any]:
        """Use probabilistic reasoning to handle uncertainty."""
        problem = f"Calculate probability or likelihood: {query}"
        return await self.think(
            problem=problem,
            evidence=uncertain_info,
            reasoning_types=[ReasoningType.PROBABILISTIC]
        )
    
    async def reason_causally(self, observations: List[str], causal_query: str) -> Dict[str, Any]:
        """Use causal reasoning to understand cause-effect relationships."""
        problem = f"Analyze causal relationships: {causal_query}"
        return await self.think(
            problem=problem,
            evidence=observations,
            reasoning_types=[ReasoningType.CAUSAL]
        )
    
    async def reason_abductively(self, observations: List[str], explanation_needed: str) -> Dict[str, Any]:
        """Use abductive reasoning to find best explanations."""
        problem = f"Find best explanation for: {explanation_needed}"
        return await self.think(
            problem=problem,
            evidence=observations,
            reasoning_types=[ReasoningType.ABDUCTIVE]
        )
    
    async def reason_analogically(self, source_case: str, target_case: str, aspect_to_analyze: str) -> Dict[str, Any]:
        """Use analogical reasoning to compare cases."""
        problem = f"Analyze similarity between source and target regarding: {aspect_to_analyze}"
        evidence = [f"Source case: {source_case}", f"Target case: {target_case}"]
        return await self.think(
            problem=problem,
            evidence=evidence,
            reasoning_types=[ReasoningType.ANALOGICAL]
        )
    
    async def reason_temporally(self, time_series_data: List[str], temporal_query: str) -> Dict[str, Any]:
        """Use temporal reasoning to analyze time-based patterns."""
        problem = f"Analyze temporal patterns: {temporal_query}"
        return await self.think(
            problem=problem,
            evidence=time_series_data,
            reasoning_types=[ReasoningType.TEMPORAL]
        )
    
    async def reason_spatially(self, spatial_data: List[str], spatial_query: str) -> Dict[str, Any]:
        """Use spatial reasoning to analyze spatial relationships."""
        problem = f"Analyze spatial relationships: {spatial_query}"
        return await self.think(
            problem=problem,
            evidence=spatial_data,
            reasoning_types=[ReasoningType.SPATIAL]
        )
    
    async def multi_step_reasoning(self, problem: str, reasoning_pipeline: List[ReasoningType]) -> Dict[str, Any]:
        """Perform multi-step reasoning using a pipeline of reasoning types."""
        
        # Create hierarchical plan
        plan = self.reasoning_coordinator.create_plan(
            strategy=ReasoningStrategy.PIPELINE,
            reasoning_types=reasoning_pipeline
        )
        
        # Create context
        context = ReasoningContext(
            problem=problem,
            evidence=[Evidence(content=problem, source=f"agent_{getattr(self, 'name', 'unknown')}")],
            metadata={'multi_step': True, 'pipeline': [rt.value for rt in reasoning_pipeline]}
        )
        
        # Execute pipeline
        results = await self.reasoning_coordinator.reason_with_plan(context, plan)
        
        return {
            'problem': problem,
            'pipeline': [rt.value for rt in reasoning_pipeline],
            'results': results,
            'success': any(result.success for result in results.values())
        }
    
    async def collaborative_think(self, problem: str, other_agents: List['ReasoningMixin'], 
                                consensus_method: str = 'majority_vote') -> Dict[str, Any]:
        """Think collaboratively with other reasoning-enabled agents."""
        
        # This would integrate with the collaborative reasoning system
        context = ReasoningContext(
            problem=problem,
            evidence=[Evidence(content=problem, source="collaborative_session")],
            metadata={
                'collaboration_type': 'consensus',
                'consensus_method': consensus_method,
                'participants': [getattr(agent, 'name', 'unknown') for agent in other_agents + [self]]
            }
        )
        
        results = await self.reasoning_coordinator.reason(
            context,
            reasoning_types=[ReasoningType.COLLABORATIVE],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        return {
            'problem': problem,
            'collaborative_results': results,
            'participants': len(other_agents) + 1,
            'consensus_reached': results.get(ReasoningType.COLLABORATIVE, ReasoningResult()).success
        }
    
    def _extract_insights(self, results: Dict[ReasoningType, ReasoningResult]) -> List[str]:
        """Extract key insights from reasoning results."""
        insights = []
        
        for reasoning_type, result in results.items():
            if result.success and result.conclusions:
                for conclusion in result.conclusions:
                    if conclusion.confidence > 0.7:  # High confidence insights
                        insights.append(f"[{reasoning_type.value}] {conclusion.statement}")
        
        return insights
    
    def _calculate_overall_confidence(self, results: Dict[ReasoningType, ReasoningResult]) -> float:
        """Calculate overall confidence from all reasoning results."""
        valid_results = [result for result in results.values() if result.success]
        if not valid_results:
            return 0.0
        
        total_confidence = sum(result.confidence for result in valid_results)
        return total_confidence / len(valid_results)
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get the agent's reasoning history."""
        return self.reasoning_history
    
    def clear_reasoning_history(self):
        """Clear the agent's reasoning history."""
        self.reasoning_history.clear()


# Integration with existing agent classes
def enhance_agent_with_reasoning(agent_class):
    """
    Decorator to enhance any agent class with reasoning capabilities.
    
    Usage:
        @enhance_agent_with_reasoning
        class MyAgent(BaseAgent):
            pass
    """
    
    class ReasoningEnhancedAgent(ReasoningMixin, agent_class):
        pass
    
    return ReasoningEnhancedAgent


# Example usage with core agent types
class ReasoningAgent:
    """
    Standalone reasoning agent that focuses primarily on thinking and problem-solving.
    """
    
    def __init__(self, name: str, role: str = "reasoning_specialist", expertise: List[str] = None):
        self.name = name
        self.role = role
        self.expertise = expertise or []
        self.reasoning_coordinator = ReasoningCoordinator()
        self.reasoning_history = []
    
    async def solve_problem(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main problem-solving method using comprehensive reasoning.
        """
        
        # Analyze problem to determine best reasoning approaches
        suggested_types = await self.reasoning_coordinator.analyze_problem(problem)
        
        # Create evidence from context
        evidence_objects = []
        if context:
            for key, value in context.items():
                evidence_objects.append(Evidence(
                    content=f"{key}: {value}",
                    source=f"context_{key}",
                    confidence=0.8
                ))
        
        # Create reasoning context
        reasoning_context = ReasoningContext(
            problem=problem,
            evidence=evidence_objects,
            metadata={
                'agent_name': self.name,
                'agent_role': self.role,
                'expertise': self.expertise
            }
        )
        
        # Use adaptive reasoning strategy for complex problems
        if len(suggested_types) > 3:
            strategy = ReasoningStrategy.ADAPTIVE
        else:
            strategy = ReasoningStrategy.PARALLEL
        
        # Perform reasoning
        results = await self.reasoning_coordinator.reason(
            reasoning_context,
            reasoning_types=suggested_types,
            strategy=strategy
        )
        
        # Synthesize solution
        solution = self._synthesize_solution(problem, results)
        
        # Store in history
        self.reasoning_history.append({
            'problem': problem,
            'context': context,
            'reasoning_types': [rt.value for rt in suggested_types],
            'results': results,
            'solution': solution
        })
        
        return solution
    
    def _synthesize_solution(self, problem: str, results: Dict[ReasoningType, ReasoningResult]) -> Dict[str, Any]:
        """Synthesize a comprehensive solution from reasoning results."""
        
        solution_components = []
        confidence_scores = []
        reasoning_evidence = []
        
        for reasoning_type, result in results.items():
            if result.success:
                for conclusion in result.conclusions:
                    solution_components.append({
                        'reasoning_type': reasoning_type.value,
                        'conclusion': conclusion.statement,
                        'confidence': conclusion.confidence,
                        'reasoning_chain': conclusion.reasoning_chain
                    })
                    confidence_scores.append(conclusion.confidence)
                    reasoning_evidence.extend(conclusion.reasoning_chain)
        
        # Calculate overall solution confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Create final solution
        solution = {
            'problem': problem,
            'solution_components': solution_components,
            'overall_confidence': overall_confidence,
            'reasoning_evidence': reasoning_evidence,
            'reasoning_types_used': list(set(comp['reasoning_type'] for comp in solution_components)),
            'success': len(solution_components) > 0,
            'solver': self.name
        }
        
        return solution