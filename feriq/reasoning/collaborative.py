"""
Collaborative Reasoning Module

Implements multi-agent reasoning coordination, consensus building, and distributed problem solving.
"""

from typing import Any, Dict, List, Optional, Set, Union, Tuple
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from .base import BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType, Evidence, Hypothesis, Conclusion


class ConsensusMethod(Enum):
    """Methods for reaching consensus among agents."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    DELPHI = "delphi"
    ARGUMENTATION = "argumentation"


class ArgumentationType(Enum):
    """Types of argumentation frameworks."""
    ABSTRACT = "abstract"
    STRUCTURED = "structured"
    PROBABILISTIC = "probabilistic"


@dataclass
class Agent:
    """Represents a reasoning agent in collaborative system."""
    id: str
    name: str
    expertise: List[str]
    confidence_threshold: float = 0.5
    weight: float = 1.0
    reasoner: Optional[BaseReasoner] = None


@dataclass
class Argument:
    """Represents an argument in argumentation framework."""
    id: str
    claim: str
    premises: List[str]
    strength: float
    author_id: str
    supports: Optional[str] = None  # ID of argument this supports
    attacks: Optional[str] = None   # ID of argument this attacks


@dataclass
class ConsensusState:
    """Tracks the state of consensus building process."""
    round_number: int = 0
    participants: Set[str] = field(default_factory=set)
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    agreements: Dict[str, float] = field(default_factory=dict)
    converged: bool = False


class ConsensusBuilder:
    """Builds consensus among multiple reasoning agents."""
    
    def __init__(self, method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE, config: Optional[Dict[str, Any]] = None):
        self.method = method
        self.config = config or {}
        self.max_rounds = config.get('max_rounds', 5)
        self.convergence_threshold = config.get('convergence_threshold', 0.8)
    
    async def build_consensus(self, agents: List[Agent], context: ReasoningContext) -> Dict[str, Any]:
        """Build consensus among agents using specified method."""
        if self.method == ConsensusMethod.MAJORITY_VOTE:
            return await self._majority_vote_consensus(agents, context)
        elif self.method == ConsensusMethod.WEIGHTED_AVERAGE:
            return await self._weighted_average_consensus(agents, context)
        elif self.method == ConsensusMethod.DELPHI:
            return await self._delphi_consensus(agents, context)
        elif self.method == ConsensusMethod.ARGUMENTATION:
            return await self._argumentation_consensus(agents, context)
        else:
            raise ValueError(f"Unknown consensus method: {self.method}")
    
    async def _majority_vote_consensus(self, agents: List[Agent], context: ReasoningContext) -> Dict[str, Any]:
        """Build consensus using majority voting."""
        # Collect votes from all agents
        votes = {}
        agent_results = {}
        
        for agent in agents:
            if agent.reasoner:
                result = await agent.reasoner.reason(context)
                agent_results[agent.id] = result
                
                # Extract vote from result
                if result.success and result.conclusions:
                    vote = result.conclusions[0].statement
                    if vote not in votes:
                        votes[vote] = []
                    votes[vote].append((agent.id, result.confidence))
        
        # Find majority
        if not votes:
            return {'success': False, 'error': 'No valid votes received'}
        
        vote_counts = {vote: len(voters) for vote, voters in votes.items()}
        majority_vote = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_vote]
        
        consensus_strength = majority_count / len(agents)
        
        return {
            'success': True,
            'method': 'majority_vote',
            'consensus_decision': majority_vote,
            'consensus_strength': consensus_strength,
            'vote_distribution': vote_counts,
            'agent_results': agent_results
        }
    
    async def _weighted_average_consensus(self, agents: List[Agent], context: ReasoningContext) -> Dict[str, Any]:
        """Build consensus using weighted averaging."""
        weighted_conclusions = defaultdict(float)
        total_weight = 0
        agent_results = {}
        
        for agent in agents:
            if agent.reasoner:
                result = await agent.reasoner.reason(context)
                agent_results[agent.id] = result
                
                if result.success and result.conclusions:
                    conclusion = result.conclusions[0]
                    weighted_value = result.confidence * agent.weight
                    weighted_conclusions[conclusion.statement] += weighted_value
                    total_weight += agent.weight
        
        if total_weight == 0:
            return {'success': False, 'error': 'No valid weighted contributions'}
        
        # Normalize and find best conclusion
        normalized_conclusions = {
            statement: weight / total_weight 
            for statement, weight in weighted_conclusions.items()
        }
        
        best_conclusion = max(normalized_conclusions, key=normalized_conclusions.get)
        consensus_strength = normalized_conclusions[best_conclusion]
        
        return {
            'success': True,
            'method': 'weighted_average',
            'consensus_decision': best_conclusion,
            'consensus_strength': consensus_strength,
            'weighted_distribution': normalized_conclusions,
            'agent_results': agent_results
        }
    
    async def _delphi_consensus(self, agents: List[Agent], context: ReasoningContext) -> Dict[str, Any]:
        """Build consensus using Delphi method with multiple rounds."""
        consensus_state = ConsensusState()
        consensus_state.participants = {agent.id for agent in agents}
        
        for round_num in range(self.max_rounds):
            consensus_state.round_number = round_num + 1
            
            # Collect responses from agents
            round_results = {}
            for agent in agents:
                if agent.reasoner:
                    # Provide previous round feedback in context
                    round_context = context
                    if round_num > 0:
                        round_context.metadata['previous_results'] = consensus_state.proposals
                    
                    result = await agent.reasoner.reason(round_context)
                    round_results[agent.id] = result
            
            # Analyze convergence
            convergence_score = await self._calculate_convergence(round_results)
            consensus_state.proposals.append({
                'round': round_num + 1,
                'results': round_results,
                'convergence': convergence_score
            })
            
            if convergence_score >= self.convergence_threshold:
                consensus_state.converged = True
                break
        
        # Generate final consensus
        final_consensus = await self._extract_delphi_consensus(consensus_state)
        
        return {
            'success': True,
            'method': 'delphi',
            'rounds_completed': consensus_state.round_number,
            'converged': consensus_state.converged,
            'final_consensus': final_consensus,
            'consensus_history': consensus_state.proposals
        }
    
    async def _argumentation_consensus(self, agents: List[Agent], context: ReasoningContext) -> Dict[str, Any]:
        """Build consensus using argumentation framework."""
        # This is a simplified implementation
        arguments = []
        agent_results = {}
        
        # Collect arguments from agents
        for agent in agents:
            if agent.reasoner:
                result = await agent.reasoner.reason(context)
                agent_results[agent.id] = result
                
                if result.success and result.conclusions:
                    for i, conclusion in enumerate(result.conclusions):
                        argument = Argument(
                            id=f"{agent.id}_arg_{i}",
                            claim=conclusion.statement,
                            premises=conclusion.reasoning_chain,
                            strength=conclusion.confidence,
                            author_id=agent.id
                        )
                        arguments.append(argument)
        
        # Evaluate argument interactions (simplified)
        evaluated_arguments = await self._evaluate_arguments(arguments)
        
        # Find strongest undefeated arguments
        consensus_arguments = [arg for arg in evaluated_arguments if arg.strength > 0.6]
        
        return {
            'success': True,
            'method': 'argumentation',
            'total_arguments': len(arguments),
            'consensus_arguments': len(consensus_arguments),
            'strongest_arguments': consensus_arguments[:3],
            'all_arguments': evaluated_arguments,
            'agent_results': agent_results
        }
    
    async def _calculate_convergence(self, round_results: Dict[str, ReasoningResult]) -> float:
        """Calculate convergence score for Delphi method."""
        if len(round_results) < 2:
            return 0.0
        
        # Simple convergence based on confidence similarity
        confidences = [
            result.confidence for result in round_results.values() 
            if result.success
        ]
        
        if len(confidences) < 2:
            return 0.0
        
        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Lower variance indicates higher convergence
        convergence = max(0.0, 1.0 - variance)
        return convergence
    
    async def _extract_delphi_consensus(self, state: ConsensusState) -> Dict[str, Any]:
        """Extract final consensus from Delphi rounds."""
        if not state.proposals:
            return {'consensus': 'No consensus reached', 'confidence': 0.0}
        
        # Use last round results
        last_round = state.proposals[-1]
        results = last_round['results']
        
        # Find most common conclusion
        conclusion_counts = defaultdict(int)
        total_confidence = 0
        
        for result in results.values():
            if result.success and result.conclusions:
                conclusion = result.conclusions[0].statement
                conclusion_counts[conclusion] += 1
                total_confidence += result.confidence
        
        if conclusion_counts:
            best_conclusion = max(conclusion_counts, key=conclusion_counts.get)
            avg_confidence = total_confidence / len(results)
            
            return {
                'consensus': best_conclusion,
                'confidence': avg_confidence,
                'support_count': conclusion_counts[best_conclusion],
                'convergence_achieved': state.converged
            }
        
        return {'consensus': 'No valid consensus', 'confidence': 0.0}
    
    async def _evaluate_arguments(self, arguments: List[Argument]) -> List[Argument]:
        """Evaluate arguments for strength and interactions."""
        # Simplified argument evaluation
        for argument in arguments:
            # Check for supporting/attacking relationships (simplified)
            support_count = sum(1 for other in arguments if other.supports == argument.id)
            attack_count = sum(1 for other in arguments if other.attacks == argument.id)
            
            # Adjust strength based on support/attacks
            adjustment = (support_count * 0.1) - (attack_count * 0.1)
            argument.strength = max(0.0, min(1.0, argument.strength + adjustment))
        
        return sorted(arguments, key=lambda x: x.strength, reverse=True)


class DistributedProblemSolver:
    """Coordinates distributed problem solving across multiple agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.task_queue = []
        self.completed_tasks = []
    
    async def solve_distributed(self, problem: Dict[str, Any], agents: List[Agent]) -> Dict[str, Any]:
        """Solve a problem using distributed approach."""
        # Decompose problem into subtasks
        subtasks = await self._decompose_problem(problem)
        
        # Assign subtasks to agents based on expertise
        assignments = await self._assign_tasks(subtasks, agents)
        
        # Execute subtasks
        subtask_results = {}
        for agent_id, tasks in assignments.items():
            agent = next((a for a in agents if a.id == agent_id), None)
            if agent and agent.reasoner:
                for task in tasks:
                    context = ReasoningContext(
                        problem=task['description'],
                        evidence=[Evidence(content=str(task), source="task_decomposition")],
                        metadata=task
                    )
                    result = await agent.reasoner.reason(context)
                    subtask_results[task['id']] = {
                        'agent_id': agent_id,
                        'task': task,
                        'result': result
                    }
        
        # Integrate results
        integrated_solution = await self._integrate_solutions(subtask_results, problem)
        
        return {
            'success': True,
            'original_problem': problem,
            'subtasks': subtasks,
            'assignments': assignments,
            'subtask_results': subtask_results,
            'integrated_solution': integrated_solution
        }
    
    async def _decompose_problem(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose problem into manageable subtasks."""
        # Simplified problem decomposition
        problem_text = problem.get('description', '')
        complexity = len(problem_text.split())
        
        if complexity < 10:
            # Simple problem - single task
            return [{'id': 'task_1', 'description': problem_text, 'complexity': complexity}]
        else:
            # Complex problem - multiple tasks
            num_subtasks = min(4, max(2, complexity // 20))
            subtasks = []
            
            for i in range(num_subtasks):
                subtask = {
                    'id': f'task_{i+1}',
                    'description': f"Subtask {i+1} of {problem_text}",
                    'complexity': complexity // num_subtasks
                }
                subtasks.append(subtask)
            
            return subtasks
    
    async def _assign_tasks(self, subtasks: List[Dict[str, Any]], agents: List[Agent]) -> Dict[str, List[Dict[str, Any]]]:
        """Assign subtasks to agents based on expertise and load."""
        assignments = defaultdict(list)
        
        # Simple round-robin assignment (could be improved with expertise matching)
        for i, task in enumerate(subtasks):
            agent = agents[i % len(agents)]
            assignments[agent.id].append(task)
        
        return dict(assignments)
    
    async def _integrate_solutions(self, subtask_results: Dict[str, Any], original_problem: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate solutions from subtasks into final solution."""
        successful_results = [
            result for result in subtask_results.values() 
            if result['result'].success
        ]
        
        if not successful_results:
            return {'success': False, 'error': 'No successful subtask results'}
        
        # Combine conclusions
        all_conclusions = []
        total_confidence = 0
        
        for result_data in successful_results:
            result = result_data['result']
            all_conclusions.extend(result.conclusions)
            total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(successful_results)
        
        return {
            'success': True,
            'combined_conclusions': all_conclusions,
            'average_confidence': avg_confidence,
            'subtasks_completed': len(successful_results),
            'integration_method': 'conclusion_aggregation'
        }


class CollaborativeReasoner(BaseReasoner):
    """Main collaborative reasoning engine."""
    
    def __init__(self, name: str = "CollaborativeReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.COLLABORATIVE, config)
        self.agents = []
        self.consensus_builder = ConsensusBuilder(config=config)
        self.distributed_solver = DistributedProblemSolver(config)
    
    def add_agent(self, agent: Agent):
        """Add an agent to the collaborative system."""
        self.agents.append(agent)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform collaborative reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            if not self.agents:
                raise ValueError("No agents available for collaborative reasoning")
            
            collaboration_type = context.metadata.get('collaboration_type', 'consensus')
            
            if collaboration_type == 'consensus':
                collab_result = await self.consensus_builder.build_consensus(self.agents, context)
                result.reasoning_trace.append(f"Built consensus using {self.consensus_builder.method.value}")
                
            elif collaboration_type == 'distributed':
                problem = {'description': context.problem}
                collab_result = await self.distributed_solver.solve_distributed(problem, self.agents)
                result.reasoning_trace.append("Solved problem using distributed approach")
            
            else:
                raise ValueError(f"Unknown collaboration type: {collaboration_type}")
            
            # Create conclusion from collaborative result
            if collab_result.get('success', False):
                if collaboration_type == 'consensus':
                    conclusion_text = collab_result.get('consensus_decision', 'Consensus reached')
                    confidence = collab_result.get('consensus_strength', 0.7)
                else:  # distributed
                    conclusion_text = f"Distributed solution with {collab_result.get('subtasks_completed', 0)} completed subtasks"
                    confidence = collab_result.get('integrated_solution', {}).get('average_confidence', 0.7)
                
                conclusion = Conclusion(
                    statement=conclusion_text,
                    confidence=confidence,
                    reasoning_type=self.reasoning_type,
                    reasoning_chain=[f"Collaborative {collaboration_type} reasoning"],
                    metadata={'collaborative_result': collab_result}
                )
                result.conclusions.append(conclusion)
                result.confidence = confidence
                result.success = True
            else:
                result.success = False
                result.error_message = collab_result.get('error', 'Collaborative reasoning failed')
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        return result