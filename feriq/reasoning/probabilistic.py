"""
Probabilistic Reasoning Module

Implements Bayesian networks, MCMC sampling, variational inference,
and uncertainty quantification for probabilistic reasoning.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
import asyncio
import numpy as np
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
import math

from .base import (
    BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType,
    Evidence, Hypothesis, Conclusion, create_evidence, create_conclusion
)


@dataclass
class ProbabilityDistribution:
    """Represents a probability distribution."""
    name: str
    distribution_type: str  # 'discrete', 'continuous', 'categorical'
    parameters: Dict[str, Any] = field(default_factory=dict)
    values: List[Any] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if self.distribution_type == 'discrete' and self.probabilities:
            # Normalize probabilities
            total = sum(self.probabilities)
            if total > 0:
                self.probabilities = [p / total for p in self.probabilities]
    
    def sample(self) -> Any:
        """Sample from the distribution."""
        if self.distribution_type == 'discrete':
            return np.random.choice(self.values, p=self.probabilities)
        elif self.distribution_type == 'categorical':
            return np.random.choice(self.values, p=self.probabilities)
        elif self.distribution_type == 'continuous':
            if self.parameters.get('type') == 'normal':
                mean = self.parameters.get('mean', 0)
                std = self.parameters.get('std', 1)
                return np.random.normal(mean, std)
            elif self.parameters.get('type') == 'uniform':
                low = self.parameters.get('low', 0)
                high = self.parameters.get('high', 1)
                return np.random.uniform(low, high)
        
        return None
    
    def probability(self, value: Any) -> float:
        """Get probability of a specific value."""
        if self.distribution_type in ['discrete', 'categorical']:
            try:
                index = self.values.index(value)
                return self.probabilities[index]
            except ValueError:
                return 0.0
        else:
            # For continuous distributions, return density
            if self.parameters.get('type') == 'normal':
                mean = self.parameters.get('mean', 0)
                std = self.parameters.get('std', 1)
                return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((value - mean) / std) ** 2)
        
        return 0.0


@dataclass
class BayesianNode:
    """Node in a Bayesian network."""
    name: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    distribution: Optional[ProbabilityDistribution] = None
    conditional_probabilities: Dict[Tuple, ProbabilityDistribution] = field(default_factory=dict)
    evidence: Optional[Any] = None
    
    def get_distribution(self, parent_values: Dict[str, Any] = None) -> ProbabilityDistribution:
        """Get distribution given parent values."""
        if not self.parents or parent_values is None:
            return self.distribution
        
        # Create key from parent values
        key = tuple(parent_values.get(parent, None) for parent in self.parents)
        
        return self.conditional_probabilities.get(key, self.distribution)


class BayesianNetwork:
    """Bayesian network for probabilistic reasoning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.nodes = {}  # name -> BayesianNode
        self.topology = defaultdict(list)  # parent -> [children]
        
    async def add_node(self, node: BayesianNode) -> None:
        """Add a node to the network."""
        self.nodes[node.name] = node
        
        # Update topology
        for parent in node.parents:
            self.topology[parent].append(node.name)
    
    async def set_evidence(self, node_name: str, value: Any) -> None:
        """Set evidence for a node."""
        if node_name in self.nodes:
            self.nodes[node_name].evidence = value
    
    async def clear_evidence(self) -> None:
        """Clear all evidence."""
        for node in self.nodes.values():
            node.evidence = None
    
    async def get_topological_order(self) -> List[str]:
        """Get topological ordering of nodes."""
        in_degree = {name: len(node.parents) for name, node in self.nodes.items()}
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node_name = queue.popleft()
            result.append(node_name)
            
            for child in self.topology[node_name]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    async def forward_sampling(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate samples using forward sampling."""
        samples = []
        topological_order = await self.get_topological_order()
        
        for _ in range(num_samples):
            sample = {}
            
            for node_name in topological_order:
                node = self.nodes[node_name]
                
                if node.evidence is not None:
                    sample[node_name] = node.evidence
                else:
                    # Get parent values
                    parent_values = {parent: sample[parent] for parent in node.parents if parent in sample}
                    
                    # Get distribution and sample
                    distribution = node.get_distribution(parent_values)
                    if distribution:
                        sample[node_name] = distribution.sample()
            
            samples.append(sample)
        
        return samples
    
    async def likelihood_weighting(self, num_samples: int = 1000) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Generate weighted samples using likelihood weighting."""
        samples = []
        weights = []
        topological_order = await self.get_topological_order()
        
        for _ in range(num_samples):
            sample = {}
            weight = 1.0
            
            for node_name in topological_order:
                node = self.nodes[node_name]
                
                # Get parent values
                parent_values = {parent: sample[parent] for parent in node.parents if parent in sample}
                distribution = node.get_distribution(parent_values)
                
                if node.evidence is not None:
                    sample[node_name] = node.evidence
                    if distribution:
                        weight *= distribution.probability(node.evidence)
                else:
                    if distribution:
                        sample[node_name] = distribution.sample()
            
            samples.append(sample)
            weights.append(weight)
        
        return samples, weights
    
    async def gibbs_sampling(self, num_samples: int = 1000, burn_in: int = 100) -> List[Dict[str, Any]]:
        """Generate samples using Gibbs sampling."""
        samples = []
        
        # Initialize sample
        current_sample = {}
        for node_name, node in self.nodes.items():
            if node.evidence is not None:
                current_sample[node_name] = node.evidence
            else:
                # Random initialization
                if node.distribution:
                    current_sample[node_name] = node.distribution.sample()
        
        # Gibbs sampling
        non_evidence_nodes = [name for name, node in self.nodes.items() if node.evidence is None]
        
        for iteration in range(num_samples + burn_in):
            for node_name in non_evidence_nodes:
                # Sample from conditional distribution
                conditional_dist = await self.get_conditional_distribution(node_name, current_sample)
                if conditional_dist:
                    current_sample[node_name] = conditional_dist.sample()
            
            # Store sample after burn-in
            if iteration >= burn_in:
                samples.append(current_sample.copy())
        
        return samples
    
    async def get_conditional_distribution(self, 
                                         node_name: str, 
                                         current_sample: Dict[str, Any]) -> Optional[ProbabilityDistribution]:
        """Get conditional distribution for a node given current sample."""
        node = self.nodes[node_name]
        
        # Get parent values
        parent_values = {parent: current_sample[parent] for parent in node.parents if parent in current_sample}
        
        # Get prior distribution
        prior_dist = node.get_distribution(parent_values)
        if not prior_dist:
            return None
        
        # Calculate unnormalized probabilities considering children
        if prior_dist.distribution_type in ['discrete', 'categorical']:
            unnormalized_probs = []
            
            for i, value in enumerate(prior_dist.values):
                prob = prior_dist.probabilities[i]
                
                # Multiply by likelihood from children
                for child_name in node.children:
                    child_node = self.nodes[child_name]
                    child_value = current_sample.get(child_name)
                    
                    if child_value is not None:
                        # Get child's distribution given this value for current node
                        child_parent_values = parent_values.copy()
                        child_parent_values[node_name] = value
                        
                        child_dist = child_node.get_distribution(child_parent_values)
                        if child_dist:
                            prob *= child_dist.probability(child_value)
                
                unnormalized_probs.append(prob)
            
            # Normalize
            total = sum(unnormalized_probs)
            if total > 0:
                normalized_probs = [p / total for p in unnormalized_probs]
                
                return ProbabilityDistribution(
                    name=f"{node_name}_conditional",
                    distribution_type=prior_dist.distribution_type,
                    values=prior_dist.values,
                    probabilities=normalized_probs
                )
        
        return prior_dist
    
    async def query_probability(self, 
                               query: Dict[str, Any], 
                               evidence: Dict[str, Any] = None,
                               method: str = 'likelihood_weighting',
                               num_samples: int = 1000) -> float:
        """Query probability P(query | evidence)."""
        # Set evidence
        if evidence:
            await self.clear_evidence()
            for node_name, value in evidence.items():
                await self.set_evidence(node_name, value)
        
        if method == 'forward_sampling':
            samples = await self.forward_sampling(num_samples)
            weights = [1.0] * len(samples)
        elif method == 'likelihood_weighting':
            samples, weights = await self.likelihood_weighting(num_samples)
        elif method == 'gibbs_sampling':
            samples = await self.gibbs_sampling(num_samples)
            weights = [1.0] * len(samples)
        else:
            return 0.0
        
        # Count matches
        matching_samples = 0
        total_weight = 0
        
        for sample, weight in zip(samples, weights):
            # Check if sample matches query
            matches_query = all(sample.get(var) == value for var, value in query.items())
            
            if matches_query:
                matching_samples += weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return matching_samples / total_weight


class MCMCReasoner:
    """Markov Chain Monte Carlo reasoning engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chains = {}
        self.burn_in = config.get('burn_in', 100)
        self.num_samples = config.get('num_samples', 1000)
    
    async def metropolis_hastings(self, 
                                 target_distribution: Callable[[Any], float],
                                 proposal_distribution: Callable[[Any], Any],
                                 initial_state: Any,
                                 num_samples: int = None) -> List[Any]:
        """Metropolis-Hastings MCMC sampling."""
        num_samples = num_samples or self.num_samples
        samples = []
        current_state = initial_state
        current_log_prob = math.log(target_distribution(current_state) + 1e-10)
        
        accepted = 0
        
        for i in range(num_samples + self.burn_in):
            # Propose new state
            proposed_state = proposal_distribution(current_state)
            proposed_log_prob = math.log(target_distribution(proposed_state) + 1e-10)
            
            # Accept/reject
            log_ratio = proposed_log_prob - current_log_prob
            
            if log_ratio > 0 or math.log(random.random()) < log_ratio:
                # Accept
                current_state = proposed_state
                current_log_prob = proposed_log_prob
                accepted += 1
            
            # Store sample after burn-in
            if i >= self.burn_in:
                samples.append(current_state)
        
        acceptance_rate = accepted / (num_samples + self.burn_in)
        return samples, acceptance_rate
    
    async def hamiltonian_monte_carlo(self, 
                                    log_prob_fn: Callable[[np.ndarray], float],
                                    grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
                                    initial_state: np.ndarray,
                                    step_size: float = 0.1,
                                    num_steps: int = 10,
                                    num_samples: int = None) -> List[np.ndarray]:
        """Hamiltonian Monte Carlo sampling."""
        num_samples = num_samples or self.num_samples
        samples = []
        current_q = initial_state.copy()
        
        for i in range(num_samples + self.burn_in):
            # Sample momentum
            current_p = np.random.normal(0, 1, size=current_q.shape)
            
            # Leapfrog integration
            q = current_q.copy()
            p = current_p.copy()
            
            # Half step for momentum
            p = p + 0.5 * step_size * grad_log_prob_fn(q)
            
            # Alternating full steps
            for _ in range(num_steps):
                q = q + step_size * p
                if _ != num_steps - 1:  # Don't do this on the last step
                    p = p + step_size * grad_log_prob_fn(q)
            
            # Half step for momentum
            p = p + 0.5 * step_size * grad_log_prob_fn(q)
            
            # Negate momentum for reversibility
            p = -p
            
            # Metropolis acceptance
            current_H = -log_prob_fn(current_q) + 0.5 * np.sum(current_p ** 2)
            proposed_H = -log_prob_fn(q) + 0.5 * np.sum(p ** 2)
            
            if math.log(random.random()) < current_H - proposed_H:
                current_q = q
            
            # Store sample after burn-in
            if i >= self.burn_in:
                samples.append(current_q.copy())
        
        return samples


class VariationalReasoner:
    """Variational inference for approximate Bayesian reasoning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.learning_rate = config.get('learning_rate', 0.01)
        self.max_iterations = config.get('max_iterations', 1000)
        self.tolerance = config.get('tolerance', 1e-6)
    
    async def mean_field_approximation(self, 
                                     target_log_prob: Callable[[Dict[str, Any]], float],
                                     variables: List[str],
                                     initial_params: Dict[str, Dict[str, float]]) -> Dict[str, ProbabilityDistribution]:
        """Mean field variational approximation."""
        params = initial_params.copy()
        
        for iteration in range(self.max_iterations):
            old_params = {var: p.copy() for var, p in params.items()}
            
            # Update each variable's parameters
            for var in variables:
                params[var] = await self.update_mean_field_params(
                    var, variables, params, target_log_prob
                )
            
            # Check convergence
            converged = all(
                abs(params[var]['mean'] - old_params[var]['mean']) < self.tolerance
                for var in variables
            )
            
            if converged:
                break
        
        # Convert parameters to distributions
        distributions = {}
        for var, var_params in params.items():
            distributions[var] = ProbabilityDistribution(
                name=var,
                distribution_type='continuous',
                parameters={
                    'type': 'normal',
                    'mean': var_params['mean'],
                    'std': math.sqrt(var_params['variance'])
                }
            )
        
        return distributions
    
    async def update_mean_field_params(self, 
                                     variable: str,
                                     all_variables: List[str],
                                     current_params: Dict[str, Dict[str, float]],
                                     target_log_prob: Callable[[Dict[str, Any]], float]) -> Dict[str, float]:
        """Update variational parameters for one variable."""
        # Sample other variables
        other_samples = {}
        for var in all_variables:
            if var != variable:
                mean = current_params[var]['mean']
                std = math.sqrt(current_params[var]['variance'])
                other_samples[var] = np.random.normal(mean, std)
        
        # Estimate optimal parameters using gradient ascent
        mean = current_params[variable]['mean']
        log_var = math.log(current_params[variable]['variance'])
        
        # Simple gradient estimation
        num_samples = 100
        mean_grad = 0
        var_grad = 0
        
        for _ in range(num_samples):
            # Sample from current distribution
            std = math.sqrt(current_params[variable]['variance'])
            sample = np.random.normal(mean, std)
            
            # Evaluate log probability
            full_sample = other_samples.copy()
            full_sample[variable] = sample
            log_prob = target_log_prob(full_sample)
            
            # Compute gradients (simplified)
            mean_grad += (sample - mean) * log_prob / (std ** 2)
            var_grad += ((sample - mean) ** 2 - std ** 2) * log_prob / (2 * std ** 4)
        
        mean_grad /= num_samples
        var_grad /= num_samples
        
        # Update parameters
        new_mean = mean + self.learning_rate * mean_grad
        new_log_var = log_var + self.learning_rate * var_grad
        new_variance = math.exp(new_log_var)
        
        return {
            'mean': new_mean,
            'variance': max(new_variance, 1e-6)  # Ensure positive variance
        }


class BayesianNetworkReasoner(BaseReasoner):
    """Bayesian network-based probabilistic reasoner."""
    
    def __init__(self, name: str = "BayesianNetworkReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.PROBABILISTIC, config)
        self.network = BayesianNetwork(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform Bayesian network reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Build network from context
            await self.build_network_from_context(context)
            
            # Set evidence
            evidence_dict = {}
            for evidence in context.available_evidence:
                if isinstance(evidence.content, dict) and 'variable' in evidence.content:
                    var_name = evidence.content['variable']
                    var_value = evidence.content['value']
                    evidence_dict[var_name] = var_value
                    await self.network.set_evidence(var_name, var_value)
            
            result.reasoning_trace.append(f"Set evidence for {len(evidence_dict)} variables")
            
            # Perform inference
            if context.goal:
                # Query probability of goal
                query_vars = await self.parse_query(context.goal)
                if query_vars:
                    probability = await self.network.query_probability(
                        query_vars, evidence_dict
                    )
                    
                    conclusion = Conclusion(
                        statement=f"P({context.goal}) = {probability:.4f}",
                        confidence=probability,
                        reasoning_type=self.reasoning_type,
                        reasoning_chain=["Bayesian network inference"]
                    )
                    result.conclusions.append(conclusion)
                    result.reasoning_trace.append(f"Queried probability: {probability:.4f}")
            
            # Generate predictions for unknown variables
            unknown_vars = [name for name, node in self.network.nodes.items() if node.evidence is None]
            if unknown_vars:
                samples = await self.network.forward_sampling(1000)
                
                for var in unknown_vars:
                    # Calculate marginal probabilities
                    var_values = [sample.get(var) for sample in samples if var in sample]
                    if var_values:
                        unique_values = list(set(var_values))
                        for value in unique_values:
                            prob = var_values.count(value) / len(var_values)
                            
                            hypothesis = Hypothesis(
                                statement=f"{var} = {value}",
                                probability=prob,
                                metadata={'variable': var, 'value': value}
                            )
                            result.hypotheses.append(hypothesis)
                
                result.reasoning_trace.append(f"Generated predictions for {len(unknown_vars)} variables")
            
            # Calculate overall confidence
            if result.conclusions:
                result.confidence = sum(c.confidence for c in result.conclusions) / len(result.conclusions)
            elif result.hypotheses:
                result.confidence = sum(h.probability for h in result.hypotheses) / len(result.hypotheses)
            else:
                result.confidence = 0.0
            
            result.success = len(result.conclusions) > 0 or len(result.hypotheses) > 0
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.reasoning_trace.append(f"Error: {e}")
        
        return result
    
    async def build_network_from_context(self, context: ReasoningContext) -> None:
        """Build Bayesian network from context information."""
        # Extract network structure from prior knowledge
        if 'bayesian_network' in context.prior_knowledge:
            network_data = context.prior_knowledge['bayesian_network']
            
            # Add nodes
            for node_data in network_data.get('nodes', []):
                node = await self.create_node_from_data(node_data)
                await self.network.add_node(node)
        else:
            # Create simple network from evidence
            variables = set()
            for evidence in context.available_evidence:
                if isinstance(evidence.content, dict) and 'variable' in evidence.content:
                    variables.add(evidence.content['variable'])
            
            # Create independent nodes
            for var in variables:
                node = BayesianNode(
                    name=var,
                    distribution=ProbabilityDistribution(
                        name=var,
                        distribution_type='categorical',
                        values=[True, False],
                        probabilities=[0.5, 0.5]
                    )
                )
                await self.network.add_node(node)
    
    async def create_node_from_data(self, node_data: Dict[str, Any]) -> BayesianNode:
        """Create a Bayesian node from data."""
        name = node_data['name']
        parents = node_data.get('parents', [])
        
        # Create distribution
        dist_data = node_data.get('distribution', {})
        distribution = ProbabilityDistribution(
            name=name,
            distribution_type=dist_data.get('type', 'categorical'),
            values=dist_data.get('values', [True, False]),
            probabilities=dist_data.get('probabilities', [0.5, 0.5])
        )
        
        node = BayesianNode(
            name=name,
            parents=parents,
            distribution=distribution
        )
        
        # Add conditional probabilities if provided
        if 'conditional_probabilities' in node_data:
            for key_str, cond_dist_data in node_data['conditional_probabilities'].items():
                key = tuple(key_str.split(',')) if isinstance(key_str, str) else key_str
                cond_dist = ProbabilityDistribution(
                    name=f"{name}_cond",
                    distribution_type=cond_dist_data.get('type', 'categorical'),
                    values=cond_dist_data.get('values', [True, False]),
                    probabilities=cond_dist_data.get('probabilities', [0.5, 0.5])
                )
                node.conditional_probabilities[key] = cond_dist
        
        return node
    
    async def parse_query(self, query_string: str) -> Optional[Dict[str, Any]]:
        """Parse a query string into variable assignments."""
        # Simple parsing - can be enhanced
        query = {}
        
        # Look for patterns like "X = value"
        assignments = query_string.split(' AND ')
        for assignment in assignments:
            if '=' in assignment:
                var, value = assignment.split('=', 1)
                var = var.strip()
                value = value.strip()
                
                # Try to convert value
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                
                query[var] = value
        
        return query if query else None


class ProbabilisticReasoner(BaseReasoner):
    """Main probabilistic reasoning engine combining multiple approaches."""
    
    def __init__(self, name: str = "ProbabilisticReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.PROBABILISTIC, config)
        self.bayesian_reasoner = BayesianNetworkReasoner("BayesianNetwork", config)
        self.mcmc_reasoner = MCMCReasoner(config)
        self.variational_reasoner = VariationalReasoner(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform probabilistic reasoning using the most appropriate method."""
        # Determine best method based on context
        if 'bayesian_network' in context.prior_knowledge:
            return await self.bayesian_reasoner.reason(context)
        elif 'mcmc_config' in context.metadata:
            return await self.mcmc_reasoning(context)
        elif 'variational_config' in context.metadata:
            return await self.variational_reasoning(context)
        else:
            # Default to Bayesian network reasoning
            return await self.bayesian_reasoner.reason(context)
    
    async def mcmc_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """Perform MCMC-based reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Extract MCMC configuration
            mcmc_config = context.metadata.get('mcmc_config', {})
            
            # Define target distribution based on evidence
            def target_distribution(state):
                # Simple scoring based on evidence consistency
                score = 1.0
                for evidence in context.available_evidence:
                    if isinstance(evidence.content, dict):
                        var = evidence.content.get('variable')
                        expected_value = evidence.content.get('value')
                        if var in state and state[var] == expected_value:
                            score *= evidence.confidence
                        else:
                            score *= (1 - evidence.confidence)
                return score
            
            # Define proposal distribution
            def proposal_distribution(current_state):
                new_state = current_state.copy()
                # Randomly flip one boolean variable
                bool_vars = [k for k, v in new_state.items() if isinstance(v, bool)]
                if bool_vars:
                    var_to_flip = random.choice(bool_vars)
                    new_state[var_to_flip] = not new_state[var_to_flip]
                return new_state
            
            # Initial state
            initial_state = {}
            for evidence in context.available_evidence:
                if isinstance(evidence.content, dict) and 'variable' in evidence.content:
                    var = evidence.content['variable']
                    if var not in initial_state:
                        initial_state[var] = random.choice([True, False])
            
            if initial_state:
                samples, acceptance_rate = await self.mcmc_reasoner.metropolis_hastings(
                    target_distribution, proposal_distribution, initial_state
                )
                
                result.reasoning_trace.append(f"MCMC sampling: {len(samples)} samples, acceptance rate: {acceptance_rate:.3f}")
                
                # Analyze samples
                for var in initial_state.keys():
                    var_values = [sample[var] for sample in samples]
                    true_count = sum(var_values)
                    prob_true = true_count / len(samples)
                    
                    hypothesis = Hypothesis(
                        statement=f"{var} = True",
                        probability=prob_true,
                        metadata={'method': 'mcmc', 'samples': len(samples)}
                    )
                    result.hypotheses.append(hypothesis)
                
                result.confidence = acceptance_rate
                result.success = True
            else:
                result.success = False
                result.error_message = "No variables found for MCMC sampling"
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.reasoning_trace.append(f"Error: {e}")
        
        return result
    
    async def variational_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """Perform variational inference."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Extract variational configuration
            var_config = context.metadata.get('variational_config', {})
            
            # Define target log probability
            def target_log_prob(state_dict):
                log_prob = 0.0
                for evidence in context.available_evidence:
                    if isinstance(evidence.content, dict):
                        var = evidence.content.get('variable')
                        expected_value = evidence.content.get('value')
                        if var in state_dict:
                            if isinstance(expected_value, (int, float)):
                                # Gaussian likelihood
                                diff = state_dict[var] - expected_value
                                log_prob += -0.5 * diff ** 2
                            else:
                                # Categorical likelihood
                                if state_dict[var] == expected_value:
                                    log_prob += math.log(evidence.confidence + 1e-10)
                                else:
                                    log_prob += math.log(1 - evidence.confidence + 1e-10)
                return log_prob
            
            # Get variables and initial parameters
            variables = []
            initial_params = {}
            
            for evidence in context.available_evidence:
                if isinstance(evidence.content, dict) and 'variable' in evidence.content:
                    var = evidence.content['variable']
                    if var not in variables:
                        variables.append(var)
                        initial_params[var] = {
                            'mean': 0.0,
                            'variance': 1.0
                        }
            
            if variables:
                distributions = await self.variational_reasoner.mean_field_approximation(
                    target_log_prob, variables, initial_params
                )
                
                result.reasoning_trace.append(f"Variational inference for {len(variables)} variables")
                
                # Generate conclusions from distributions
                for var, dist in distributions.items():
                    mean = dist.parameters['mean']
                    std = dist.parameters['std']
                    
                    conclusion = Conclusion(
                        statement=f"{var} ~ N({mean:.3f}, {std:.3f})",
                        confidence=0.8,  # Variational approximation uncertainty
                        reasoning_type=self.reasoning_type,
                        reasoning_chain=["Variational mean field approximation"]
                    )
                    result.conclusions.append(conclusion)
                
                result.confidence = 0.8
                result.success = True
            else:
                result.success = False
                result.error_message = "No variables found for variational inference"
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.reasoning_trace.append(f"Error: {e}")
        
        return result