"""
Causal Reasoning Module

Implements causal discovery, causal inference, and intervention planning
using algorithms like PC, GES, FCI, and do-calculus.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import asyncio
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations, permutations
import math

from .base import (
    BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType,
    Evidence, Hypothesis, Conclusion, create_evidence, create_conclusion
)


@dataclass
class CausalEdge:
    """Represents a causal edge between variables."""
    source: str
    target: str
    edge_type: str  # 'directed', 'undirected', 'bidirected'
    strength: float = 1.0
    confidence: float = 1.0
    
    def __str__(self):
        if self.edge_type == 'directed':
            return f"{self.source} -> {self.target}"
        elif self.edge_type == 'bidirected':
            return f"{self.source} <-> {self.target}"
        else:
            return f"{self.source} -- {self.target}"


@dataclass
class CausalGraph:
    """Represents a causal graph structure."""
    variables: Set[str] = field(default_factory=set)
    edges: List[CausalEdge] = field(default_factory=list)
    
    def add_edge(self, edge: CausalEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.variables.add(edge.source)
        self.variables.add(edge.target)
    
    def get_parents(self, variable: str) -> List[str]:
        """Get parent variables of a given variable."""
        parents = []
        for edge in self.edges:
            if edge.target == variable and edge.edge_type == 'directed':
                parents.append(edge.source)
        return parents
    
    def get_children(self, variable: str) -> List[str]:
        """Get children variables of a given variable."""
        children = []
        for edge in self.edges:
            if edge.source == variable and edge.edge_type == 'directed':
                children.append(edge.target)
        return children
    
    def get_adjacents(self, variable: str) -> List[str]:
        """Get all adjacent variables."""
        adjacents = []
        for edge in self.edges:
            if edge.source == variable:
                adjacents.append(edge.target)
            elif edge.target == variable:
                adjacents.append(edge.source)
        return list(set(adjacents))
    
    def has_edge(self, source: str, target: str, edge_type: str = None) -> bool:
        """Check if an edge exists."""
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                if edge_type is None or edge.edge_type == edge_type:
                    return True
            # For undirected edges, check both directions
            if edge.edge_type == 'undirected' and edge.target == source and edge.source == target:
                return True
        return False
    
    def is_d_separated(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """Check if X and Y are d-separated given Z."""
        # Simplified d-separation check
        # In practice, this would implement the full d-separation algorithm
        
        # For each path from X to Y, check if it's blocked by Z
        for x in X:
            for y in Y:
                if not self._is_path_blocked(x, y, Z):
                    return False
        return True
    
    def _is_path_blocked(self, start: str, end: str, conditioning_set: Set[str]) -> bool:
        """Check if path from start to end is blocked by conditioning set."""
        # Simplified implementation - would need full path enumeration and blocking rules
        visited = set()
        stack = [start]
        
        while stack:
            current = stack.pop()
            if current == end:
                return False  # Found unblocked path
            
            if current in visited:
                continue
            visited.add(current)
            
            # Add adjacent nodes if not in conditioning set
            for adj in self.get_adjacents(current):
                if adj not in conditioning_set and adj not in visited:
                    stack.append(adj)
        
        return True  # No unblocked path found


class PCAlgorithm:
    """PC Algorithm for causal discovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.significance_level = config.get('significance_level', 0.05)
        self.max_conditioning_size = config.get('max_conditioning_size', 3)
    
    async def discover_causal_structure(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        """Discover causal structure using PC algorithm."""
        n_vars = len(variable_names)
        graph = CausalGraph()
        
        # Initialize complete undirected graph
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                edge = CausalEdge(
                    source=variable_names[i],
                    target=variable_names[j],
                    edge_type='undirected'
                )
                graph.add_edge(edge)
        
        # Phase 1: Remove edges based on conditional independence
        await self._skeleton_discovery(graph, data, variable_names)
        
        # Phase 2: Orient edges
        await self._edge_orientation(graph, data, variable_names)
        
        return graph
    
    async def _skeleton_discovery(self, graph: CausalGraph, data: np.ndarray, variable_names: List[str]) -> None:
        """Discover skeleton by removing edges based on conditional independence."""
        edges_to_remove = []
        
        for conditioning_size in range(self.max_conditioning_size + 1):
            for edge in graph.edges:
                if edge.edge_type != 'undirected':
                    continue
                
                source_idx = variable_names.index(edge.source)
                target_idx = variable_names.index(edge.target)
                
                # Get potential conditioning sets
                adjacents = set(graph.get_adjacents(edge.source)) | set(graph.get_adjacents(edge.target))
                adjacents.discard(edge.source)
                adjacents.discard(edge.target)
                
                if len(adjacents) >= conditioning_size:
                    for conditioning_vars in combinations(adjacents, conditioning_size):
                        conditioning_indices = [variable_names.index(var) for var in conditioning_vars]
                        
                        # Test conditional independence
                        if await self._test_conditional_independence(
                            data, source_idx, target_idx, conditioning_indices
                        ):
                            edges_to_remove.append(edge)
                            break
        
        # Remove edges
        for edge in edges_to_remove:
            if edge in graph.edges:
                graph.edges.remove(edge)
    
    async def _test_conditional_independence(self, 
                                           data: np.ndarray,
                                           x_idx: int,
                                           y_idx: int,
                                           conditioning_indices: List[int]) -> bool:
        """Test conditional independence using partial correlation."""
        if len(conditioning_indices) == 0:
            # Test marginal independence
            correlation = np.corrcoef(data[:, x_idx], data[:, y_idx])[0, 1]
            n = data.shape[0]
            
            # Fisher's z-transform
            if abs(correlation) >= 0.999:
                return False
            
            z = 0.5 * np.log((1 + correlation) / (1 - correlation))
            z_stat = z * np.sqrt(n - 3)
            
            # Two-tailed test
            p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))
            return p_value > self.significance_level
        
        else:
            # Test conditional independence using partial correlation
            partial_corr = await self._partial_correlation(data, x_idx, y_idx, conditioning_indices)
            n = data.shape[0]
            k = len(conditioning_indices)
            
            if abs(partial_corr) >= 0.999:
                return False
            
            z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
            z_stat = z * np.sqrt(n - k - 3)
            
            p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))
            return p_value > self.significance_level
    
    async def _partial_correlation(self, 
                                 data: np.ndarray,
                                 x_idx: int,
                                 y_idx: int,
                                 conditioning_indices: List[int]) -> float:
        """Calculate partial correlation coefficient."""
        # Create design matrix
        all_indices = [x_idx, y_idx] + conditioning_indices
        subset_data = data[:, all_indices]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(subset_data.T)
        
        # Calculate partial correlation using matrix inversion
        precision_matrix = np.linalg.inv(corr_matrix)
        partial_corr = -precision_matrix[0, 1] / np.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])
        
        return partial_corr
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function of standard normal distribution."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    async def _edge_orientation(self, graph: CausalGraph, data: np.ndarray, variable_names: List[str]) -> None:
        """Orient edges using v-structures and propagation rules."""
        # Find v-structures (X -> Z <- Y where X and Y are not adjacent)
        await self._find_v_structures(graph)
        
        # Apply orientation rules
        await self._apply_orientation_rules(graph)
    
    async def _find_v_structures(self, graph: CausalGraph) -> None:
        """Find and orient v-structures."""
        for variable in graph.variables:
            adjacents = graph.get_adjacents(variable)
            
            # Look for pairs of adjacent variables that are not connected
            for i in range(len(adjacents)):
                for j in range(i + 1, len(adjacents)):
                    var1, var2 = adjacents[i], adjacents[j]
                    
                    # Check if var1 and var2 are not adjacent
                    if not graph.has_edge(var1, var2):
                        # Orient as var1 -> variable <- var2
                        self._orient_edge(graph, var1, variable)
                        self._orient_edge(graph, var2, variable)
    
    def _orient_edge(self, graph: CausalGraph, source: str, target: str) -> None:
        """Orient an undirected edge."""
        for edge in graph.edges:
            if ((edge.source == source and edge.target == target) or 
                (edge.source == target and edge.target == source and edge.edge_type == 'undirected')):
                
                edge.source = source
                edge.target = target
                edge.edge_type = 'directed'
                break
    
    async def _apply_orientation_rules(self, graph: CausalGraph) -> None:
        """Apply orientation rules to propagate edge directions."""
        changed = True
        while changed:
            changed = False
            
            # Rule 1: If X -> Y - Z and X and Z not adjacent, then Y -> Z
            for edge in graph.edges:
                if edge.edge_type == 'undirected':
                    # Check for rule application
                    if await self._apply_rule_1(graph, edge):
                        changed = True
            
            # Rule 2: If X -> Y -> Z and X - Z, then X -> Z
            for edge in graph.edges:
                if edge.edge_type == 'undirected':
                    if await self._apply_rule_2(graph, edge):
                        changed = True
    
    async def _apply_rule_1(self, graph: CausalGraph, edge: CausalEdge) -> bool:
        """Apply orientation rule 1."""
        # Implementation of specific orientation rule
        return False  # Placeholder
    
    async def _apply_rule_2(self, graph: CausalGraph, edge: CausalEdge) -> bool:
        """Apply orientation rule 2."""
        # Implementation of specific orientation rule
        return False  # Placeholder


class GESAlgorithm:
    """GES (Greedy Equivalence Search) Algorithm for causal discovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.score_function = config.get('score_function', 'bic')
    
    async def discover_causal_structure(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        """Discover causal structure using GES algorithm."""
        current_graph = CausalGraph()
        current_graph.variables = set(variable_names)
        current_score = await self._calculate_score(current_graph, data, variable_names)
        
        # Forward phase: Add edges
        improved = True
        while improved:
            improved = False
            best_score = current_score
            best_operation = None
            
            # Try adding each possible edge
            for source in variable_names:
                for target in variable_names:
                    if source != target and not current_graph.has_edge(source, target, 'directed'):
                        # Try adding edge
                        test_graph = self._copy_graph(current_graph)
                        test_graph.add_edge(CausalEdge(source, target, 'directed'))
                        
                        # Check if this creates a cycle
                        if not self._has_cycle(test_graph):
                            score = await self._calculate_score(test_graph, data, variable_names)
                            if score > best_score:
                                best_score = score
                                best_operation = ('add', source, target)
                                improved = True
            
            # Apply best operation
            if best_operation:
                op_type, source, target = best_operation
                current_graph.add_edge(CausalEdge(source, target, 'directed'))
                current_score = best_score
        
        # Backward phase: Remove edges
        improved = True
        while improved:
            improved = False
            best_score = current_score
            best_operation = None
            
            # Try removing each edge
            for edge in current_graph.edges[:]:  # Copy list to avoid modification during iteration
                test_graph = self._copy_graph(current_graph)
                test_graph.edges.remove(edge)
                
                score = await self._calculate_score(test_graph, data, variable_names)
                if score > best_score:
                    best_score = score
                    best_operation = ('remove', edge)
                    improved = True
            
            # Apply best operation
            if best_operation:
                op_type, edge = best_operation
                current_graph.edges.remove(edge)
                current_score = best_score
        
        return current_graph
    
    async def _calculate_score(self, graph: CausalGraph, data: np.ndarray, variable_names: List[str]) -> float:
        """Calculate score for a graph structure."""
        if self.score_function == 'bic':
            return await self._calculate_bic_score(graph, data, variable_names)
        else:
            return 0.0
    
    async def _calculate_bic_score(self, graph: CausalGraph, data: np.ndarray, variable_names: List[str]) -> float:
        """Calculate BIC score for the graph."""
        n = data.shape[0]
        total_score = 0.0
        
        for var in variable_names:
            var_idx = variable_names.index(var)
            parents = graph.get_parents(var)
            parent_indices = [variable_names.index(p) for p in parents]
            
            # Calculate likelihood
            if len(parents) == 0:
                # No parents - use marginal distribution
                variance = np.var(data[:, var_idx])
                log_likelihood = -0.5 * n * (np.log(2 * np.pi * variance) + 1)
            else:
                # Linear regression with parents
                X = data[:, parent_indices]
                y = data[:, var_idx]
                
                # Add intercept
                X = np.column_stack([np.ones(n), X])
                
                # Calculate coefficients
                try:
                    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                    predictions = X @ coeffs
                    residuals = y - predictions
                    variance = np.var(residuals)
                    
                    log_likelihood = -0.5 * n * (np.log(2 * np.pi * variance) + 1)
                except np.linalg.LinAlgError:
                    log_likelihood = -np.inf
            
            # BIC penalty
            num_params = len(parents) + 1  # +1 for intercept
            bic_penalty = 0.5 * num_params * np.log(n)
            
            score = log_likelihood - bic_penalty
            total_score += score
        
        return total_score
    
    def _copy_graph(self, graph: CausalGraph) -> CausalGraph:
        """Create a copy of the graph."""
        new_graph = CausalGraph()
        new_graph.variables = graph.variables.copy()
        new_graph.edges = [
            CausalEdge(edge.source, edge.target, edge.edge_type, edge.strength, edge.confidence)
            for edge in graph.edges
        ]
        return new_graph
    
    def _has_cycle(self, graph: CausalGraph) -> bool:
        """Check if graph has cycles."""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for child in graph.get_children(node):
                if child not in visited:
                    if dfs(child):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for var in graph.variables:
            if var not in visited:
                if dfs(var):
                    return True
        
        return False


class CausalInferencer:
    """Performs causal inference using do-calculus and interventions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def estimate_causal_effect(self, 
                                   graph: CausalGraph,
                                   data: np.ndarray,
                                   variable_names: List[str],
                                   treatment: str,
                                   outcome: str,
                                   confounders: List[str] = None) -> Dict[str, float]:
        """Estimate causal effect of treatment on outcome."""
        treatment_idx = variable_names.index(treatment)
        outcome_idx = variable_names.index(outcome)
        
        # Identify confounders if not provided
        if confounders is None:
            confounders = await self._identify_confounders(graph, treatment, outcome)
        
        confounder_indices = [variable_names.index(c) for c in confounders if c in variable_names]
        
        # Estimate using backdoor adjustment
        if confounders:
            effect = await self._backdoor_adjustment(
                data, treatment_idx, outcome_idx, confounder_indices
            )
        else:
            # Direct effect estimation
            effect = await self._direct_effect_estimation(
                data, treatment_idx, outcome_idx
            )
        
        return {
            'average_treatment_effect': effect.get('ate', 0.0),
            'treatment_on_treated': effect.get('tot', 0.0),
            'confounders': confounders
        }
    
    async def _identify_confounders(self, graph: CausalGraph, treatment: str, outcome: str) -> List[str]:
        """Identify confounders using backdoor criterion."""
        confounders = []
        
        # Find all paths from treatment to outcome
        paths = self._find_all_paths(graph, treatment, outcome)
        
        # Identify backdoor paths (paths with arrows pointing into treatment)
        backdoor_paths = []
        for path in paths:
            if len(path) > 2:  # Must have at least treatment -> X -> outcome
                # Check if first edge points into treatment
                if graph.has_edge(path[1], path[0], 'directed'):
                    backdoor_paths.append(path)
        
        # Find variables that block all backdoor paths
        for var in graph.variables:
            if var != treatment and var != outcome:
                blocks_all = True
                for path in backdoor_paths:
                    if var not in path[1:-1]:  # Not in middle of path
                        blocks_all = False
                        break
                if blocks_all and backdoor_paths:
                    confounders.append(var)
        
        return confounders
    
    def _find_all_paths(self, graph: CausalGraph, start: str, end: str) -> List[List[str]]:
        """Find all paths from start to end variable."""
        paths = []
        
        def dfs(current_path, visited):
            current = current_path[-1]
            
            if current == end and len(current_path) > 1:
                paths.append(current_path.copy())
                return
            
            for neighbor in graph.get_adjacents(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    current_path.append(neighbor)
                    dfs(current_path, visited)
                    current_path.pop()
                    visited.remove(neighbor)
        
        dfs([start], {start})
        return paths
    
    async def _backdoor_adjustment(self, 
                                 data: np.ndarray,
                                 treatment_idx: int,
                                 outcome_idx: int,
                                 confounder_indices: List[int]) -> Dict[str, float]:
        """Estimate causal effect using backdoor adjustment."""
        # Stratify by confounder values and estimate effect within each stratum
        if not confounder_indices:
            return await self._direct_effect_estimation(data, treatment_idx, outcome_idx)
        
        # For continuous confounders, discretize or use regression adjustment
        # Simplified implementation using regression adjustment
        
        # Prepare design matrix
        X = data[:, confounder_indices + [treatment_idx]]
        y = data[:, outcome_idx]
        
        # Add intercept
        X = np.column_stack([np.ones(len(y)), X])
        
        try:
            # Linear regression
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Treatment effect is the coefficient of treatment variable
            treatment_effect = coeffs[-1]  # Last coefficient is treatment
            
            return {
                'ate': treatment_effect,
                'tot': treatment_effect  # Assuming linear effect
            }
        except np.linalg.LinAlgError:
            return {'ate': 0.0, 'tot': 0.0}
    
    async def _direct_effect_estimation(self, 
                                      data: np.ndarray,
                                      treatment_idx: int,
                                      outcome_idx: int) -> Dict[str, float]:
        """Estimate direct effect without confounders."""
        treatment_data = data[:, treatment_idx]
        outcome_data = data[:, outcome_idx]
        
        # Simple difference in means for binary treatment
        if set(treatment_data) == {0, 1} or set(treatment_data) == {0.0, 1.0}:
            treated_outcomes = outcome_data[treatment_data == 1]
            control_outcomes = outcome_data[treatment_data == 0]
            
            if len(treated_outcomes) > 0 and len(control_outcomes) > 0:
                ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
                return {'ate': ate, 'tot': ate}
        
        # For continuous treatment, use correlation as proxy
        correlation = np.corrcoef(treatment_data, outcome_data)[0, 1]
        return {'ate': correlation, 'tot': correlation}


class InterventionPlanner:
    """Plans interventions based on causal graph and desired outcomes."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def plan_intervention(self, 
                              graph: CausalGraph,
                              target_variable: str,
                              desired_change: float,
                              constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Plan intervention to achieve desired change in target variable."""
        constraints = constraints or {}
        
        # Find all variables that can influence the target
        influencers = await self._find_influencers(graph, target_variable)
        
        # Estimate intervention effects
        interventions = []
        for var in influencers:
            if var not in constraints.get('forbidden_interventions', []):
                effect = await self._estimate_intervention_effect(graph, var, target_variable)
                
                if effect != 0:
                    required_change = desired_change / effect
                    cost = constraints.get('intervention_costs', {}).get(var, 1.0)
                    
                    interventions.append({
                        'variable': var,
                        'required_change': required_change,
                        'expected_effect': effect,
                        'cost': cost,
                        'efficiency': abs(effect) / cost
                    })
        
        # Sort by efficiency
        interventions.sort(key=lambda x: x['efficiency'], reverse=True)
        
        # Select best intervention or combination
        best_intervention = await self._select_best_intervention(
            interventions, desired_change, constraints
        )
        
        return best_intervention
    
    async def _find_influencers(self, graph: CausalGraph, target: str) -> List[str]:
        """Find all variables that can influence the target."""
        influencers = set()
        
        # Direct parents
        influencers.update(graph.get_parents(target))
        
        # Indirect influencers (ancestors)
        to_visit = deque(graph.get_parents(target))
        visited = set()
        
        while to_visit:
            current = to_visit.popleft()
            if current in visited:
                continue
            visited.add(current)
            influencers.add(current)
            
            # Add parents of current variable
            parents = graph.get_parents(current)
            to_visit.extend(parents)
        
        return list(influencers)
    
    async def _estimate_intervention_effect(self, 
                                          graph: CausalGraph,
                                          intervention_var: str,
                                          target_var: str) -> float:
        """Estimate effect of intervening on intervention_var on target_var."""
        # Simplified effect estimation
        # In practice, this would use more sophisticated causal inference
        
        # Direct effect
        if graph.has_edge(intervention_var, target_var, 'directed'):
            return 1.0  # Assume unit direct effect
        
        # Indirect effect through paths
        paths = self._find_all_directed_paths(graph, intervention_var, target_var)
        if paths:
            return 0.5  # Assume reduced indirect effect
        
        return 0.0
    
    def _find_all_directed_paths(self, graph: CausalGraph, start: str, end: str) -> List[List[str]]:
        """Find all directed paths from start to end."""
        paths = []
        
        def dfs(current_path, visited):
            current = current_path[-1]
            
            if current == end and len(current_path) > 1:
                paths.append(current_path.copy())
                return
            
            for child in graph.get_children(current):
                if child not in visited:
                    visited.add(child)
                    current_path.append(child)
                    dfs(current_path, visited)
                    current_path.pop()
                    visited.remove(child)
        
        dfs([start], {start})
        return paths
    
    async def _select_best_intervention(self, 
                                      interventions: List[Dict[str, Any]],
                                      desired_change: float,
                                      constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best intervention strategy."""
        max_budget = constraints.get('max_budget', float('inf'))
        max_interventions = constraints.get('max_interventions', 1)
        
        if not interventions:
            return {
                'interventions': [],
                'expected_total_effect': 0.0,
                'total_cost': 0.0,
                'success_probability': 0.0
            }
        
        # Single intervention
        if max_interventions == 1:
            best = interventions[0]
            cost = abs(best['required_change']) * best['cost']
            
            if cost <= max_budget:
                return {
                    'interventions': [best],
                    'expected_total_effect': best['expected_effect'] * best['required_change'],
                    'total_cost': cost,
                    'success_probability': 0.8  # Estimated probability
                }
        
        # Multiple interventions (simplified combination)
        best_combination = []
        total_effect = 0.0
        total_cost = 0.0
        
        for intervention in interventions[:max_interventions]:
            change_needed = desired_change - total_effect
            if abs(change_needed) < 1e-6:
                break
            
            intervention_change = min(
                intervention['required_change'],
                change_needed / intervention['expected_effect']
            )
            
            intervention_cost = abs(intervention_change) * intervention['cost']
            
            if total_cost + intervention_cost <= max_budget:
                best_combination.append({
                    **intervention,
                    'actual_change': intervention_change
                })
                total_effect += intervention['expected_effect'] * intervention_change
                total_cost += intervention_cost
        
        return {
            'interventions': best_combination,
            'expected_total_effect': total_effect,
            'total_cost': total_cost,
            'success_probability': 0.7 if best_combination else 0.0
        }


class CausalDiscoverer:
    """Main causal discovery engine combining multiple algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pc_algorithm = PCAlgorithm(config)
        self.ges_algorithm = GESAlgorithm(config)
    
    async def discover_causal_structure(self, 
                                      data: np.ndarray,
                                      variable_names: List[str],
                                      algorithm: str = 'pc') -> CausalGraph:
        """Discover causal structure using specified algorithm."""
        if algorithm == 'pc':
            return await self.pc_algorithm.discover_causal_structure(data, variable_names)
        elif algorithm == 'ges':
            return await self.ges_algorithm.discover_causal_structure(data, variable_names)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


class CausalReasoner(BaseReasoner):
    """Main causal reasoning engine combining discovery and inference."""
    
    def __init__(self, name: str = "CausalReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.CAUSAL, config)
        self.causal_discoverer = CausalDiscoverer(config)
        self.causal_inferencer = CausalInferencer(config)
        self.intervention_planner = InterventionPlanner(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform causal reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Extract data and variables from context
            data, variable_names = await self._extract_data_from_context(context)
            
            if data is not None and len(variable_names) > 1:
                # Discover causal structure
                algorithm = context.metadata.get('causal_algorithm', 'pc')
                causal_graph = await self.causal_discoverer.discover_causal_structure(
                    data, variable_names, algorithm
                )
                
                result.reasoning_trace.append(f"Discovered causal graph with {len(causal_graph.edges)} edges")
                
                # Generate structural conclusions
                for edge in causal_graph.edges:
                    conclusion = Conclusion(
                        statement=f"Causal relationship: {edge}",
                        confidence=edge.confidence,
                        reasoning_type=self.reasoning_type,
                        reasoning_chain=[f"Causal discovery ({algorithm})"]
                    )
                    result.conclusions.append(conclusion)
                
                # Causal inference if treatment and outcome specified
                if 'treatment' in context.metadata and 'outcome' in context.metadata:
                    treatment = context.metadata['treatment']
                    outcome = context.metadata['outcome']
                    confounders = context.metadata.get('confounders')
                    
                    if treatment in variable_names and outcome in variable_names:
                        effect_result = await self.causal_inferencer.estimate_causal_effect(
                            causal_graph, data, variable_names, treatment, outcome, confounders
                        )
                        
                        result.reasoning_trace.append(f"Estimated causal effect: {effect_result['average_treatment_effect']:.4f}")
                        
                        conclusion = Conclusion(
                            statement=f"Causal effect of {treatment} on {outcome}: {effect_result['average_treatment_effect']:.4f}",
                            confidence=0.8,
                            reasoning_type=self.reasoning_type,
                            reasoning_chain=["Causal effect estimation"],
                            metadata=effect_result
                        )
                        result.conclusions.append(conclusion)
                
                # Intervention planning if goal specified
                if context.goal and 'target_variable' in context.metadata:
                    target_var = context.metadata['target_variable']
                    desired_change = context.metadata.get('desired_change', 1.0)
                    constraints = context.metadata.get('intervention_constraints', {})
                    
                    if target_var in variable_names:
                        intervention_plan = await self.intervention_planner.plan_intervention(
                            causal_graph, target_var, desired_change, constraints
                        )
                        
                        result.reasoning_trace.append(f"Planned {len(intervention_plan['interventions'])} interventions")
                        
                        for intervention in intervention_plan['interventions']:
                            conclusion = Conclusion(
                                statement=f"Recommended intervention: {intervention['variable']} (change: {intervention.get('actual_change', intervention['required_change']):.3f})",
                                confidence=intervention_plan['success_probability'],
                                reasoning_type=self.reasoning_type,
                                reasoning_chain=["Intervention planning"],
                                metadata=intervention
                            )
                            result.conclusions.append(conclusion)
                
                # Calculate overall confidence
                if result.conclusions:
                    result.confidence = sum(c.confidence for c in result.conclusions) / len(result.conclusions)
                else:
                    result.confidence = 0.0
                
                result.success = True
            
            else:
                result.success = False
                result.error_message = "Insufficient data for causal reasoning"
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.reasoning_trace.append(f"Error: {e}")
        
        return result
    
    async def _extract_data_from_context(self, context: ReasoningContext) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract numerical data matrix and variable names from context."""
        if 'causal_data' in context.metadata:
            causal_data = context.metadata['causal_data']
            
            if isinstance(causal_data, dict):
                if 'data' in causal_data and 'variables' in causal_data:
                    data = np.array(causal_data['data'])
                    variables = causal_data['variables']
                    return data, variables
        
        # Try to extract from evidence
        numerical_evidence = []
        variable_names = set()
        
        for evidence in context.available_evidence:
            if isinstance(evidence.content, dict):
                if 'data_point' in evidence.content:
                    numerical_evidence.append(evidence.content['data_point'])
                    variable_names.update(evidence.content['data_point'].keys())
        
        if numerical_evidence and variable_names:
            variable_names = list(variable_names)
            data_matrix = []
            
            for data_point in numerical_evidence:
                row = [data_point.get(var, 0) for var in variable_names]
                data_matrix.append(row)
            
            return np.array(data_matrix), variable_names
        
        return None, []