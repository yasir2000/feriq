"""
Deductive Reasoning Module

Implements logical inference, theorem proving, and rule-based reasoning.
Supports forward chaining, backward chaining, and automated theorem proving.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import asyncio
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .base import (
    BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType,
    Evidence, Hypothesis, Conclusion, create_evidence, create_conclusion
)


@dataclass
class LogicalStatement:
    """Represents a logical statement or proposition."""
    id: str
    statement: str
    variables: Set[str] = field(default_factory=set)
    predicates: Set[str] = field(default_factory=set)
    is_fact: bool = False
    is_rule: bool = False
    confidence: float = 1.0
    
    def __post_init__(self):
        self.extract_components()
    
    def extract_components(self):
        """Extract variables and predicates from statement."""
        # Simple extraction - can be enhanced with proper parsing
        variables = re.findall(r'\b[a-z][a-zA-Z0-9_]*\b', self.statement)
        predicates = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', self.statement)
        
        self.variables = set(variables)
        self.predicates = set(predicates)


@dataclass
class Rule:
    """Represents a logical rule (if-then statement)."""
    id: str
    antecedent: List[LogicalStatement]  # premises
    consequent: LogicalStatement        # conclusion
    confidence: float = 1.0
    priority: int = 0
    
    def __str__(self):
        premises = " AND ".join(str(stmt.statement) for stmt in self.antecedent)
        return f"IF {premises} THEN {self.consequent.statement}"


class LogicalInferenceEngine:
    """Core logical inference engine supporting various reasoning methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.facts = {}  # statement_id -> LogicalStatement
        self.rules = {}  # rule_id -> Rule
        self.derived_facts = {}
        self.inference_chain = []
    
    async def add_fact(self, statement: LogicalStatement) -> None:
        """Add a fact to the knowledge base."""
        statement.is_fact = True
        self.facts[statement.id] = statement
    
    async def add_rule(self, rule: Rule) -> None:
        """Add a rule to the knowledge base."""
        self.rules[rule.id] = rule
    
    async def forward_chaining(self, max_iterations: int = 100) -> List[LogicalStatement]:
        """Apply forward chaining inference."""
        new_facts = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            facts_added_this_iteration = 0
            
            for rule_id, rule in self.rules.items():
                if await self.can_apply_rule(rule):
                    new_fact = await self.apply_rule(rule)
                    if new_fact and new_fact.id not in self.facts:
                        await self.add_fact(new_fact)
                        new_facts.append(new_fact)
                        facts_added_this_iteration += 1
                        
                        self.inference_chain.append({
                            'type': 'forward_chaining',
                            'rule': str(rule),
                            'derived_fact': new_fact.statement,
                            'iteration': iteration
                        })
            
            # Stop if no new facts were derived
            if facts_added_this_iteration == 0:
                break
        
        return new_facts
    
    async def backward_chaining(self, goal: LogicalStatement) -> bool:
        """Apply backward chaining to prove a goal."""
        return await self.prove_goal(goal, set())
    
    async def prove_goal(self, goal: LogicalStatement, visited: Set[str]) -> bool:
        """Recursively prove a goal using backward chaining."""
        # Avoid cycles
        if goal.statement in visited:
            return False
        
        visited.add(goal.statement)
        
        # Check if goal is already a known fact
        for fact in self.facts.values():
            if await self.statements_match(goal, fact):
                self.inference_chain.append({
                    'type': 'backward_chaining',
                    'goal': goal.statement,
                    'result': 'fact_found'
                })
                return True
        
        # Try to prove goal using rules
        for rule in self.rules.values():
            if await self.statements_match(goal, rule.consequent):
                # Try to prove all antecedents
                all_antecedents_proven = True
                for antecedent in rule.antecedent:
                    if not await self.prove_goal(antecedent, visited.copy()):
                        all_antecedents_proven = False
                        break
                
                if all_antecedents_proven:
                    self.inference_chain.append({
                        'type': 'backward_chaining',
                        'goal': goal.statement,
                        'rule': str(rule),
                        'result': 'proven'
                    })
                    return True
        
        self.inference_chain.append({
            'type': 'backward_chaining',
            'goal': goal.statement,
            'result': 'not_proven'
        })
        return False
    
    async def can_apply_rule(self, rule: Rule) -> bool:
        """Check if a rule can be applied given current facts."""
        for antecedent in rule.antecedent:
            antecedent_satisfied = False
            for fact in self.facts.values():
                if await self.statements_match(antecedent, fact):
                    antecedent_satisfied = True
                    break
            if not antecedent_satisfied:
                return False
        return True
    
    async def apply_rule(self, rule: Rule) -> Optional[LogicalStatement]:
        """Apply a rule to derive a new fact."""
        if not await self.can_apply_rule(rule):
            return None
        
        # Simple rule application - derive the consequent
        new_fact = LogicalStatement(
            id=f"derived_{len(self.derived_facts)}",
            statement=rule.consequent.statement,
            confidence=min(rule.confidence, 
                          min(self.get_fact_confidence(ant) for ant in rule.antecedent))
        )
        
        self.derived_facts[new_fact.id] = new_fact
        return new_fact
    
    def get_fact_confidence(self, statement: LogicalStatement) -> float:
        """Get confidence of a fact statement."""
        for fact in self.facts.values():
            if fact.statement == statement.statement:
                return fact.confidence
        return 0.0
    
    async def statements_match(self, stmt1: LogicalStatement, stmt2: LogicalStatement) -> bool:
        """Check if two statements match (simple string comparison for now)."""
        return stmt1.statement.strip().lower() == stmt2.statement.strip().lower()
    
    async def resolution_theorem_proving(self, goal: LogicalStatement) -> bool:
        """Apply resolution-based theorem proving."""
        # Convert to CNF and apply resolution
        # This is a simplified implementation
        
        clauses = await self.convert_to_cnf()
        negated_goal = LogicalStatement(
            id="negated_goal",
            statement=f"NOT ({goal.statement})"
        )
        clauses.append(negated_goal)
        
        return await self.resolution(clauses)
    
    async def convert_to_cnf(self) -> List[LogicalStatement]:
        """Convert knowledge base to Conjunctive Normal Form."""
        # Simplified CNF conversion
        cnf_clauses = []
        
        # Add facts as unit clauses
        for fact in self.facts.values():
            cnf_clauses.append(fact)
        
        # Convert rules to clauses
        for rule in self.rules.values():
            # Convert "A AND B -> C" to "NOT A OR NOT B OR C"
            clause_parts = []
            for antecedent in rule.antecedent:
                clause_parts.append(f"NOT ({antecedent.statement})")
            clause_parts.append(rule.consequent.statement)
            
            clause = LogicalStatement(
                id=f"rule_clause_{rule.id}",
                statement=" OR ".join(clause_parts)
            )
            cnf_clauses.append(clause)
        
        return cnf_clauses
    
    async def resolution(self, clauses: List[LogicalStatement]) -> bool:
        """Apply resolution algorithm."""
        # Simplified resolution - checks for contradictions
        new_clauses = clauses.copy()
        
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                resolvents = await self.resolve_clauses(clauses[i], clauses[j])
                for resolvent in resolvents:
                    if resolvent.statement == "FALSE":
                        # Found contradiction - goal is provable
                        return True
                    if resolvent not in new_clauses:
                        new_clauses.append(resolvent)
        
        return False
    
    async def resolve_clauses(self, clause1: LogicalStatement, clause2: LogicalStatement) -> List[LogicalStatement]:
        """Resolve two clauses."""
        # Simplified resolution - look for complementary literals
        resolvents = []
        
        # This is a very basic implementation
        # Real resolution would parse the clauses properly
        if "NOT" in clause1.statement and clause1.statement.replace("NOT ", "") in clause2.statement:
            resolvent = LogicalStatement(
                id=f"resolvent_{clause1.id}_{clause2.id}",
                statement="TRUE"
            )
            resolvents.append(resolvent)
        
        return resolvents


class TheoremProver:
    """Automated theorem prover using multiple proof strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.inference_engine = LogicalInferenceEngine(config)
        self.proof_strategies = ['forward_chaining', 'backward_chaining', 'resolution']
    
    async def prove_theorem(self, 
                           theorem: LogicalStatement,
                           axioms: List[LogicalStatement],
                           rules: List[Rule]) -> Dict[str, Any]:
        """Prove a theorem given axioms and rules."""
        proof_result = {
            'theorem': theorem.statement,
            'proven': False,
            'proof_method': None,
            'proof_steps': [],
            'confidence': 0.0
        }
        
        # Add axioms and rules to inference engine
        for axiom in axioms:
            await self.inference_engine.add_fact(axiom)
        
        for rule in rules:
            await self.inference_engine.add_rule(rule)
        
        # Try different proof strategies
        for strategy in self.proof_strategies:
            try:
                if strategy == 'forward_chaining':
                    derived_facts = await self.inference_engine.forward_chaining()
                    for fact in derived_facts:
                        if await self.inference_engine.statements_match(theorem, fact):
                            proof_result['proven'] = True
                            proof_result['proof_method'] = 'forward_chaining'
                            proof_result['confidence'] = fact.confidence
                            break
                
                elif strategy == 'backward_chaining':
                    proven = await self.inference_engine.backward_chaining(theorem)
                    if proven:
                        proof_result['proven'] = True
                        proof_result['proof_method'] = 'backward_chaining'
                        proof_result['confidence'] = 0.9  # High confidence for logical proof
                        break
                
                elif strategy == 'resolution':
                    proven = await self.inference_engine.resolution_theorem_proving(theorem)
                    if proven:
                        proof_result['proven'] = True
                        proof_result['proof_method'] = 'resolution'
                        proof_result['confidence'] = 1.0  # Certainty for resolution proof
                        break
                
                if proof_result['proven']:
                    break
                    
            except Exception as e:
                continue  # Try next strategy
        
        proof_result['proof_steps'] = self.inference_engine.inference_chain
        return proof_result


class RuleEngine:
    """Rule-based reasoning engine for production systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rules = []
        self.working_memory = {}
        self.conflict_resolution_strategy = config.get('conflict_resolution', 'priority')
    
    async def add_rule(self, rule: Dict[str, Any]) -> None:
        """Add a production rule."""
        rule_obj = {
            'id': rule.get('id', f"rule_{len(self.rules)}"),
            'conditions': rule.get('conditions', []),
            'actions': rule.get('actions', []),
            'priority': rule.get('priority', 0),
            'confidence': rule.get('confidence', 1.0),
            'fired_count': 0
        }
        self.rules.append(rule_obj)
    
    async def add_fact(self, fact_name: str, fact_value: Any) -> None:
        """Add a fact to working memory."""
        self.working_memory[fact_name] = fact_value
    
    async def evaluate_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        fact_name = condition.get('fact')
        operator = condition.get('operator', '==')
        value = condition.get('value')
        
        if fact_name not in self.working_memory:
            return False
        
        fact_value = self.working_memory[fact_name]
        
        if operator == '==':
            return fact_value == value
        elif operator == '!=':
            return fact_value != value
        elif operator == '>':
            return fact_value > value
        elif operator == '<':
            return fact_value < value
        elif operator == '>=':
            return fact_value >= value
        elif operator == '<=':
            return fact_value <= value
        elif operator == 'in':
            return fact_value in value
        elif operator == 'contains':
            return value in fact_value
        else:
            return False
    
    async def evaluate_conditions(self, conditions: List[Dict[str, Any]]) -> bool:
        """Evaluate all conditions for a rule."""
        for condition in conditions:
            if not await self.evaluate_condition(condition):
                return False
        return True
    
    async def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute a rule action."""
        action_type = action.get('type')
        
        if action_type == 'assert':
            fact_name = action.get('fact')
            fact_value = action.get('value')
            await self.add_fact(fact_name, fact_value)
        
        elif action_type == 'retract':
            fact_name = action.get('fact')
            if fact_name in self.working_memory:
                del self.working_memory[fact_name]
        
        elif action_type == 'modify':
            fact_name = action.get('fact')
            new_value = action.get('value')
            if fact_name in self.working_memory:
                self.working_memory[fact_name] = new_value
    
    async def find_applicable_rules(self) -> List[Dict[str, Any]]:
        """Find rules whose conditions are satisfied."""
        applicable_rules = []
        
        for rule in self.rules:
            if await self.evaluate_conditions(rule['conditions']):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    async def resolve_conflicts(self, applicable_rules: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select one rule from multiple applicable rules."""
        if not applicable_rules:
            return None
        
        if self.conflict_resolution_strategy == 'priority':
            return max(applicable_rules, key=lambda r: r['priority'])
        elif self.conflict_resolution_strategy == 'recency':
            return applicable_rules[-1]  # Most recently added
        elif self.conflict_resolution_strategy == 'specificity':
            return max(applicable_rules, key=lambda r: len(r['conditions']))
        else:
            return applicable_rules[0]  # First applicable rule
    
    async def run_inference(self, max_cycles: int = 100) -> List[Dict[str, Any]]:
        """Run the inference engine."""
        fired_rules = []
        cycle = 0
        
        while cycle < max_cycles:
            cycle += 1
            
            # Find applicable rules
            applicable_rules = await self.find_applicable_rules()
            
            # Resolve conflicts
            selected_rule = await self.resolve_conflicts(applicable_rules)
            
            if not selected_rule:
                break  # No more applicable rules
            
            # Execute rule actions
            for action in selected_rule['actions']:
                await self.execute_action(action)
            
            # Update rule statistics
            selected_rule['fired_count'] += 1
            
            fired_rules.append({
                'cycle': cycle,
                'rule_id': selected_rule['id'],
                'rule': selected_rule
            })
        
        return fired_rules


class DeductiveReasoner(BaseReasoner):
    """Main deductive reasoning engine combining logical inference approaches."""
    
    def __init__(self, name: str = "DeductiveReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.DEDUCTIVE, config)
        self.inference_engine = LogicalInferenceEngine(config)
        self.theorem_prover = TheoremProver(config)
        self.rule_engine = RuleEngine(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform deductive reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Extract logical statements from evidence
            statements, rules = await self.extract_logical_components(context)
            
            # Add statements and rules to inference engines
            for statement in statements:
                await self.inference_engine.add_fact(statement)
            
            for rule in rules:
                await self.inference_engine.add_rule(rule)
            
            # Apply forward chaining
            derived_facts = await self.inference_engine.forward_chaining()
            result.reasoning_trace.append(f"Forward chaining derived {len(derived_facts)} facts")
            
            # Generate conclusions from derived facts
            for fact in derived_facts:
                conclusion = Conclusion(
                    statement=fact.statement,
                    confidence=fact.confidence,
                    reasoning_type=self.reasoning_type,
                    reasoning_chain=["Forward chaining inference"]
                )
                result.conclusions.append(conclusion)
            
            # Apply backward chaining if goal is specified
            if context.goal:
                goal_statement = LogicalStatement(
                    id="goal",
                    statement=context.goal
                )
                
                proven = await self.inference_engine.backward_chaining(goal_statement)
                result.reasoning_trace.append(f"Backward chaining goal proof: {proven}")
                
                if proven:
                    conclusion = Conclusion(
                        statement=f"Goal proven: {context.goal}",
                        confidence=0.95,
                        reasoning_type=self.reasoning_type,
                        reasoning_chain=["Backward chaining proof"]
                    )
                    result.conclusions.append(conclusion)
            
            # Apply rule-based reasoning if facts are present
            if 'facts' in context.metadata:
                for fact_name, fact_value in context.metadata['facts'].items():
                    await self.rule_engine.add_fact(fact_name, fact_value)
                
                if 'rules' in context.metadata:
                    for rule_data in context.metadata['rules']:
                        await self.rule_engine.add_rule(rule_data)
                
                fired_rules = await self.rule_engine.run_inference()
                result.reasoning_trace.append(f"Rule engine fired {len(fired_rules)} rules")
                
                # Generate conclusions from rule firing
                for fired_rule in fired_rules:
                    rule_data = fired_rule['rule']
                    conclusion = Conclusion(
                        statement=f"Rule applied: {rule_data['id']}",
                        confidence=rule_data['confidence'],
                        reasoning_type=self.reasoning_type,
                        reasoning_chain=[f"Rule firing cycle {fired_rule['cycle']}"]
                    )
                    result.conclusions.append(conclusion)
            
            # Calculate overall confidence
            if result.conclusions:
                result.confidence = sum(c.confidence for c in result.conclusions) / len(result.conclusions)
            else:
                result.confidence = 0.0
            
            result.success = len(result.conclusions) > 0
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.reasoning_trace.append(f"Error: {e}")
        
        return result
    
    async def extract_logical_components(self, context: ReasoningContext) -> Tuple[List[LogicalStatement], List[Rule]]:
        """Extract logical statements and rules from context."""
        statements = []
        rules = []
        
        # Extract from evidence
        for evidence in context.available_evidence:
            if isinstance(evidence.content, str):
                # Simple heuristic to identify statements vs rules
                if "if" in evidence.content.lower() and "then" in evidence.content.lower():
                    # This looks like a rule
                    rule = await self.parse_rule(evidence.content, evidence)
                    if rule:
                        rules.append(rule)
                else:
                    # This looks like a statement/fact
                    statement = LogicalStatement(
                        id=evidence.id,
                        statement=evidence.content,
                        confidence=evidence.confidence
                    )
                    statements.append(statement)
        
        # Extract from prior knowledge
        if 'logical_statements' in context.prior_knowledge:
            for stmt_data in context.prior_knowledge['logical_statements']:
                statement = LogicalStatement(
                    id=stmt_data.get('id', f"stmt_{len(statements)}"),
                    statement=stmt_data['statement'],
                    confidence=stmt_data.get('confidence', 1.0)
                )
                statements.append(statement)
        
        if 'logical_rules' in context.prior_knowledge:
            for rule_data in context.prior_knowledge['logical_rules']:
                rule = await self.parse_rule_from_data(rule_data)
                if rule:
                    rules.append(rule)
        
        return statements, rules
    
    async def parse_rule(self, rule_text: str, evidence: Evidence) -> Optional[Rule]:
        """Parse a rule from text."""
        # Simple rule parsing - can be enhanced
        rule_text = rule_text.lower()
        
        if_index = rule_text.find("if")
        then_index = rule_text.find("then")
        
        if if_index == -1 or then_index == -1 or if_index >= then_index:
            return None
        
        antecedent_text = rule_text[if_index + 2:then_index].strip()
        consequent_text = rule_text[then_index + 4:].strip()
        
        # Split antecedent by "and"
        antecedent_parts = [part.strip() for part in antecedent_text.split(" and ")]
        
        antecedents = []
        for part in antecedent_parts:
            if part:
                stmt = LogicalStatement(
                    id=f"ant_{len(antecedents)}",
                    statement=part
                )
                antecedents.append(stmt)
        
        consequent = LogicalStatement(
            id="cons",
            statement=consequent_text
        )
        
        rule = Rule(
            id=evidence.id + "_rule",
            antecedent=antecedents,
            consequent=consequent,
            confidence=evidence.confidence
        )
        
        return rule
    
    async def parse_rule_from_data(self, rule_data: Dict[str, Any]) -> Optional[Rule]:
        """Parse a rule from structured data."""
        antecedents = []
        for ant_data in rule_data.get('antecedents', []):
            stmt = LogicalStatement(
                id=ant_data.get('id', f"ant_{len(antecedents)}"),
                statement=ant_data['statement']
            )
            antecedents.append(stmt)
        
        consequent_data = rule_data.get('consequent', {})
        consequent = LogicalStatement(
            id=consequent_data.get('id', 'cons'),
            statement=consequent_data['statement']
        )
        
        rule = Rule(
            id=rule_data.get('id', f"rule_{len(antecedents)}"),
            antecedent=antecedents,
            consequent=consequent,
            confidence=rule_data.get('confidence', 1.0),
            priority=rule_data.get('priority', 0)
        )
        
        return rule