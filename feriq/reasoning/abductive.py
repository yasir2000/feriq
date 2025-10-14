"""
Abductive Reasoning Module

Implements inference to best explanation, hypothesis generation,
and diagnostic reasoning for finding the most plausible explanations.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass, field

from .base import (
    BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType,
    Evidence, Hypothesis, Conclusion, create_evidence, create_conclusion
)


@dataclass
class Explanation:
    """Represents an explanation for observed evidence."""
    id: str
    description: str
    plausibility: float = 0.5
    simplicity: float = 0.5
    coverage: float = 0.5
    consistency: float = 1.0
    supported_evidence: List[Evidence] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    def calculate_score(self) -> float:
        """Calculate overall explanation score."""
        return (self.plausibility * 0.4 + 
                self.simplicity * 0.2 + 
                self.coverage * 0.3 + 
                self.consistency * 0.1)


class HypothesisGenerator:
    """Generates hypotheses to explain observed evidence."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_hypotheses = config.get('max_hypotheses', 10)
    
    async def generate_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate hypotheses from evidence."""
        hypotheses = []
        
        # Pattern-based hypothesis generation
        pattern_hypotheses = await self._generate_pattern_hypotheses(evidence_list)
        hypotheses.extend(pattern_hypotheses)
        
        # Causal hypothesis generation
        causal_hypotheses = await self._generate_causal_hypotheses(evidence_list)
        hypotheses.extend(causal_hypotheses)
        
        # Analogical hypothesis generation
        analogical_hypotheses = await self._generate_analogical_hypotheses(evidence_list)
        hypotheses.extend(analogical_hypotheses)
        
        # Limit and rank hypotheses
        ranked_hypotheses = await self._rank_hypotheses(hypotheses, evidence_list)
        return ranked_hypotheses[:self.max_hypotheses]
    
    async def _generate_pattern_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate hypotheses based on patterns in evidence."""
        hypotheses = []
        
        # Group evidence by confidence levels
        high_conf_evidence = [e for e in evidence_list if e.confidence > 0.8]
        medium_conf_evidence = [e for e in evidence_list if 0.4 <= e.confidence <= 0.8]
        
        if high_conf_evidence:
            hypothesis = Hypothesis(
                statement="High confidence evidence suggests a reliable pattern",
                probability=0.8,
                evidence_for=high_conf_evidence,
                metadata={'type': 'pattern', 'basis': 'high_confidence'}
            )
            hypotheses.append(hypothesis)
        
        if len(medium_conf_evidence) > len(high_conf_evidence):
            hypothesis = Hypothesis(
                statement="Moderate evidence dominance suggests uncertainty pattern",
                probability=0.6,
                evidence_for=medium_conf_evidence,
                metadata={'type': 'pattern', 'basis': 'uncertainty'}
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_causal_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate causal hypotheses."""
        hypotheses = []
        
        # Look for temporal patterns
        temporal_evidence = [e for e in evidence_list if hasattr(e, 'timestamp')]
        if len(temporal_evidence) >= 2:
            sorted_evidence = sorted(temporal_evidence, key=lambda e: e.timestamp)
            
            hypothesis = Hypothesis(
                statement=f"Earlier event ({sorted_evidence[0].content}) may have caused later events",
                probability=0.7,
                evidence_for=sorted_evidence,
                metadata={'type': 'causal', 'basis': 'temporal_sequence'}
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_analogical_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate hypotheses based on analogies."""
        hypotheses = []
        
        # Simple analogical reasoning based on content similarity
        text_evidence = [e for e in evidence_list if isinstance(e.content, str)]
        
        if len(text_evidence) >= 2:
            # Look for common themes
            all_words = set()
            for evidence in text_evidence:
                words = evidence.content.lower().split()
                all_words.update(words)
            
            common_words = []
            for word in all_words:
                count = sum(1 for e in text_evidence if word in e.content.lower())
                if count >= 2:
                    common_words.append(word)
            
            if common_words:
                hypothesis = Hypothesis(
                    statement=f"Common themes ({', '.join(common_words[:3])}) suggest underlying similarity",
                    probability=0.6,
                    evidence_for=text_evidence,
                    metadata={'type': 'analogical', 'common_themes': common_words}
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _rank_hypotheses(self, hypotheses: List[Hypothesis], evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Rank hypotheses by quality."""
        for hypothesis in hypotheses:
            # Calculate coverage (how much evidence does it explain?)
            coverage = len(hypothesis.evidence_for) / len(evidence_list) if evidence_list else 0
            
            # Calculate simplicity (fewer assumptions = simpler)
            simplicity = 1.0 / (len(hypothesis.metadata.get('assumptions', [])) + 1)
            
            # Adjust probability based on coverage and simplicity
            hypothesis.probability *= (0.5 + 0.3 * coverage + 0.2 * simplicity)
            hypothesis.probability = min(hypothesis.probability, 0.95)
        
        return sorted(hypotheses, key=lambda h: h.probability, reverse=True)


class ExplanationRanker:
    """Ranks explanations based on multiple criteria."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.criteria_weights = config.get('criteria_weights', {
            'plausibility': 0.4,
            'simplicity': 0.2,
            'coverage': 0.3,
            'consistency': 0.1
        })
    
    async def rank_explanations(self, explanations: List[Explanation], evidence_list: List[Evidence]) -> List[Explanation]:
        """Rank explanations by quality."""
        for explanation in explanations:
            await self._evaluate_explanation(explanation, evidence_list)
        
        return sorted(explanations, key=lambda e: e.calculate_score(), reverse=True)
    
    async def _evaluate_explanation(self, explanation: Explanation, evidence_list: List[Evidence]) -> None:
        """Evaluate an explanation on multiple criteria."""
        # Calculate coverage
        explanation.coverage = len(explanation.supported_evidence) / len(evidence_list) if evidence_list else 0
        
        # Calculate simplicity (inverse of complexity)
        complexity = len(explanation.assumptions) + len(explanation.description.split())
        explanation.simplicity = 1.0 / (1.0 + complexity / 10.0)
        
        # Calculate consistency (no contradictions)
        explanation.consistency = await self._check_consistency(explanation)
        
        # Calculate plausibility based on evidence strength
        if explanation.supported_evidence:
            avg_confidence = sum(e.confidence for e in explanation.supported_evidence) / len(explanation.supported_evidence)
            explanation.plausibility = avg_confidence
        else:
            explanation.plausibility = 0.1
    
    async def _check_consistency(self, explanation: Explanation) -> float:
        """Check internal consistency of explanation."""
        # Simple consistency check - no obvious contradictions
        description_lower = explanation.description.lower()
        
        # Look for contradictory terms
        contradictory_pairs = [
            ('increase', 'decrease'),
            ('positive', 'negative'),
            ('cause', 'prevent'),
            ('enable', 'disable')
        ]
        
        for term1, term2 in contradictory_pairs:
            if term1 in description_lower and term2 in description_lower:
                return 0.5  # Reduced consistency
        
        return 1.0  # No obvious contradictions


class DiagnosticReasoner:
    """Performs diagnostic reasoning to identify likely causes."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.diagnostic_rules = config.get('diagnostic_rules', [])
    
    async def diagnose(self, symptoms: List[Evidence], possible_causes: List[str] = None) -> List[Hypothesis]:
        """Perform diagnostic reasoning."""
        if possible_causes is None:
            possible_causes = await self._generate_possible_causes(symptoms)
        
        diagnoses = []
        
        for cause in possible_causes:
            probability = await self._calculate_diagnostic_probability(cause, symptoms)
            
            if probability > 0.1:  # Only consider plausible diagnoses
                hypothesis = Hypothesis(
                    statement=f"Diagnosis: {cause}",
                    probability=probability,
                    evidence_for=symptoms,
                    metadata={'type': 'diagnostic', 'cause': cause}
                )
                diagnoses.append(hypothesis)
        
        return sorted(diagnoses, key=lambda d: d.probability, reverse=True)
    
    async def _generate_possible_causes(self, symptoms: List[Evidence]) -> List[str]:
        """Generate possible causes from symptoms."""
        causes = set()
        
        # Extract potential causes from evidence content
        for symptom in symptoms:
            if isinstance(symptom.content, str):
                content_lower = symptom.content.lower()
                
                # Simple heuristics for cause identification
                if 'error' in content_lower:
                    causes.add('system_error')
                if 'failure' in content_lower:
                    causes.add('component_failure')
                if 'slow' in content_lower or 'delay' in content_lower:
                    causes.add('performance_issue')
                if 'missing' in content_lower or 'not found' in content_lower:
                    causes.add('missing_component')
                if 'conflict' in content_lower or 'contradiction' in content_lower:
                    causes.add('configuration_conflict')
        
        # Add generic causes if no specific ones found
        if not causes:
            causes.update(['unknown_cause', 'external_factor', 'user_error'])
        
        return list(causes)
    
    async def _calculate_diagnostic_probability(self, cause: str, symptoms: List[Evidence]) -> float:
        """Calculate probability of a cause given symptoms."""
        # Simple Bayesian-like calculation
        prior_probability = 0.1  # Base rate for any cause
        
        supporting_evidence = 0
        total_evidence = len(symptoms)
        
        for symptom in symptoms:
            if await self._symptom_supports_cause(symptom, cause):
                supporting_evidence += symptom.confidence
        
        if total_evidence == 0:
            return prior_probability
        
        # Likelihood based on supporting evidence
        likelihood = supporting_evidence / total_evidence
        
        # Simple Bayesian update
        posterior = (likelihood * prior_probability) / (likelihood * prior_probability + (1 - likelihood) * (1 - prior_probability))
        
        return min(posterior, 0.9)  # Cap at 90%
    
    async def _symptom_supports_cause(self, symptom: Evidence, cause: str) -> bool:
        """Check if a symptom supports a particular cause."""
        if not isinstance(symptom.content, str):
            return False
        
        content_lower = symptom.content.lower()
        
        # Simple keyword matching
        if cause == 'system_error' and ('error' in content_lower or 'exception' in content_lower):
            return True
        elif cause == 'component_failure' and ('failure' in content_lower or 'broken' in content_lower):
            return True
        elif cause == 'performance_issue' and ('slow' in content_lower or 'delay' in content_lower):
            return True
        elif cause == 'missing_component' and ('missing' in content_lower or 'not found' in content_lower):
            return True
        elif cause == 'configuration_conflict' and ('conflict' in content_lower or 'contradiction' in content_lower):
            return True
        
        return False


class AbductiveReasoner(BaseReasoner):
    """Main abductive reasoning engine combining all abductive approaches."""
    
    def __init__(self, name: str = "AbductiveReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.ABDUCTIVE, config)
        self.hypothesis_generator = HypothesisGenerator(config)
        self.explanation_ranker = ExplanationRanker(config)
        self.diagnostic_reasoner = DiagnosticReasoner(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform abductive reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Generate hypotheses
            hypotheses = await self.hypothesis_generator.generate_hypotheses(context.available_evidence)
            result.hypotheses = hypotheses
            result.reasoning_trace.append(f"Generated {len(hypotheses)} hypotheses")
            
            # Create explanations from hypotheses
            explanations = await self._create_explanations_from_hypotheses(hypotheses, context)
            
            # Rank explanations
            ranked_explanations = await self.explanation_ranker.rank_explanations(explanations, context.available_evidence)
            result.reasoning_trace.append(f"Ranked {len(ranked_explanations)} explanations")
            
            # Convert best explanations to conclusions
            for explanation in ranked_explanations[:3]:  # Top 3 explanations
                conclusion = Conclusion(
                    statement=f"Best explanation: {explanation.description}",
                    confidence=explanation.calculate_score(),
                    reasoning_type=self.reasoning_type,
                    reasoning_chain=["Abductive inference", "Explanation ranking"],
                    metadata={'explanation': explanation}
                )
                result.conclusions.append(conclusion)
            
            # Diagnostic reasoning if symptoms are present
            if await self._has_diagnostic_context(context):
                diagnoses = await self.diagnostic_reasoner.diagnose(context.available_evidence)
                result.reasoning_trace.append(f"Generated {len(diagnoses)} diagnostic hypotheses")
                
                for diagnosis in diagnoses[:2]:  # Top 2 diagnoses
                    conclusion = Conclusion(
                        statement=diagnosis.statement,
                        confidence=diagnosis.probability,
                        reasoning_type=self.reasoning_type,
                        reasoning_chain=["Diagnostic reasoning"],
                        metadata={'diagnosis': diagnosis}
                    )
                    result.conclusions.append(conclusion)
            
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
    
    async def _create_explanations_from_hypotheses(self, hypotheses: List[Hypothesis], context: ReasoningContext) -> List[Explanation]:
        """Create explanations from hypotheses."""
        explanations = []
        
        for i, hypothesis in enumerate(hypotheses):
            explanation = Explanation(
                id=f"explanation_{i}",
                description=hypothesis.statement,
                plausibility=hypothesis.probability,
                supported_evidence=hypothesis.evidence_for,
                assumptions=hypothesis.metadata.get('assumptions', [])
            )
            explanations.append(explanation)
        
        return explanations
    
    async def _has_diagnostic_context(self, context: ReasoningContext) -> bool:
        """Check if context suggests diagnostic reasoning is appropriate."""
        # Look for diagnostic keywords in goal or evidence
        diagnostic_keywords = ['diagnose', 'cause', 'problem', 'issue', 'error', 'failure', 'symptom']
        
        if context.goal:
            goal_lower = context.goal.lower()
            if any(keyword in goal_lower for keyword in diagnostic_keywords):
                return True
        
        # Check evidence content
        for evidence in context.available_evidence:
            if isinstance(evidence.content, str):
                content_lower = evidence.content.lower()
                if any(keyword in content_lower for keyword in diagnostic_keywords):
                    return True
        
        return False