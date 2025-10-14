"""
Inductive Reasoning Module

Implements pattern recognition, generalization, and learning from examples.
Supports few-shot learning, case-based reasoning, and analogical inference.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import asyncio
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from .base import (
    BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType,
    Evidence, Hypothesis, Conclusion, create_evidence, create_conclusion
)


class PatternRecognizer:
    """Recognizes patterns in data and evidence."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.patterns = {}
        self.pattern_confidence = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    async def identify_patterns(self, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """Identify patterns in a list of evidence."""
        patterns = []
        
        # Text pattern recognition
        text_patterns = await self.identify_text_patterns(evidence_list)
        patterns.extend(text_patterns)
        
        # Structural pattern recognition
        structural_patterns = await self.identify_structural_patterns(evidence_list)
        patterns.extend(structural_patterns)
        
        # Temporal pattern recognition
        temporal_patterns = await self.identify_temporal_patterns(evidence_list)
        patterns.extend(temporal_patterns)
        
        return patterns
    
    async def identify_text_patterns(self, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """Identify patterns in text evidence."""
        text_evidence = [e for e in evidence_list if isinstance(e.content, str)]
        if len(text_evidence) < 2:
            return []
        
        patterns = []
        texts = [e.content for e in text_evidence]
        
        # Common phrases
        common_phrases = await self.find_common_phrases(texts)
        for phrase, frequency in common_phrases.items():
            patterns.append({
                'type': 'common_phrase',
                'pattern': phrase,
                'frequency': frequency,
                'confidence': min(frequency / len(texts), 1.0),
                'evidence_ids': [e.id for e in text_evidence if phrase in e.content]
            })
        
        # Semantic clustering
        try:
            if len(texts) >= 3:
                vectors = self.vectorizer.fit_transform(texts)
                clusters = await self.cluster_texts(vectors.toarray(), texts)
                for cluster_id, cluster_info in clusters.items():
                    patterns.append({
                        'type': 'semantic_cluster',
                        'pattern': cluster_info['theme'],
                        'members': cluster_info['texts'],
                        'confidence': cluster_info['coherence'],
                        'evidence_ids': cluster_info['evidence_ids']
                    })
        except Exception:
            pass  # Skip if vectorization fails
        
        return patterns
    
    async def find_common_phrases(self, texts: List[str]) -> Dict[str, int]:
        """Find common phrases across texts."""
        phrase_counts = Counter()
        
        for text in texts:
            # Extract n-grams (2-4 words)
            words = re.findall(r'\b\w+\b', text.lower())
            for n in range(2, min(5, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    phrase_counts[phrase] += 1
        
        # Filter phrases that appear in multiple texts
        common_phrases = {
            phrase: count for phrase, count in phrase_counts.items()
            if count > 1 and len(phrase.split()) >= 2
        }
        
        return dict(sorted(common_phrases.items(), key=lambda x: x[1], reverse=True)[:10])
    
    async def cluster_texts(self, vectors: np.ndarray, texts: List[str]) -> Dict[int, Dict[str, Any]]:
        """Cluster texts based on semantic similarity."""
        if len(vectors) < 3:
            return {}
        
        # Determine optimal number of clusters
        n_clusters = min(3, len(vectors) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)
        
        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if len(cluster_indices) < 2:
                continue
            
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_vectors = vectors[cluster_indices]
            
            # Calculate cluster coherence
            similarities = cosine_similarity(cluster_vectors)
            coherence = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            
            # Extract theme (most common words)
            all_words = ' '.join(cluster_texts).lower().split()
            word_counts = Counter(all_words)
            theme_words = [word for word, count in word_counts.most_common(3)]
            theme = ' '.join(theme_words)
            
            clusters[cluster_id] = {
                'texts': cluster_texts,
                'theme': theme,
                'coherence': float(coherence),
                'evidence_ids': [f"text_{i}" for i in cluster_indices]
            }
        
        return clusters
    
    async def identify_structural_patterns(self, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """Identify structural patterns in evidence."""
        patterns = []
        
        # Group evidence by type
        type_groups = defaultdict(list)
        for evidence in evidence_list:
            evidence_type = type(evidence.content).__name__
            type_groups[evidence_type].append(evidence)
        
        # Identify prevalent types
        for evidence_type, group in type_groups.items():
            if len(group) >= 2:
                patterns.append({
                    'type': 'structural_type',
                    'pattern': f"prevalent_{evidence_type}",
                    'frequency': len(group),
                    'confidence': len(group) / len(evidence_list),
                    'evidence_ids': [e.id for e in group]
                })
        
        # Confidence level patterns
        confidence_groups = defaultdict(list)
        for evidence in evidence_list:
            confidence_range = self.get_confidence_range(evidence.confidence)
            confidence_groups[confidence_range].append(evidence)
        
        for conf_range, group in confidence_groups.items():
            if len(group) >= 2:
                patterns.append({
                    'type': 'confidence_pattern',
                    'pattern': f"confidence_{conf_range}",
                    'frequency': len(group),
                    'confidence': len(group) / len(evidence_list),
                    'evidence_ids': [e.id for e in group]
                })
        
        return patterns
    
    def get_confidence_range(self, confidence: float) -> str:
        """Categorize confidence into ranges."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    async def identify_temporal_patterns(self, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """Identify temporal patterns in evidence."""
        patterns = []
        
        # Sort evidence by timestamp
        sorted_evidence = sorted(evidence_list, key=lambda e: e.timestamp)
        
        # Time-based clustering (evidence appearing close in time)
        time_clusters = []
        current_cluster = [sorted_evidence[0]] if sorted_evidence else []
        
        for i in range(1, len(sorted_evidence)):
            prev_time = sorted_evidence[i-1].timestamp
            curr_time = sorted_evidence[i].timestamp
            time_diff = (curr_time - prev_time).total_seconds()
            
            # If evidence appears within 1 hour, group together
            if time_diff <= 3600:
                current_cluster.append(sorted_evidence[i])
            else:
                if len(current_cluster) >= 2:
                    time_clusters.append(current_cluster)
                current_cluster = [sorted_evidence[i]]
        
        if len(current_cluster) >= 2:
            time_clusters.append(current_cluster)
        
        # Create temporal patterns
        for i, cluster in enumerate(time_clusters):
            patterns.append({
                'type': 'temporal_cluster',
                'pattern': f"time_cluster_{i}",
                'frequency': len(cluster),
                'confidence': len(cluster) / len(evidence_list),
                'evidence_ids': [e.id for e in cluster],
                'time_span': (cluster[-1].timestamp - cluster[0].timestamp).total_seconds()
            })
        
        return patterns


class FewShotLearner:
    """Learns from few examples using meta-learning approaches."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.examples = []
        self.learned_patterns = {}
    
    async def learn_from_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn patterns from few examples."""
        self.examples.extend(examples)
        
        learning_result = {
            'patterns': [],
            'generalizations': [],
            'confidence': 0.0
        }
        
        if len(examples) < 2:
            return learning_result
        
        # Feature extraction
        features = await self.extract_features(examples)
        
        # Pattern identification
        patterns = await self.identify_common_patterns(features)
        learning_result['patterns'] = patterns
        
        # Generalization
        generalizations = await self.generate_generalizations(patterns)
        learning_result['generalizations'] = generalizations
        
        # Confidence estimation
        learning_result['confidence'] = await self.estimate_confidence(patterns, examples)
        
        return learning_result
    
    async def extract_features(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract features from examples."""
        features = []
        
        for example in examples:
            feature_dict = {}
            
            # Basic features
            if 'input' in example:
                feature_dict['input_type'] = type(example['input']).__name__
                if isinstance(example['input'], str):
                    feature_dict['input_length'] = len(example['input'])
                    feature_dict['input_words'] = len(example['input'].split())
            
            if 'output' in example:
                feature_dict['output_type'] = type(example['output']).__name__
                if isinstance(example['output'], str):
                    feature_dict['output_length'] = len(example['output'])
                    feature_dict['output_words'] = len(example['output'].split())
            
            # Relationship features
            if 'input' in example and 'output' in example:
                feature_dict['io_relationship'] = await self.analyze_io_relationship(
                    example['input'], example['output']
                )
            
            features.append(feature_dict)
        
        return features
    
    async def analyze_io_relationship(self, input_data: Any, output_data: Any) -> str:
        """Analyze the relationship between input and output."""
        if isinstance(input_data, str) and isinstance(output_data, str):
            if output_data.lower() in input_data.lower():
                return "extraction"
            elif len(output_data) > len(input_data):
                return "expansion"
            elif len(output_data) < len(input_data):
                return "summarization"
            else:
                return "transformation"
        elif isinstance(input_data, (int, float)) and isinstance(output_data, (int, float)):
            ratio = output_data / input_data if input_data != 0 else 0
            if ratio > 1:
                return "amplification"
            elif ratio < 1:
                return "reduction"
            else:
                return "identity"
        else:
            return "type_conversion"
    
    async def identify_common_patterns(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify common patterns across feature sets."""
        patterns = []
        
        if not features:
            return patterns
        
        # Find common feature values
        common_features = {}
        for feature_name in features[0].keys():
            values = [f.get(feature_name) for f in features if feature_name in f]
            value_counts = Counter(values)
            most_common = value_counts.most_common(1)[0]
            if most_common[1] > 1:  # Appears in multiple examples
                common_features[feature_name] = {
                    'value': most_common[0],
                    'frequency': most_common[1],
                    'confidence': most_common[1] / len(features)
                }
        
        for feature_name, feature_info in common_features.items():
            patterns.append({
                'type': 'common_feature',
                'feature': feature_name,
                'value': feature_info['value'],
                'frequency': feature_info['frequency'],
                'confidence': feature_info['confidence']
            })
        
        return patterns
    
    async def generate_generalizations(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate generalizations from identified patterns."""
        generalizations = []
        
        for pattern in patterns:
            if pattern['type'] == 'common_feature':
                if pattern['confidence'] > 0.7:
                    generalization = f"Most examples have {pattern['feature']} = {pattern['value']}"
                    generalizations.append(generalization)
        
        # Cross-pattern generalizations
        high_conf_patterns = [p for p in patterns if p.get('confidence', 0) > 0.8]
        if len(high_conf_patterns) >= 2:
            feature_names = [p['feature'] for p in high_conf_patterns]
            generalization = f"Strong patterns found in features: {', '.join(feature_names)}"
            generalizations.append(generalization)
        
        return generalizations
    
    async def estimate_confidence(self, patterns: List[Dict[str, Any]], examples: List[Dict[str, Any]]) -> float:
        """Estimate confidence in learned patterns."""
        if not patterns or not examples:
            return 0.0
        
        # Base confidence on pattern strength and example count
        pattern_strengths = [p.get('confidence', 0) for p in patterns]
        avg_pattern_strength = sum(pattern_strengths) / len(pattern_strengths)
        
        # Adjust for number of examples
        example_count_factor = min(len(examples) / 5, 1.0)  # Plateau at 5 examples
        
        confidence = avg_pattern_strength * example_count_factor
        return min(confidence, 0.95)  # Cap at 95%
    
    async def predict(self, new_input: Any) -> Dict[str, Any]:
        """Make prediction based on learned patterns."""
        if not self.learned_patterns:
            return {'prediction': None, 'confidence': 0.0}
        
        # Extract features from new input
        new_features = await self.extract_features([{'input': new_input}])
        if not new_features:
            return {'prediction': None, 'confidence': 0.0}
        
        new_feature_dict = new_features[0]
        
        # Find matching patterns
        matching_patterns = []
        for pattern in self.learned_patterns.get('patterns', []):
            if pattern['type'] == 'common_feature':
                feature_name = pattern['feature']
                if feature_name in new_feature_dict:
                    if new_feature_dict[feature_name] == pattern['value']:
                        matching_patterns.append(pattern)
        
        if not matching_patterns:
            return {'prediction': None, 'confidence': 0.0}
        
        # Generate prediction based on matching patterns
        confidence = sum(p['confidence'] for p in matching_patterns) / len(matching_patterns)
        
        # Simple prediction logic (can be enhanced)
        prediction = f"Likely follows pattern similar to {len(matching_patterns)} learned examples"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'matching_patterns': matching_patterns
        }


class CaseBasedReasoner:
    """Implements case-based reasoning with retrieval, adaptation, and retention."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.case_library = []
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
    
    async def add_case(self, case: Dict[str, Any]) -> None:
        """Add a case to the case library."""
        case_id = len(self.case_library)
        case['id'] = case_id
        case['usage_count'] = 0
        case['success_rate'] = 1.0
        self.case_library.append(case)
    
    async def retrieve_similar_cases(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve cases similar to the current problem."""
        similar_cases = []
        
        for case in self.case_library:
            similarity = await self.calculate_similarity(problem, case)
            if similarity >= self.similarity_threshold:
                case_copy = case.copy()
                case_copy['similarity'] = similarity
                similar_cases.append(case_copy)
        
        # Sort by similarity and success rate
        similar_cases.sort(key=lambda c: (c['similarity'], c['success_rate']), reverse=True)
        
        return similar_cases[:5]  # Return top 5 similar cases
    
    async def calculate_similarity(self, problem: Dict[str, Any], case: Dict[str, Any]) -> float:
        """Calculate similarity between problem and case."""
        similarity_scores = []
        
        # Feature-based similarity
        problem_features = problem.get('features', {})
        case_features = case.get('features', {})
        
        common_features = set(problem_features.keys()) & set(case_features.keys())
        if common_features:
            feature_similarities = []
            for feature in common_features:
                if problem_features[feature] == case_features[feature]:
                    feature_similarities.append(1.0)
                elif isinstance(problem_features[feature], (int, float)) and isinstance(case_features[feature], (int, float)):
                    # Numerical similarity
                    max_val = max(abs(problem_features[feature]), abs(case_features[feature]))
                    if max_val > 0:
                        diff = abs(problem_features[feature] - case_features[feature])
                        similarity = 1.0 - (diff / max_val)
                        feature_similarities.append(max(similarity, 0))
                    else:
                        feature_similarities.append(1.0)
                else:
                    feature_similarities.append(0.0)
            
            similarity_scores.append(sum(feature_similarities) / len(feature_similarities))
        
        # Text-based similarity
        if 'description' in problem and 'description' in case:
            text_similarity = await self.calculate_text_similarity(
                problem['description'], case['description']
            )
            similarity_scores.append(text_similarity)
        
        if not similarity_scores:
            return 0.0
        
        return sum(similarity_scores) / len(similarity_scores)
    
    async def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    async def adapt_solution(self, case: Dict[str, Any], problem: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt a case solution to the current problem."""
        adapted_solution = case.get('solution', {}).copy()
        
        # Simple adaptation rules
        problem_features = problem.get('features', {})
        case_features = case.get('features', {})
        
        # Adjust numerical parameters
        for feature, problem_value in problem_features.items():
            if feature in case_features:
                case_value = case_features[feature]
                if isinstance(problem_value, (int, float)) and isinstance(case_value, (int, float)):
                    if case_value != 0:
                        adjustment_factor = problem_value / case_value
                        # Apply adjustment to solution parameters
                        for param_name, param_value in adapted_solution.items():
                            if isinstance(param_value, (int, float)):
                                adapted_solution[param_name] = param_value * adjustment_factor
        
        # Add adaptation metadata
        adapted_solution['adapted_from_case'] = case['id']
        adapted_solution['adaptation_confidence'] = case.get('similarity', 0.0)
        
        return adapted_solution
    
    async def retain_case(self, problem: Dict[str, Any], solution: Dict[str, Any], success: bool) -> None:
        """Retain a new case or update existing case statistics."""
        new_case = {
            'problem': problem,
            'solution': solution,
            'success': success,
            'features': problem.get('features', {}),
            'description': problem.get('description', '')
        }
        
        await self.add_case(new_case)
        
        # Update success rates of similar cases
        if success:
            similar_cases = await self.retrieve_similar_cases(problem)
            for case in similar_cases:
                case_id = case['id']
                if case_id < len(self.case_library):
                    self.case_library[case_id]['usage_count'] += 1
                    # Update success rate (simple moving average)
                    current_rate = self.case_library[case_id]['success_rate']
                    usage_count = self.case_library[case_id]['usage_count']
                    new_rate = (current_rate * (usage_count - 1) + 1.0) / usage_count
                    self.case_library[case_id]['success_rate'] = new_rate


class InductiveReasoner(BaseReasoner):
    """Main inductive reasoning engine combining all inductive approaches."""
    
    def __init__(self, name: str = "InductiveReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.INDUCTIVE, config)
        self.pattern_recognizer = PatternRecognizer(config)
        self.few_shot_learner = FewShotLearner(config)
        self.case_based_reasoner = CaseBasedReasoner(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform inductive reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Pattern recognition
            patterns = await self.pattern_recognizer.identify_patterns(context.available_evidence)
            result.reasoning_trace.append(f"Identified {len(patterns)} patterns")
            
            # Generate hypotheses from patterns
            hypotheses = await self.generate_hypotheses_from_patterns(patterns, context)
            result.hypotheses = hypotheses
            
            # Few-shot learning if examples are available
            if 'examples' in context.metadata:
                learning_result = await self.few_shot_learner.learn_from_examples(
                    context.metadata['examples']
                )
                result.reasoning_trace.append(f"Learned from {len(context.metadata['examples'])} examples")
                
                # Generate additional hypotheses from learning
                learning_hypotheses = await self.generate_hypotheses_from_learning(learning_result, context)
                result.hypotheses.extend(learning_hypotheses)
            
            # Case-based reasoning if problem is defined
            if 'problem' in context.metadata:
                similar_cases = await self.case_based_reasoner.retrieve_similar_cases(
                    context.metadata['problem']
                )
                result.reasoning_trace.append(f"Retrieved {len(similar_cases)} similar cases")
                
                # Generate conclusions from cases
                case_conclusions = await self.generate_conclusions_from_cases(similar_cases, context)
                result.conclusions.extend(case_conclusions)
            
            # Generate final conclusions
            pattern_conclusions = await self.generate_conclusions_from_patterns(patterns, context)
            result.conclusions.extend(pattern_conclusions)
            
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
    
    async def generate_hypotheses_from_patterns(self, 
                                               patterns: List[Dict[str, Any]], 
                                               context: ReasoningContext) -> List[Hypothesis]:
        """Generate hypotheses based on identified patterns."""
        hypotheses = []
        
        for pattern in patterns:
            if pattern.get('confidence', 0) > 0.5:
                hypothesis_statement = f"Pattern '{pattern['pattern']}' is significant"
                
                hypothesis = Hypothesis(
                    statement=hypothesis_statement,
                    probability=pattern['confidence'],
                    metadata={
                        'pattern_type': pattern['type'],
                        'pattern_data': pattern
                    }
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def generate_hypotheses_from_learning(self, 
                                               learning_result: Dict[str, Any], 
                                               context: ReasoningContext) -> List[Hypothesis]:
        """Generate hypotheses from few-shot learning results."""
        hypotheses = []
        
        for generalization in learning_result.get('generalizations', []):
            hypothesis = Hypothesis(
                statement=f"Generalization: {generalization}",
                probability=learning_result.get('confidence', 0.5),
                metadata={
                    'source': 'few_shot_learning',
                    'learning_data': learning_result
                }
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def generate_conclusions_from_patterns(self, 
                                               patterns: List[Dict[str, Any]], 
                                               context: ReasoningContext) -> List[Conclusion]:
        """Generate conclusions from identified patterns."""
        conclusions = []
        
        # High-confidence patterns become conclusions
        high_conf_patterns = [p for p in patterns if p.get('confidence', 0) > 0.8]
        
        for pattern in high_conf_patterns:
            conclusion_statement = f"Strong pattern detected: {pattern['pattern']}"
            
            conclusion = Conclusion(
                statement=conclusion_statement,
                confidence=pattern['confidence'],
                reasoning_type=self.reasoning_type,
                reasoning_chain=[f"Pattern identification: {pattern['type']}"],
                metadata={
                    'pattern_data': pattern
                }
            )
            conclusions.append(conclusion)
        
        return conclusions
    
    async def generate_conclusions_from_cases(self, 
                                            similar_cases: List[Dict[str, Any]], 
                                            context: ReasoningContext) -> List[Conclusion]:
        """Generate conclusions from similar cases."""
        conclusions = []
        
        if not similar_cases:
            return conclusions
        
        # Aggregate solutions from similar cases
        for case in similar_cases:
            if case.get('success', False) and case.get('similarity', 0) > 0.8:
                solution = case.get('solution', {})
                
                conclusion_statement = f"Based on similar case: {solution.get('description', 'Solution found')}"
                
                conclusion = Conclusion(
                    statement=conclusion_statement,
                    confidence=case['similarity'] * case.get('success_rate', 1.0),
                    reasoning_type=self.reasoning_type,
                    reasoning_chain=[f"Case-based reasoning: Case {case['id']}"],
                    metadata={
                        'case_data': case,
                        'solution': solution
                    }
                )
                conclusions.append(conclusion)
        
        return conclusions