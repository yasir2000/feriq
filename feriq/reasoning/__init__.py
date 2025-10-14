"""
Feriq Reasoning System

Comprehensive reasoning engine supporting multiple reasoning types:
- Inductive Reasoning: Pattern recognition and generalization
- Deductive Reasoning: Logical inference and theorem proving
- Probabilistic Reasoning: Uncertainty quantification and Bayesian inference
- Causal Reasoning: Causal discovery and inference
- Abductive Reasoning: Inference to best explanation
- Analogical Reasoning: Structure mapping and similarity
- Temporal Reasoning: Time-based logic and sequences
- Spatial Reasoning: Geographic and geometric relationships
- Hybrid Reasoning: Neuro-symbolic and multi-modal approaches
- Collaborative Reasoning: Multi-agent coordination and consensus building
"""

from .base import (
    BaseReasoner,
    ReasoningContext,
    ReasoningResult,
    ReasoningType,
    Evidence,
    Hypothesis,
    Conclusion,
    CompositeReasoner
)

from .inductive import (
    InductiveReasoner,
    PatternRecognizer,
    FewShotLearner,
    CaseBasedReasoner
)

from .deductive import (
    DeductiveReasoner,
    LogicalInferenceEngine,
    TheoremProver,
    RuleEngine
)

from .probabilistic import (
    ProbabilisticReasoner,
    BayesianNetwork,
    MCMCReasoner,
    VariationalReasoner
)

from .causal import (
    CausalReasoner,
    PCAlgorithm,
    GESAlgorithm,
    CausalInferencer,
    InterventionPlanner
)

from .abductive import (
    AbductiveReasoner,
    HypothesisGenerator,
    ExplanationRanker,
    DiagnosticReasoner
)

from .analogical import (
    AnalogicalReasoner,
    StructureMapper
)

from .temporal import (
    TemporalReasoner,
    SequenceAnalyzer,
    EventCorrelator,
    TimeSeriesReasoner
)

from .spatial import (
    SpatialReasoner,
    GeometricReasoner,
    TopologicalReasoner,
    RegionAnalyzer,
    Point,
    SpatialRegion
)

from .hybrid import (
    HybridReasoner,
    NeuroSymbolicReasoner,
    MultiModalReasoner,
    EnsembleReasoner,
    TransformerBasedReasoner,
    HybridApproach,
    NeuralComponent,
    SymbolicRule
)

from .collaborative import (
    CollaborativeReasoner,
    ConsensusBuilder,
    DistributedProblemSolver,
    Agent,
    Argument,
    ConsensusState,
    ConsensusMethod,
    ArgumentationType
)

from .manager import (
    ReasoningManager,
    ReasoningOrchestrator,
    ReasoningCoordinator,
    ReasoningStrategy,
    ReasoningPlan,
    ReasoningSession
)

__all__ = [
    # Base classes
    'BaseReasoner',
    'ReasoningContext',
    'ReasoningResult',
    'ReasoningType',
    'Evidence',
    'Hypothesis',
    'Conclusion',
    'CompositeReasoner',
    
    # Inductive reasoning
    'InductiveReasoner',
    'PatternRecognizer',
    'FewShotLearner',
    'CaseBasedReasoner',
    
    # Deductive reasoning
    'DeductiveReasoner',
    'LogicalInferenceEngine',
    'TheoremProver',
    'RuleEngine',
    
    # Probabilistic reasoning
    'ProbabilisticReasoner',
    'BayesianNetwork',
    'MCMCReasoner',
    'VariationalReasoner',
    
    # Causal reasoning
    'CausalReasoner',
    'PCAlgorithm',
    'GESAlgorithm',
    'CausalInferencer',
    'InterventionPlanner',
    
    # Abductive reasoning
    'AbductiveReasoner',
    'HypothesisGenerator',
    'ExplanationRanker',
    'DiagnosticReasoner',
    
    # Analogical reasoning
    'AnalogicalReasoner',
    'StructureMapper',
    
    # Temporal reasoning
    'TemporalReasoner',
    'SequenceAnalyzer',
    'EventCorrelator',
    'TimeSeriesReasoner',
    
    # Spatial reasoning
    'SpatialReasoner',
    'GeometricReasoner',
    'TopologicalReasoner',
    'RegionAnalyzer',
    'Point',
    'SpatialRegion',
    
    # Hybrid reasoning
    'HybridReasoner',
    'NeuroSymbolicReasoner',
    'MultiModalReasoner',
    'EnsembleReasoner',
    'TransformerBasedReasoner',
    'HybridApproach',
    'NeuralComponent',
    'SymbolicRule',
    
    # Collaborative reasoning
    'CollaborativeReasoner',
    'ConsensusBuilder',
    'DistributedProblemSolver',
    'Agent',
    'Argument',
    'ConsensusState',
    'ConsensusMethod',
    'ArgumentationType',
    
    # Management and orchestration
    'ReasoningManager',
    'ReasoningOrchestrator',
    'ReasoningCoordinator',
    'ReasoningStrategy',
    'ReasoningPlan',
    'ReasoningSession',
]