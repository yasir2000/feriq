"""
Reasoning-Enhanced Plan Designer

This module extends the Feriq Plan Designer with comprehensive reasoning capabilities,
making planning more intelligent through causal analysis, temporal optimization,
probabilistic assessment, and collaborative decision-making.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import asyncio

from ..components.plan_designer import PlanDesigner, PlanningStrategy, PlanTemplate
from ..core.goal import Goal, GoalType, GoalStatus
from ..core.task import FeriqTask, TaskStatus, TaskPriority, TaskComplexity
from ..core.plan import Plan, PlanStatus, Milestone
from ..reasoning import (
    ReasoningCoordinator, ReasoningType, ReasoningStrategy, ReasoningContext,
    Evidence, Hypothesis, ReasoningPlan, ReasoningResult
)


class ReasoningPlanningStrategy(str, Enum):
    """Enhanced planning strategies using reasoning engines."""
    CAUSAL_OPTIMIZED = "causal_optimized"          # Use causal reasoning for dependencies
    PROBABILISTIC_RISK = "probabilistic_risk"      # Use probabilistic reasoning for risk assessment
    TEMPORAL_SEQUENCED = "temporal_sequenced"      # Use temporal reasoning for sequencing
    SPATIAL_DISTRIBUTED = "spatial_distributed"    # Use spatial reasoning for resource allocation
    COLLABORATIVE_CONSENSUS = "collaborative_consensus"  # Use collaborative reasoning for planning
    INDUCTIVE_LEARNED = "inductive_learned"        # Use inductive reasoning from past plans
    HYBRID_INTELLIGENT = "hybrid_intelligent"      # Use multiple reasoning types adaptively


@dataclass
class ReasoningPlanContext:
    """Context for reasoning-enhanced planning."""
    historical_plans: List[Dict[str, Any]] = field(default_factory=list)
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    stakeholder_preferences: Dict[str, Any] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)


class ReasoningEnhancedPlanDesigner(PlanDesigner):
    """
    Plan Designer enhanced with comprehensive reasoning capabilities.
    
    Uses multiple reasoning types to create more intelligent, optimized plans:
    - Causal reasoning for understanding task dependencies and effects
    - Temporal reasoning for optimal scheduling and sequencing
    - Probabilistic reasoning for risk assessment and success prediction
    - Spatial reasoning for resource allocation and distribution
    - Inductive reasoning for learning from historical plan data
    - Collaborative reasoning for multi-stakeholder planning
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.reasoning_coordinator = ReasoningCoordinator()
        self.planning_history = []
        self.learned_patterns = {}
        
    async def design_intelligent_plan(self, 
                                    goal: Goal, 
                                    reasoning_strategy: ReasoningPlanningStrategy = ReasoningPlanningStrategy.HYBRID_INTELLIGENT,
                                    planning_context: Optional[ReasoningPlanContext] = None) -> Plan:
        """
        Design a plan using enhanced reasoning capabilities.
        
        Args:
            goal: The goal to create a plan for
            reasoning_strategy: The reasoning approach to use
            planning_context: Additional context for reasoning
        
        Returns:
            An optimized plan created using reasoning engines
        """
        
        if planning_context is None:
            planning_context = ReasoningPlanContext()
        
        self.logger.info(f"Designing intelligent plan for goal: {goal.description}")
        self.logger.info(f"Using reasoning strategy: {reasoning_strategy}")
        
        # Phase 1: Analyze goal using appropriate reasoning
        goal_analysis = await self._analyze_goal_with_reasoning(goal, planning_context)
        
        # Phase 2: Generate plan components using reasoning
        plan_components = await self._generate_plan_components(goal, goal_analysis, reasoning_strategy, planning_context)
        
        # Phase 3: Optimize plan using reasoning engines
        optimized_plan = await self._optimize_plan_with_reasoning(plan_components, reasoning_strategy, planning_context)
        
        # Phase 4: Validate plan using deductive reasoning
        validation_result = await self._validate_plan_logically(optimized_plan)
        
        if not validation_result['valid']:
            self.logger.warning(f"Plan validation issues: {validation_result['issues']}")
            # Attempt to fix issues
            optimized_plan = await self._fix_plan_issues(optimized_plan, validation_result['issues'])
        
        # Store planning experience for future learning
        self._store_planning_experience(goal, optimized_plan, reasoning_strategy, planning_context)
        
        return optimized_plan
    
    async def _analyze_goal_with_reasoning(self, goal: Goal, context: ReasoningPlanContext) -> Dict[str, Any]:
        """Analyze goal using multiple reasoning types to understand requirements."""
        
        # Create evidence from goal and context
        evidence_items = [
            Evidence(content=f"Goal: {goal.description}", source="goal_definition"),
            Evidence(content=f"Goal type: {goal.goal_type.value}", source="goal_metadata"),
            Evidence(content=f"Priority: {goal.priority}", source="goal_metadata"),
            Evidence(content=f"Deadline: {goal.deadline}", source="goal_metadata")
        ]
        
        # Add context evidence
        if context.risk_factors:
            evidence_items.append(Evidence(content=f"Risk factors: {', '.join(context.risk_factors)}", source="risk_assessment"))
        
        if context.resource_constraints:
            evidence_items.append(Evidence(content=f"Resource constraints: {context.resource_constraints}", source="resource_planning"))
        
        # Create reasoning context
        reasoning_context = ReasoningContext(
            problem=f"Analyze planning requirements for goal: {goal.description}",
            evidence=evidence_items,
            metadata={
                'goal_id': goal.id,
                'goal_type': goal.goal_type.value,
                'analysis_purpose': 'planning_preparation'
            }
        )
        
        # Use multiple reasoning types for comprehensive analysis
        analysis_results = await self.reasoning_coordinator.reason(
            reasoning_context,
            reasoning_types=[
                ReasoningType.ABDUCTIVE,      # Generate hypotheses about requirements
                ReasoningType.CAUSAL,         # Understand cause-effect relationships
                ReasoningType.PROBABILISTIC,  # Assess success likelihood
                ReasoningType.INDUCTIVE       # Learn from similar past goals
            ],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Synthesize analysis results
        requirements = []
        risk_assessment = {}
        success_factors = []
        dependencies = []
        
        for reasoning_type, result in analysis_results.items():
            if result.success:
                for conclusion in result.conclusions:
                    if reasoning_type == ReasoningType.ABDUCTIVE:
                        requirements.append(conclusion.statement)
                    elif reasoning_type == ReasoningType.CAUSAL:
                        dependencies.append(conclusion.statement)
                    elif reasoning_type == ReasoningType.PROBABILISTIC:
                        risk_assessment[conclusion.statement] = conclusion.confidence
                    elif reasoning_type == ReasoningType.INDUCTIVE:
                        success_factors.append(conclusion.statement)
        
        return {
            'requirements': requirements,
            'dependencies': dependencies,
            'risk_assessment': risk_assessment,
            'success_factors': success_factors,
            'analysis_confidence': self._calculate_analysis_confidence(analysis_results)
        }
    
    async def _generate_plan_components(self, goal: Goal, analysis: Dict[str, Any], 
                                      strategy: ReasoningPlanningStrategy, 
                                      context: ReasoningPlanContext) -> Dict[str, Any]:
        """Generate plan components using reasoning-informed approach."""
        
        # Start with base plan generation
        base_plan = await super().design_plan(goal)
        
        # Enhance with reasoning-based improvements
        if strategy == ReasoningPlanningStrategy.CAUSAL_OPTIMIZED:
            enhanced_tasks = await self._generate_causal_optimized_tasks(base_plan, analysis)
        
        elif strategy == ReasoningPlanningStrategy.TEMPORAL_SEQUENCED:
            enhanced_tasks = await self._generate_temporal_sequenced_tasks(base_plan, analysis)
        
        elif strategy == ReasoningPlanningStrategy.PROBABILISTIC_RISK:
            enhanced_tasks = await self._generate_risk_aware_tasks(base_plan, analysis)
        
        elif strategy == ReasoningPlanningStrategy.SPATIAL_DISTRIBUTED:
            enhanced_tasks = await self._generate_spatially_distributed_tasks(base_plan, analysis, context)
        
        elif strategy == ReasoningPlanningStrategy.COLLABORATIVE_CONSENSUS:
            enhanced_tasks = await self._generate_collaborative_tasks(base_plan, analysis, context)
        
        elif strategy == ReasoningPlanningStrategy.INDUCTIVE_LEARNED:
            enhanced_tasks = await self._generate_learned_pattern_tasks(base_plan, analysis, context)
        
        else:  # HYBRID_INTELLIGENT
            enhanced_tasks = await self._generate_hybrid_intelligent_tasks(base_plan, analysis, context)
        
        return {
            'base_plan': base_plan,
            'enhanced_tasks': enhanced_tasks,
            'reasoning_insights': analysis,
            'strategy_used': strategy
        }
    
    async def _generate_causal_optimized_tasks(self, base_plan: Plan, analysis: Dict[str, Any]) -> List[FeriqTask]:
        """Generate tasks optimized using causal reasoning."""
        
        # Analyze causal relationships between tasks
        causal_context = ReasoningContext(
            problem="Optimize task dependencies using causal relationships",
            evidence=[
                Evidence(content=f"Base plan tasks: {[task.name for task in base_plan.tasks]}", source="base_plan"),
                Evidence(content=f"Dependencies: {analysis.get('dependencies', [])}", source="goal_analysis")
            ],
            metadata={'optimization_type': 'causal'}
        )
        
        causal_results = await self.reasoning_coordinator.reason(
            causal_context,
            reasoning_types=[ReasoningType.CAUSAL],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Apply causal insights to task optimization
        optimized_tasks = []
        for task in base_plan.tasks:
            # Create enhanced task with causal insights
            enhanced_task = FeriqTask(
                name=task.name,
                description=task.description,
                complexity=task.complexity,
                priority=task.priority,
                agent_role=task.agent_role,
                dependencies=task.dependencies,
                estimated_duration=task.estimated_duration
            )
            
            # Add causal reasoning metadata
            enhanced_task.metadata = enhanced_task.metadata or {}
            enhanced_task.metadata['causal_optimization'] = True
            enhanced_task.metadata['causal_insights'] = [
                conclusion.statement for result in causal_results.values() 
                if result.success for conclusion in result.conclusions
            ]
            
            optimized_tasks.append(enhanced_task)
        
        return optimized_tasks
    
    async def _generate_temporal_sequenced_tasks(self, base_plan: Plan, analysis: Dict[str, Any]) -> List[FeriqTask]:
        """Generate tasks with optimal temporal sequencing."""
        
        temporal_context = ReasoningContext(
            problem="Optimize task sequencing and timing",
            evidence=[
                Evidence(content=f"Task durations: {[(task.name, task.estimated_duration) for task in base_plan.tasks]}", source="timing_data"),
                Evidence(content=f"Deadline: {base_plan.deadline}", source="constraints")
            ],
            metadata={'optimization_type': 'temporal'}
        )
        
        temporal_results = await self.reasoning_coordinator.reason(
            temporal_context,
            reasoning_types=[ReasoningType.TEMPORAL],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Apply temporal optimization
        sequenced_tasks = []
        for i, task in enumerate(base_plan.tasks):
            enhanced_task = FeriqTask(
                name=task.name,
                description=task.description,
                complexity=task.complexity,
                priority=task.priority,
                agent_role=task.agent_role,
                dependencies=task.dependencies,
                estimated_duration=task.estimated_duration
            )
            
            # Add temporal optimization
            enhanced_task.metadata = enhanced_task.metadata or {}
            enhanced_task.metadata['temporal_optimization'] = True
            enhanced_task.metadata['optimal_sequence_position'] = i
            enhanced_task.metadata['temporal_insights'] = [
                conclusion.statement for result in temporal_results.values()
                if result.success for conclusion in result.conclusions
            ]
            
            sequenced_tasks.append(enhanced_task)
        
        return sequenced_tasks
    
    async def _generate_risk_aware_tasks(self, base_plan: Plan, analysis: Dict[str, Any]) -> List[FeriqTask]:
        """Generate tasks with probabilistic risk assessment."""
        
        risk_context = ReasoningContext(
            problem="Assess and mitigate task execution risks",
            evidence=[
                Evidence(content=f"Risk factors: {analysis.get('risk_assessment', {})}", source="risk_analysis"),
                Evidence(content=f"Task complexities: {[(task.name, task.complexity) for task in base_plan.tasks]}", source="complexity_analysis")
            ],
            metadata={'optimization_type': 'risk_assessment'}
        )
        
        risk_results = await self.reasoning_coordinator.reason(
            risk_context,
            reasoning_types=[ReasoningType.PROBABILISTIC],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Apply risk-aware modifications
        risk_aware_tasks = []
        for task in base_plan.tasks:
            enhanced_task = FeriqTask(
                name=task.name,
                description=task.description,
                complexity=task.complexity,
                priority=task.priority,
                agent_role=task.agent_role,
                dependencies=task.dependencies,
                estimated_duration=task.estimated_duration
            )
            
            # Add risk assessment
            enhanced_task.metadata = enhanced_task.metadata or {}
            enhanced_task.metadata['risk_assessment'] = True
            enhanced_task.metadata['risk_score'] = min(1.0, task.complexity + 0.2)  # Simple risk calculation
            enhanced_task.metadata['risk_insights'] = [
                conclusion.statement for result in risk_results.values()
                if result.success for conclusion in result.conclusions
            ]
            
            # Add contingency planning for high-risk tasks
            if enhanced_task.metadata['risk_score'] > 0.7:
                enhanced_task.metadata['contingency_plan'] = f"Alternative approach for {task.name}"
            
            risk_aware_tasks.append(enhanced_task)
        
        return risk_aware_tasks
    
    async def _generate_spatially_distributed_tasks(self, base_plan: Plan, analysis: Dict[str, Any], 
                                                  context: ReasoningPlanContext) -> List[FeriqTask]:
        """Generate tasks with spatial reasoning for resource distribution."""
        
        spatial_context = ReasoningContext(
            problem="Optimize spatial distribution of tasks and resources",
            evidence=[
                Evidence(content=f"Resource constraints: {context.resource_constraints}", source="resource_data"),
                Evidence(content=f"Environmental factors: {context.environmental_factors}", source="environment_data")
            ],
            metadata={'optimization_type': 'spatial'}
        )
        
        spatial_results = await self.reasoning_coordinator.reason(
            spatial_context,
            reasoning_types=[ReasoningType.SPATIAL],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Apply spatial optimization
        spatially_optimized_tasks = []
        for task in base_plan.tasks:
            enhanced_task = FeriqTask(
                name=task.name,
                description=task.description,
                complexity=task.complexity,
                priority=task.priority,
                agent_role=task.agent_role,
                dependencies=task.dependencies,
                estimated_duration=task.estimated_duration
            )
            
            enhanced_task.metadata = enhanced_task.metadata or {}
            enhanced_task.metadata['spatial_optimization'] = True
            enhanced_task.metadata['resource_allocation'] = f"Optimized for {task.agent_role}"
            enhanced_task.metadata['spatial_insights'] = [
                conclusion.statement for result in spatial_results.values()
                if result.success for conclusion in result.conclusions
            ]
            
            spatially_optimized_tasks.append(enhanced_task)
        
        return spatially_optimized_tasks
    
    async def _generate_collaborative_tasks(self, base_plan: Plan, analysis: Dict[str, Any], 
                                          context: ReasoningPlanContext) -> List[FeriqTask]:
        """Generate tasks using collaborative reasoning with stakeholders."""
        
        collaborative_context = ReasoningContext(
            problem="Design collaborative task execution plan",
            evidence=[
                Evidence(content=f"Stakeholder preferences: {context.stakeholder_preferences}", source="stakeholder_input"),
                Evidence(content=f"Success metrics: {context.success_metrics}", source="success_criteria")
            ],
            metadata={'optimization_type': 'collaborative', 'collaboration_type': 'consensus'}
        )
        
        collaborative_results = await self.reasoning_coordinator.reason(
            collaborative_context,
            reasoning_types=[ReasoningType.COLLABORATIVE],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Apply collaborative insights
        collaborative_tasks = []
        for task in base_plan.tasks:
            enhanced_task = FeriqTask(
                name=task.name,
                description=task.description,
                complexity=task.complexity,
                priority=task.priority,
                agent_role=task.agent_role,
                dependencies=task.dependencies,
                estimated_duration=task.estimated_duration
            )
            
            enhanced_task.metadata = enhanced_task.metadata or {}
            enhanced_task.metadata['collaborative_design'] = True
            enhanced_task.metadata['stakeholder_consensus'] = True
            enhanced_task.metadata['collaborative_insights'] = [
                conclusion.statement for result in collaborative_results.values()
                if result.success for conclusion in result.conclusions
            ]
            
            collaborative_tasks.append(enhanced_task)
        
        return collaborative_tasks
    
    async def _generate_learned_pattern_tasks(self, base_plan: Plan, analysis: Dict[str, Any], 
                                            context: ReasoningPlanContext) -> List[FeriqTask]:
        """Generate tasks using inductive learning from historical patterns."""
        
        learning_context = ReasoningContext(
            problem="Apply learned patterns from historical planning data",
            evidence=[
                Evidence(content=f"Historical plans: {len(context.historical_plans)} examples", source="historical_data"),
                Evidence(content=f"Current goal type: {base_plan.goal.goal_type.value}", source="current_context")
            ],
            metadata={'optimization_type': 'pattern_learning'}
        )
        
        learning_results = await self.reasoning_coordinator.reason(
            learning_context,
            reasoning_types=[ReasoningType.INDUCTIVE],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Apply learned patterns
        pattern_enhanced_tasks = []
        for task in base_plan.tasks:
            enhanced_task = FeriqTask(
                name=task.name,
                description=task.description,
                complexity=task.complexity,
                priority=task.priority,
                agent_role=task.agent_role,
                dependencies=task.dependencies,
                estimated_duration=task.estimated_duration
            )
            
            enhanced_task.metadata = enhanced_task.metadata or {}
            enhanced_task.metadata['pattern_learning'] = True
            enhanced_task.metadata['learned_optimizations'] = [
                conclusion.statement for result in learning_results.values()
                if result.success for conclusion in result.conclusions
            ]
            
            pattern_enhanced_tasks.append(enhanced_task)
        
        return pattern_enhanced_tasks
    
    async def _generate_hybrid_intelligent_tasks(self, base_plan: Plan, analysis: Dict[str, Any], 
                                               context: ReasoningPlanContext) -> List[FeriqTask]:
        """Generate tasks using hybrid reasoning with multiple intelligence types."""
        
        # Use adaptive reasoning strategy
        hybrid_context = ReasoningContext(
            problem="Create optimal plan using multiple reasoning approaches",
            evidence=[
                Evidence(content=f"Goal analysis: {analysis}", source="analysis_results"),
                Evidence(content=f"Planning context: {context}", source="context_data")
            ],
            metadata={'optimization_type': 'hybrid_intelligent'}
        )
        
        # Use multiple reasoning types adaptively
        hybrid_results = await self.reasoning_coordinator.reason(
            hybrid_context,
            reasoning_types=[
                ReasoningType.CAUSAL,
                ReasoningType.TEMPORAL, 
                ReasoningType.PROBABILISTIC,
                ReasoningType.INDUCTIVE
            ],
            strategy=ReasoningStrategy.ADAPTIVE
        )
        
        # Apply hybrid intelligence
        hybrid_tasks = []
        for task in base_plan.tasks:
            enhanced_task = FeriqTask(
                name=task.name,
                description=task.description,
                complexity=task.complexity,
                priority=task.priority,
                agent_role=task.agent_role,
                dependencies=task.dependencies,
                estimated_duration=task.estimated_duration
            )
            
            enhanced_task.metadata = enhanced_task.metadata or {}
            enhanced_task.metadata['hybrid_intelligence'] = True
            enhanced_task.metadata['reasoning_types_applied'] = [
                rt.value for rt in hybrid_results.keys()
            ]
            enhanced_task.metadata['hybrid_insights'] = [
                conclusion.statement for result in hybrid_results.values()
                if result.success for conclusion in result.conclusions
            ]
            
            hybrid_tasks.append(enhanced_task)
        
        return hybrid_tasks
    
    async def _optimize_plan_with_reasoning(self, plan_components: Dict[str, Any], 
                                          strategy: ReasoningPlanningStrategy,
                                          context: ReasoningPlanContext) -> Plan:
        """Apply final optimization using reasoning engines."""
        
        base_plan = plan_components['base_plan']
        enhanced_tasks = plan_components['enhanced_tasks']
        
        # Create optimized plan
        optimized_plan = Plan(
            id=str(uuid.uuid4()),
            goal=base_plan.goal,
            tasks=enhanced_tasks,
            milestones=base_plan.milestones,
            dependencies=base_plan.dependencies,
            created_at=datetime.now(),
            status=PlanStatus.DRAFT
        )
        
        # Add reasoning metadata
        optimized_plan.metadata = {
            'reasoning_strategy': strategy.value,
            'reasoning_insights': plan_components['reasoning_insights'],
            'optimization_timestamp': datetime.now().isoformat(),
            'reasoning_enhanced': True
        }
        
        return optimized_plan
    
    async def _validate_plan_logically(self, plan: Plan) -> Dict[str, Any]:
        """Validate plan using deductive reasoning."""
        
        validation_context = ReasoningContext(
            problem="Validate plan logic and consistency",
            evidence=[
                Evidence(content=f"Plan tasks: {[task.name for task in plan.tasks]}", source="plan_structure"),
                Evidence(content=f"Dependencies: {plan.dependencies}", source="dependency_structure"),
                Evidence(content=f"Timeline: {plan.deadline}", source="temporal_constraints")
            ],
            metadata={'validation_type': 'logical_consistency'}
        )
        
        validation_results = await self.reasoning_coordinator.reason(
            validation_context,
            reasoning_types=[ReasoningType.DEDUCTIVE],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Check for logical issues
        issues = []
        valid = True
        
        for result in validation_results.values():
            if result.success:
                for conclusion in result.conclusions:
                    if conclusion.confidence < 0.7:
                        issues.append(f"Low confidence in: {conclusion.statement}")
                        valid = False
            else:
                issues.append(f"Validation error: {result.error_message}")
                valid = False
        
        return {
            'valid': valid,
            'issues': issues,
            'validation_results': validation_results
        }
    
    async def _fix_plan_issues(self, plan: Plan, issues: List[str]) -> Plan:
        """Attempt to fix plan issues using abductive reasoning."""
        
        fix_context = ReasoningContext(
            problem="Generate solutions for plan validation issues",
            evidence=[
                Evidence(content=f"Issues found: {issues}", source="validation_results"),
                Evidence(content=f"Current plan structure: {plan}", source="plan_data")
            ],
            metadata={'fix_type': 'plan_correction'}
        )
        
        fix_results = await self.reasoning_coordinator.reason(
            fix_context,
            reasoning_types=[ReasoningType.ABDUCTIVE],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Apply fixes (simplified implementation)
        fixed_plan = plan
        fixed_plan.metadata = fixed_plan.metadata or {}
        fixed_plan.metadata['fixes_applied'] = [
            conclusion.statement for result in fix_results.values()
            if result.success for conclusion in result.conclusions
        ]
        
        return fixed_plan
    
    def _store_planning_experience(self, goal: Goal, plan: Plan, strategy: ReasoningPlanningStrategy, 
                                 context: ReasoningPlanContext):
        """Store planning experience for future inductive learning."""
        
        experience = {
            'goal_type': goal.goal_type.value,
            'goal_description': goal.description,
            'strategy_used': strategy.value,
            'plan_success_indicators': {
                'task_count': len(plan.tasks),
                'estimated_duration': str(plan.deadline - plan.created_at),
                'complexity_distribution': [task.complexity for task in plan.tasks]
            },
            'context_factors': {
                'resource_constraints': context.resource_constraints,
                'risk_factors': context.risk_factors,
                'stakeholder_count': len(context.stakeholder_preferences)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.planning_history.append(experience)
        
        # Update learned patterns
        goal_type_key = goal.goal_type.value
        if goal_type_key not in self.learned_patterns:
            self.learned_patterns[goal_type_key] = []
        self.learned_patterns[goal_type_key].append(experience)
    
    def _calculate_analysis_confidence(self, analysis_results: Dict[ReasoningType, ReasoningResult]) -> float:
        """Calculate overall confidence from analysis results."""
        valid_results = [result for result in analysis_results.values() if result.success]
        if not valid_results:
            return 0.0
        
        total_confidence = sum(result.confidence for result in valid_results)
        return total_confidence / len(valid_results)
    
    # Additional helper methods for reasoning integration
    
    async def get_reasoning_recommendations(self, goal: Goal) -> Dict[str, Any]:
        """Get reasoning-based recommendations for planning approach."""
        
        recommendation_context = ReasoningContext(
            problem=f"Recommend best planning approach for goal: {goal.description}",
            evidence=[
                Evidence(content=f"Goal type: {goal.goal_type.value}", source="goal_metadata"),
                Evidence(content=f"Goal complexity: {goal.priority}", source="goal_assessment"),
                Evidence(content=f"Historical patterns: {len(self.planning_history)} past plans", source="experience_data")
            ],
            metadata={'recommendation_type': 'planning_strategy'}
        )
        
        recommendations = await self.reasoning_coordinator.reason(
            recommendation_context,
            reasoning_types=[ReasoningType.INDUCTIVE, ReasoningType.ABDUCTIVE],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        return {
            'recommended_strategy': self._extract_strategy_recommendation(recommendations),
            'reasoning_confidence': self._calculate_analysis_confidence(recommendations),
            'recommendations': recommendations
        }
    
    def _extract_strategy_recommendation(self, recommendations: Dict[ReasoningType, ReasoningResult]) -> ReasoningPlanningStrategy:
        """Extract strategy recommendation from reasoning results."""
        # Simplified strategy selection based on reasoning results
        
        if any(result.success and result.confidence > 0.8 for result in recommendations.values()):
            return ReasoningPlanningStrategy.HYBRID_INTELLIGENT
        
        return ReasoningPlanningStrategy.CAUSAL_OPTIMIZED