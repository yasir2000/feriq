"""
Feriq Reasoning Integration Examples

This module demonstrates how to integrate and use the comprehensive reasoning system
within the Feriq collaborative AI agents framework.
"""

import asyncio
from typing import Dict, List, Any, Optional
from feriq.reasoning import (
    ReasoningCoordinator, ReasoningManager, ReasoningContext, ReasoningType,
    Evidence, Hypothesis, ReasoningStrategy, ReasoningPlan,
    InductiveReasoner, DeductiveReasoner, ProbabilisticReasoner,
    Agent as ReasoningAgent, CollaborativeReasoner
)
from feriq.core.agents import BaseAgent
from feriq.core.tasks import BaseTask


class ReasoningEnhancedAgent(BaseAgent):
    """Agent enhanced with comprehensive reasoning capabilities."""
    
    def __init__(self, name: str, role: str = "reasoning_agent", **kwargs):
        super().__init__(name, role, **kwargs)
        self.reasoning_coordinator = ReasoningCoordinator()
        self.preferred_reasoning_types = [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE]
    
    async def reason_about_problem(self, problem: str, evidence: List[str] = None, 
                                 reasoning_types: List[ReasoningType] = None) -> Dict[str, Any]:
        """Use reasoning system to analyze and solve problems."""
        
        # Create reasoning context
        evidence_objects = []
        if evidence:
            for i, ev in enumerate(evidence):
                evidence_objects.append(Evidence(
                    content=ev,
                    source=f"agent_{self.name}",
                    confidence=0.8
                ))
        
        context = ReasoningContext(
            problem=problem,
            evidence=evidence_objects,
            metadata={'agent_name': self.name, 'role': self.role}
        )
        
        # Use specified reasoning types or defaults
        if reasoning_types is None:
            reasoning_types = self.preferred_reasoning_types
        
        # Perform reasoning
        results = await self.reasoning_coordinator.reason(
            context, 
            reasoning_types=reasoning_types,
            strategy=ReasoningStrategy.PARALLEL
        )
        
        return {
            'problem': problem,
            'reasoning_results': results,
            'agent': self.name,
            'success': any(result.success for result in results.values())
        }
    
    async def collaborative_reasoning(self, problem: str, other_agents: List['ReasoningEnhancedAgent']) -> Dict[str, Any]:
        """Collaborate with other agents using collaborative reasoning."""
        
        # Create reasoning agents for collaborative reasoner
        reasoning_agents = []
        for agent in other_agents + [self]:
            r_agent = ReasoningAgent(
                id=agent.name,
                name=agent.name,
                expertise=[agent.role],
                weight=1.0,
                reasoner=DeductiveReasoner()  # Default reasoner
            )
            reasoning_agents.append(r_agent)
        
        # Set up collaborative reasoner
        collab_reasoner = CollaborativeReasoner()
        for agent in reasoning_agents:
            collab_reasoner.add_agent(agent)
        
        # Create context for collaborative reasoning
        context = ReasoningContext(
            problem=problem,
            evidence=[Evidence(content=problem, source="collaborative_session")],
            metadata={'collaboration_type': 'consensus'}
        )
        
        # Perform collaborative reasoning
        result = await collab_reasoner.reason(context)
        
        return {
            'collaborative_result': result,
            'participants': [agent.name for agent in other_agents + [self]],
            'consensus_reached': result.success
        }


class ReasoningTask(BaseTask):
    """Task that incorporates reasoning capabilities."""
    
    def __init__(self, name: str, description: str, reasoning_approach: str = "adaptive", **kwargs):
        super().__init__(name, description, **kwargs)
        self.reasoning_approach = reasoning_approach
        self.reasoning_coordinator = ReasoningCoordinator()
    
    async def execute_with_reasoning(self, agent: ReasoningEnhancedAgent, **kwargs) -> Dict[str, Any]:
        """Execute task using specified reasoning approach."""
        
        # Analyze problem to suggest reasoning types
        suggested_types = await self.reasoning_coordinator.analyze_problem(self.description)
        
        # Create execution context
        context = ReasoningContext(
            problem=self.description,
            evidence=[Evidence(content=str(kwargs), source="task_parameters")],
            metadata={
                'task_name': self.name,
                'reasoning_approach': self.reasoning_approach,
                'agent': agent.name
            }
        )
        
        if self.reasoning_approach == "adaptive":
            # Use adaptive reasoning strategy
            plan = self.reasoning_coordinator.create_plan(
                strategy=ReasoningStrategy.ADAPTIVE,
                reasoning_types=suggested_types
            )
            results = await self.reasoning_coordinator.reason_with_plan(context, plan)
            
        elif self.reasoning_approach == "hierarchical":
            # Use hierarchical reasoning with dependencies
            dependencies = {
                ReasoningType.PROBABILISTIC: [ReasoningType.INDUCTIVE],
                ReasoningType.CAUSAL: [ReasoningType.PROBABILISTIC],
                ReasoningType.ABDUCTIVE: [ReasoningType.CAUSAL]
            }
            plan = self.reasoning_coordinator.create_plan(
                strategy=ReasoningStrategy.HIERARCHICAL,
                reasoning_types=suggested_types,
                dependencies=dependencies
            )
            results = await self.reasoning_coordinator.reason_with_plan(context, plan)
            
        else:
            # Use parallel reasoning (default)
            results = await self.reasoning_coordinator.reason(
                context,
                reasoning_types=suggested_types,
                strategy=ReasoningStrategy.PARALLEL
            )
        
        return {
            'task_name': self.name,
            'reasoning_results': results,
            'suggested_reasoning_types': [rt.value for rt in suggested_types],
            'execution_success': any(result.success for result in results.values())
        }


# Practical usage examples
async def example_diagnostic_reasoning():
    """Example: Medical diagnostic reasoning using multiple reasoning types."""
    
    print("🔬 Medical Diagnostic Reasoning Example")
    print("=" * 50)
    
    # Create diagnostic agent
    doctor_agent = ReasoningEnhancedAgent(
        name="Dr_Watson",
        role="medical_diagnostician"
    )
    
    # Patient symptoms and history
    symptoms = [
        "Patient has fever (101.5°F)",
        "Persistent cough for 5 days",
        "Shortness of breath",
        "Recent travel to endemic area",
        "No vaccination history for target disease"
    ]
    
    # Use multiple reasoning types for diagnosis
    reasoning_types = [
        ReasoningType.ABDUCTIVE,      # Generate diagnostic hypotheses
        ReasoningType.PROBABILISTIC,  # Calculate disease probabilities
        ReasoningType.CAUSAL,         # Understand symptom causation
        ReasoningType.TEMPORAL        # Analyze symptom progression
    ]
    
    diagnosis_result = await doctor_agent.reason_about_problem(
        problem="Diagnose patient condition based on symptoms and history",
        evidence=symptoms,
        reasoning_types=reasoning_types
    )
    
    print(f"Agent: {diagnosis_result['agent']}")
    print(f"Problem: {diagnosis_result['problem']}")
    print(f"Success: {diagnosis_result['success']}")
    
    for reasoning_type, result in diagnosis_result['reasoning_results'].items():
        if result.success:
            print(f"\n{reasoning_type.value.upper()} REASONING:")
            for conclusion in result.conclusions:
                print(f"  - {conclusion.statement} (confidence: {conclusion.confidence:.2f})")
    
    return diagnosis_result


async def example_business_strategy_reasoning():
    """Example: Business strategy development using collaborative reasoning."""
    
    print("\n💼 Business Strategy Development Example")
    print("=" * 50)
    
    # Create strategic planning team
    ceo_agent = ReasoningEnhancedAgent("CEO", "strategic_leader")
    cfo_agent = ReasoningEnhancedAgent("CFO", "financial_analyst")
    cto_agent = ReasoningEnhancedAgent("CTO", "technology_strategist")
    cmo_agent = ReasoningEnhancedAgent("CMO", "market_analyst")
    
    team = [cfo_agent, cto_agent, cmo_agent]
    
    # Strategic problem
    strategic_problem = """
    Company needs to decide on market expansion strategy for Q1 2026.
    Options: 1) Expand to European markets, 2) Develop new product line, 
    3) Acquire competitor, 4) Focus on digital transformation.
    Consider: budget constraints, market conditions, competitive landscape, 
    technological capabilities, and risk factors.
    """
    
    # Collaborative strategic reasoning
    strategy_result = await ceo_agent.collaborative_reasoning(
        problem=strategic_problem,
        other_agents=team
    )
    
    print(f"Participants: {', '.join(strategy_result['participants'])}")
    print(f"Consensus Reached: {strategy_result['consensus_reached']}")
    
    if strategy_result['collaborative_result'].success:
        print("\nSTRATEGIC RECOMMENDATIONS:")
        for conclusion in strategy_result['collaborative_result'].conclusions:
            print(f"  - {conclusion.statement}")
            print(f"    Confidence: {conclusion.confidence:.2f}")
            print(f"    Reasoning: {', '.join(conclusion.reasoning_chain)}")
    
    return strategy_result


async def example_research_reasoning_task():
    """Example: Scientific research task using hierarchical reasoning."""
    
    print("\n🔬 Scientific Research Task Example")
    print("=" * 50)
    
    # Create research agent
    researcher = ReasoningEnhancedAgent(
        name="Dr_Science",
        role="research_scientist"
    )
    
    # Create research task
    research_task = ReasoningTask(
        name="climate_impact_analysis",
        description="""
        Analyze the causal relationship between industrial emissions and climate change.
        Use available climate data to identify patterns, establish causal links,
        and predict future climate scenarios under different emission policies.
        """,
        reasoning_approach="hierarchical"
    )
    
    # Execute research task
    research_result = await research_task.execute_with_reasoning(
        agent=researcher,
        dataset="climate_data_2020_2025",
        variables=["CO2_emissions", "temperature", "precipitation", "sea_level"],
        time_period="2020-2025"
    )
    
    print(f"Task: {research_result['task_name']}")
    print(f"Success: {research_result['execution_success']}")
    print(f"Suggested Reasoning Types: {', '.join(research_result['suggested_reasoning_types'])}")
    
    print("\nRESEARCH FINDINGS:")
    for reasoning_type, result in research_result['reasoning_results'].items():
        if result.success:
            print(f"\n{reasoning_type.value.upper()}:")
            for conclusion in result.conclusions:
                print(f"  - {conclusion.statement}")
    
    return research_result


async def example_spatial_reasoning():
    """Example: Urban planning using spatial reasoning."""
    
    print("\n🏙️ Urban Planning Spatial Reasoning Example")
    print("=" * 50)
    
    # Create urban planner agent
    planner = ReasoningEnhancedAgent(
        name="UrbanPlanner",
        role="city_planner"
    )
    
    # Spatial planning problem with geographic data
    spatial_problem = "Optimal location for new hospital considering population density, accessibility, and existing healthcare facilities"
    
    # Spatial evidence (mock geographic data)
    spatial_evidence = [
        "High population density in districts A, B, and C",
        "Existing hospitals located at coordinates (10,15) and (25,30)",
        "Major transportation routes along coordinates (0,20) to (50,20)",
        "Emergency services response time target: under 10 minutes",
        "Available land parcels at (15,25), (20,10), and (35,35)"
    ]
    
    # Use spatial reasoning
    spatial_result = await planner.reason_about_problem(
        problem=spatial_problem,
        evidence=spatial_evidence,
        reasoning_types=[ReasoningType.SPATIAL, ReasoningType.CAUSAL, ReasoningType.PROBABILISTIC]
    )
    
    print(f"Urban Planning Problem: {spatial_problem}")
    print(f"Success: {spatial_result['success']}")
    
    for reasoning_type, result in spatial_result['reasoning_results'].items():
        if result.success:
            print(f"\n{reasoning_type.value.upper()} ANALYSIS:")
            for conclusion in result.conclusions:
                print(f"  - {conclusion.statement}")
    
    return spatial_result


async def run_all_examples():
    """Run all reasoning integration examples."""
    
    print("🧠 FERIQ REASONING SYSTEM INTEGRATION EXAMPLES")
    print("=" * 60)
    print("Demonstrating how Feriq agents use comprehensive reasoning capabilities\n")
    
    # Run all examples
    await example_diagnostic_reasoning()
    await example_business_strategy_reasoning() 
    await example_research_reasoning_task()
    await example_spatial_reasoning()
    
    print("\n✅ All reasoning integration examples completed!")
    print("🚀 Feriq agents can now leverage 10+ reasoning types for intelligent problem-solving!")


if __name__ == "__main__":
    asyncio.run(run_all_examples())