"""
Feriq Reasoning Usage Guide

This guide demonstrates how to use the comprehensive reasoning system in Feriq
for intelligent agent problem-solving and collaboration.
"""

# Basic Usage Examples

## 1. Simple Agent Reasoning

```python
from feriq.core.reasoning_integration import ReasoningMixin, ReasoningAgent
from feriq.core.agents import BaseAgent
from feriq.reasoning import ReasoningType

# Option A: Use ReasoningMixin with existing agents
class MyAgent(ReasoningMixin, BaseAgent):
    def __init__(self, name, role):
        super().__init__(name, role)

# Create and use reasoning agent
agent = MyAgent("Dr_Watson", "diagnostician")

# Simple thinking
result = await agent.think(
    problem="Patient has fever and cough",
    evidence=["Temperature: 101.5°F", "Persistent cough", "Shortness of breath"],
    reasoning_types=[ReasoningType.ABDUCTIVE, ReasoningType.PROBABILISTIC]
)

print(f"Insights: {result['insights']}")
print(f"Confidence: {result['confidence']}")

# Option B: Use dedicated ReasoningAgent
reasoning_agent = ReasoningAgent("Sherlock", "detective", expertise=["deduction", "observation"])

solution = await reasoning_agent.solve_problem(
    problem="Who committed the crime?",
    context={
        "suspects": ["Alice", "Bob", "Charlie"],
        "evidence": "Fingerprints on the weapon match Alice",
        "alibis": "Bob was at work, Charlie was at home",
        "motive": "Alice had financial dispute with victim"
    }
)

print(f"Solution: {solution}")
```

## 2. Specific Reasoning Types

```python
# Inductive reasoning - find patterns
patterns = await agent.reason_inductively(
    examples=[
        "Company A launched product in Q1, sales increased 30%",
        "Company B launched product in Q2, sales increased 25%", 
        "Company C launched product in Q3, sales increased 35%"
    ],
    pattern_to_find="relationship between launch timing and sales"
)

# Deductive reasoning - logical proof
proof = await agent.reason_deductively(
    rules=[
        "If it rains, then the ground gets wet",
        "If the ground is wet, then it's slippery"
    ],
    facts=["It is raining"],
    conclusion_to_prove="The ground is slippery"
)

# Causal reasoning - cause and effect
causality = await agent.reason_causally(
    observations=[
        "Increased marketing spend in January",
        "Website traffic increased 40% in February",
        "Sales conversions up 20% in March"
    ],
    causal_query="Does marketing spend cause increased sales?"
)

# Spatial reasoning - geographic analysis
spatial_analysis = await agent.reason_spatially(
    spatial_data=[
        "Hospital A located at (10, 15)",
        "Hospital B located at (25, 30)",
        "Population center at (20, 20)",
        "Emergency response time requirement: < 10 minutes"
    ],
    spatial_query="Optimal location for new ambulance station"
)
```

## 3. Multi-Step Reasoning Pipeline

```python
# Complex analysis using reasoning pipeline
pipeline_result = await agent.multi_step_reasoning(
    problem="Predict market trends for next quarter",
    reasoning_pipeline=[
        ReasoningType.INDUCTIVE,      # Find patterns in historical data
        ReasoningType.PROBABILISTIC,  # Calculate trend probabilities  
        ReasoningType.CAUSAL,         # Understand causal factors
        ReasoningType.TEMPORAL        # Analyze time-based trends
    ]
)

print(f"Pipeline results: {pipeline_result}")
```

## 4. Collaborative Reasoning

```python
# Multi-agent collaborative reasoning
ceo = MyAgent("CEO", "strategic_leader")
cfo = MyAgent("CFO", "financial_analyst") 
cto = MyAgent("CTO", "technology_expert")

# Collaborative decision making
consensus = await ceo.collaborative_think(
    problem="Should we acquire competitor X?",
    other_agents=[cfo, cto],
    consensus_method="weighted_average"
)

print(f"Consensus reached: {consensus['consensus_reached']}")
```

## 5. CLI Usage

```bash
# List available reasoning types
feriq reason types

# Analyze a problem
feriq reason analyze -p "How to improve customer satisfaction?" \
  -e "Current NPS score: 7.2" \
  -e "Main complaints: slow response time" \
  -e "Competition has better mobile app" \
  --strategy adaptive

# Create reasoning plan
feriq reason plan -p "Launch new product successfully" \
  --strategy hierarchical \
  --save product_launch_plan.json

# Run collaborative reasoning
feriq reason collaborate \
  -p "Best pricing strategy for new service" \
  -a "Marketing_Agent" -a "Finance_Agent" -a "Sales_Agent" \
  --method delphi

# Run demonstrations
feriq reason demo --type medical
feriq reason demo --type business  
feriq reason demo --type all
```

## 6. Advanced Integration Examples

### Task-Based Reasoning
```python
from feriq.core.reasoning_integration import ReasoningMixin
from feriq.core.tasks import BaseTask

class IntelligentTask(BaseTask):
    def __init__(self, name, description, reasoning_approach="adaptive"):
        super().__init__(name, description)
        self.reasoning_approach = reasoning_approach
        self.coordinator = ReasoningCoordinator()
    
    async def execute(self, agent, **kwargs):
        # Auto-analyze task to determine reasoning needs
        reasoning_types = await self.coordinator.analyze_problem(self.description)
        
        # Execute with appropriate reasoning
        if hasattr(agent, 'think'):
            result = await agent.think(
                problem=self.description,
                evidence=[str(kwargs)],
                reasoning_types=reasoning_types
            )
            return result
        else:
            # Regular execution for non-reasoning agents
            return await super().execute(agent, **kwargs)

# Usage
intelligent_task = IntelligentTask(
    "market_analysis", 
    "Analyze market trends and predict customer behavior"
)

reasoning_agent = MyAgent("Analyst", "market_researcher")
result = await intelligent_task.execute(reasoning_agent, 
                                      market_data="...", 
                                      timeframe="Q1-2026")
```

### Workflow Integration
```python
from feriq.core.workflow import WorkflowOrchestrator

class ReasoningWorkflow:
    def __init__(self):
        self.orchestrator = WorkflowOrchestrator()
        self.reasoning_agents = []
    
    def add_reasoning_step(self, agent, reasoning_type, depends_on=None):
        """Add a reasoning step to the workflow."""
        step = {
            'agent': agent,
            'reasoning_type': reasoning_type,
            'depends_on': depends_on
        }
        self.orchestrator.add_step(step)
    
    async def execute_reasoning_workflow(self, problem, evidence=None):
        """Execute a workflow with reasoning steps."""
        results = {}
        
        for step in self.orchestrator.steps:
            agent = step['agent']
            reasoning_type = step['reasoning_type']
            
            # Wait for dependencies
            if step['depends_on']:
                await self._wait_for_dependencies(step['depends_on'], results)
            
            # Execute reasoning step
            if hasattr(agent, 'think'):
                result = await agent.think(
                    problem=problem,
                    evidence=evidence,
                    reasoning_types=[reasoning_type]
                )
                results[step['agent'].name] = result
        
        return results

# Usage
workflow = ReasoningWorkflow()
workflow.add_reasoning_step(analyst_agent, ReasoningType.INDUCTIVE)
workflow.add_reasoning_step(expert_agent, ReasoningType.CAUSAL, depends_on=['analyst'])
workflow.add_reasoning_step(strategist_agent, ReasoningType.ABDUCTIVE, depends_on=['expert'])

results = await workflow.execute_reasoning_workflow(
    problem="Strategic planning for digital transformation",
    evidence=["Current tech stack assessment", "Market analysis", "Competitor review"]
)
```

## 7. Performance and Optimization

```python
# For high-performance scenarios
from feriq.reasoning import ReasoningStrategy

# Use parallel reasoning for independent analysis
parallel_result = await agent.think(
    problem="Multi-faceted analysis",
    reasoning_types=[
        ReasoningType.PROBABILISTIC,
        ReasoningType.CAUSAL, 
        ReasoningType.TEMPORAL,
        ReasoningType.SPATIAL
    ],
    strategy=ReasoningStrategy.PARALLEL  # All reasoning types run simultaneously
)

# Use sequential for dependent reasoning
sequential_result = await agent.think(
    problem="Step-by-step analysis", 
    reasoning_types=[
        ReasoningType.INDUCTIVE,     # First: find patterns
        ReasoningType.PROBABILISTIC, # Then: quantify uncertainty  
        ReasoningType.CAUSAL,        # Then: establish causation
        ReasoningType.ABDUCTIVE      # Finally: generate explanations
    ],
    strategy=ReasoningStrategy.SEQUENTIAL
)

# Use adaptive for complex problems
adaptive_result = await agent.think(
    problem="Complex adaptive problem",
    strategy=ReasoningStrategy.ADAPTIVE  # Dynamically selects reasoning approach
)
```

## 8. Real-World Use Cases

### Medical Diagnosis
```python
medical_agent = ReasoningAgent("Dr_AI", "physician", expertise=["internal_medicine"])

diagnosis = await medical_agent.solve_problem(
    "Patient diagnosis based on symptoms and test results",
    context={
        "symptoms": ["fever", "cough", "fatigue", "shortness_of_breath"],
        "vital_signs": "BP: 140/90, HR: 95, Temp: 101.2°F",
        "lab_results": "WBC: elevated, CRP: high",
        "patient_history": "Recent travel, no chronic conditions",
        "age": "45 years old"
    }
)
# Uses: Abductive (hypothesis generation), Probabilistic (disease likelihood), 
#        Causal (symptom causation), Temporal (symptom progression)
```

### Business Strategy
```python
strategy_agent = ReasoningAgent("Strategist", "business_analyst")

strategy = await strategy_agent.solve_problem(
    "Market entry strategy for European expansion",
    context={
        "target_markets": ["Germany", "France", "UK"],
        "budget": "$5M",
        "timeline": "18 months", 
        "competition": "High in Germany, Medium in France, Low in UK",
        "regulatory": "Complex in Germany, Standard in others",
        "market_size": "Germany: $100M, France: $80M, UK: $60M"
    }
)
# Uses: Probabilistic (success likelihood), Causal (factor analysis),
#        Spatial (geographic considerations), Temporal (timing strategy)
```

### Scientific Research
```python
research_agent = ReasoningAgent("Scientist", "researcher")

research = await research_agent.solve_problem(
    "Analyze correlation between climate variables and crop yields",
    context={
        "variables": ["temperature", "precipitation", "CO2_levels", "crop_yield"],
        "data_period": "2000-2025",
        "geographic_scope": "Global agricultural regions",
        "hypothesis": "Rising CO2 initially helps crops but temperature rises hurt yields"
    }
)
# Uses: Inductive (pattern finding), Causal (variable relationships),
#        Temporal (trend analysis), Probabilistic (prediction confidence)
```

This comprehensive reasoning system makes Feriq agents incredibly intelligent and capable of human-like problem-solving across diverse domains! 🧠🚀