# Reasoning-Enhanced Planning Integration Guide

This guide demonstrates how Feriq's planner leverages comprehensive reasoning engines to create intelligent, optimized plans across various domains.

## Overview

Feriq's `ReasoningEnhancedPlanDesigner` integrates multiple reasoning types to enhance planning capabilities:

- **Causal Reasoning**: Analyzes cause-effect relationships and dependencies
- **Probabilistic Reasoning**: Manages uncertainty and risk assessment
- **Temporal Reasoning**: Optimizes timing and scheduling
- **Spatial Reasoning**: Handles geographic and resource distribution
- **Collaborative Reasoning**: Manages stakeholder alignment and consensus
- **Inductive Reasoning**: Learns from patterns and historical data
- **Hybrid Reasoning**: Combines multiple reasoning types adaptively

## Quick Start

### Basic Usage

```python
from feriq.components.reasoning_plan_designer import (
    ReasoningEnhancedPlanDesigner,
    ReasoningPlanningStrategy,
    ReasoningPlanContext
)
from feriq.core.goal import Goal, GoalType, GoalPriority
from datetime import datetime, timedelta

# Initialize the reasoning-enhanced planner
planner = ReasoningEnhancedPlanDesigner()

# Create a goal
goal = Goal(
    description="Develop AI-powered customer service system",
    goal_type=GoalType.DEVELOPMENT,
    priority=GoalPriority.HIGH,
    deadline=datetime.now() + timedelta(days=90)
)

# Define planning context
context = ReasoningPlanContext(
    resource_constraints={
        "team_size": "8 developers",
        "budget": "$300,000",
        "technology": "Python, React, PostgreSQL"
    },
    risk_factors=[
        "Integration complexity",
        "Performance requirements",
        "User adoption challenges"
    ],
    stakeholder_preferences={
        "CTO": "Focus on scalability",
        "Product Manager": "Prioritize user experience",
        "Support Team": "Ensure easy maintenance"
    }
)

# Generate intelligent plan
plan = await planner.design_intelligent_plan(
    goal=goal,
    reasoning_strategy=ReasoningPlanningStrategy.HYBRID_INTELLIGENT,
    planning_context=context
)

print(f"Plan created with {len(plan.tasks)} optimized tasks")
```

## Reasoning Strategies

### 1. Causal Optimized Strategy

Best for: Complex projects with many dependencies

```python
# Use for software development, engineering projects
plan = await planner.design_intelligent_plan(
    goal=development_goal,
    reasoning_strategy=ReasoningPlanningStrategy.CAUSAL_OPTIMIZED,
    planning_context=context
)

# The planner will:
# - Analyze task dependencies using causal reasoning
# - Identify critical path and bottlenecks
# - Optimize task ordering based on cause-effect relationships
# - Predict downstream impacts of decisions
```

### 2. Probabilistic Risk Strategy

Best for: High-risk projects, medical research, financial planning

```python
# Use for research projects, clinical trials, investment planning
plan = await planner.design_intelligent_plan(
    goal=research_goal,
    reasoning_strategy=ReasoningPlanningStrategy.PROBABILISTIC_RISK,
    planning_context=context
)

# The planner will:
# - Calculate probability distributions for risks
# - Generate mitigation strategies with confidence levels
# - Optimize resource allocation under uncertainty
# - Create contingency plans for different scenarios
```

### 3. Temporal Sequenced Strategy

Best for: Time-sensitive projects, event planning, manufacturing

```python
# Use for project schedules, event coordination, production planning
plan = await planner.design_intelligent_plan(
    goal=timing_critical_goal,
    reasoning_strategy=ReasoningPlanningStrategy.TEMPORAL_SEQUENCED,
    planning_context=context
)

# The planner will:
# - Optimize temporal relationships between tasks
# - Consider time constraints and deadlines
# - Balance parallel vs sequential execution
# - Account for temporal dependencies and waiting periods
```

### 4. Spatial Distributed Strategy

Best for: Multi-location projects, supply chain, distributed teams

```python
# Use for global projects, logistics, distributed development
plan = await planner.design_intelligent_plan(
    goal=distributed_goal,
    reasoning_strategy=ReasoningPlanningStrategy.SPATIAL_DISTRIBUTED,
    planning_context=context
)

# The planner will:
# - Optimize geographic distribution of resources
# - Consider location-based constraints and opportunities
# - Balance transportation and communication costs
# - Account for time zones and regional differences
```

### 5. Collaborative Consensus Strategy

Best for: Multi-stakeholder projects, organizational changes

```python
# Use for strategic initiatives, organizational transformations
plan = await planner.design_intelligent_plan(
    goal=strategic_goal,
    reasoning_strategy=ReasoningPlanningStrategy.COLLABORATIVE_CONSENSUS,
    planning_context=context
)

# The planner will:
# - Analyze stakeholder preferences and conflicts
# - Generate consensus-building strategies
# - Balance competing interests and priorities
# - Create alignment mechanisms and communication plans
```

### 6. Inductive Learned Strategy

Best for: Data-rich environments, adaptive systems, educational projects

```python
# Use for AI/ML projects, educational systems, adaptive platforms
plan = await planner.design_intelligent_plan(
    goal=learning_goal,
    reasoning_strategy=ReasoningPlanningStrategy.INDUCTIVE_LEARNED,
    planning_context=context
)

# The planner will:
# - Learn from historical project patterns
# - Identify successful strategies from similar projects
# - Adapt planning based on domain-specific insights
# - Generate data-driven recommendations
```

### 7. Hybrid Intelligent Strategy

Best for: Complex, multi-faceted projects requiring multiple reasoning types

```python
# Use for comprehensive projects with multiple challenges
plan = await planner.design_intelligent_plan(
    goal=complex_goal,
    reasoning_strategy=ReasoningPlanningStrategy.HYBRID_INTELLIGENT,
    planning_context=context
)

# The planner will:
# - Apply multiple reasoning types adaptively
# - Select best reasoning approach for each aspect
# - Combine insights from different reasoning engines
# - Provide comprehensive, multi-dimensional optimization
```

## CLI Usage

Feriq provides comprehensive CLI commands for reasoning-enhanced planning:

### List Available Strategies

```bash
feriq plan strategies
```

### Create Intelligent Plan

```bash
# Basic plan creation
feriq plan create \
  --goal "Develop mobile app for healthcare monitoring" \
  --type development \
  --priority high \
  --deadline 60 \
  --strategy hybrid_intelligent

# Advanced plan with constraints
feriq plan create \
  --goal "Launch fintech startup" \
  --type strategy \
  --priority critical \
  --deadline 365 \
  --strategy collaborative_consensus \
  --resources "funding:$10M" \
  --resources "team:25_people" \
  --risks "Market competition" \
  --risks "Regulatory changes" \
  --stakeholders "CEO:market_focus" \
  --stakeholders "CTO:technical_excellence"
```

### Analyze Planning Requirements

```bash
feriq plan analyze \
  --goal "Implement enterprise AI solution" \
  --context "Legacy system integration" \
  --context "Multi-department coordination"
```

### Run Planning Demonstrations

```bash
# Run all demos
feriq plan demo --type all

# Run specific demo
feriq plan demo --type software
feriq plan demo --type medical
feriq plan demo --type supply-chain
```

## Advanced Integration

### Custom Reasoning Integration

```python
from feriq.core.reasoning_integration import ReasoningMixin

class CustomPlanDesigner(ReasoningMixin):
    def __init__(self):
        super().__init__()
        self.planner = ReasoningEnhancedPlanDesigner()
    
    async def create_domain_specific_plan(self, goal, domain_context):
        # Apply domain-specific reasoning
        reasoning_results = await self.apply_multiple_reasoning(
            query=f"Plan for {goal.description}",
            reasoning_types=[ReasoningType.CAUSAL, ReasoningType.PROBABILISTIC],
            context=domain_context
        )
        
        # Use reasoning results to enhance planning context
        enhanced_context = self.enhance_planning_context(
            domain_context, reasoning_results
        )
        
        # Generate intelligent plan
        return await self.planner.design_intelligent_plan(
            goal=goal,
            reasoning_strategy=ReasoningPlanningStrategy.HYBRID_INTELLIGENT,
            planning_context=enhanced_context
        )
```

### Integration with Existing Workflows

```python
from feriq.components.plan_designer import PlanDesigner
from feriq.components.reasoning_plan_designer import ReasoningEnhancedPlanDesigner

class AdaptivePlanDesigner:
    def __init__(self):
        self.standard_planner = PlanDesigner()
        self.reasoning_planner = ReasoningEnhancedPlanDesigner()
    
    async def design_plan(self, goal, use_reasoning=True, **kwargs):
        if use_reasoning and self.should_use_reasoning(goal):
            # Use reasoning-enhanced planning for complex goals
            return await self.reasoning_planner.design_intelligent_plan(
                goal=goal, **kwargs
            )
        else:
            # Use standard planning for simple goals
            return await self.standard_planner.design_plan(goal)
    
    def should_use_reasoning(self, goal):
        # Determine when to use reasoning enhancement
        complexity_indicators = [
            len(goal.description.split()) > 10,  # Complex description
            goal.priority in [GoalPriority.HIGH, GoalPriority.CRITICAL],
            goal.goal_type in [GoalType.RESEARCH, GoalType.STRATEGY]
        ]
        return any(complexity_indicators)
```

## Best Practices

### 1. Strategy Selection

- **Use Causal Optimized** for projects with complex dependencies
- **Use Probabilistic Risk** for high-uncertainty environments
- **Use Temporal Sequenced** for time-critical projects
- **Use Spatial Distributed** for multi-location projects
- **Use Collaborative Consensus** for multi-stakeholder initiatives
- **Use Inductive Learned** for data-rich, adaptive projects
- **Use Hybrid Intelligent** when unsure or for complex projects

### 2. Context Preparation

```python
# Provide comprehensive context for better reasoning
context = ReasoningPlanContext(
    resource_constraints={
        "team_size": "specific number and skills",
        "budget": "detailed budget breakdown",
        "technology": "specific tech stack and versions",
        "timeline": "key milestones and deadlines"
    },
    risk_factors=[
        "Specific, actionable risk descriptions",
        "Quantified impact and probability when possible"
    ],
    stakeholder_preferences={
        "role": "specific, measurable preferences",
        "department": "clear priority statements"
    }
)
```

### 3. Result Analysis

```python
# Analyze reasoning insights from generated plans
insights = plan.metadata.get('reasoning_insights', {})

# Review causal relationships
dependencies = insights.get('dependencies', [])
print(f"Critical dependencies identified: {len(dependencies)}")

# Review risk assessments
risk_analysis = insights.get('risk_assessment', {})
high_risk_items = [r for r, p in risk_analysis.items() if p > 0.7]

# Review optimization recommendations
optimizations = insights.get('optimizations', [])
print(f"Optimization opportunities: {len(optimizations)}")
```

### 4. Iterative Improvement

```python
async def iterative_planning(goal, initial_context):
    # Generate initial plan
    plan_v1 = await planner.design_intelligent_plan(
        goal=goal,
        reasoning_strategy=ReasoningPlanningStrategy.HYBRID_INTELLIGENT,
        planning_context=initial_context
    )
    
    # Analyze results and refine context
    insights = plan_v1.metadata.get('reasoning_insights', {})
    refined_context = refine_context_based_on_insights(initial_context, insights)
    
    # Generate improved plan
    plan_v2 = await planner.design_intelligent_plan(
        goal=goal,
        reasoning_strategy=determine_optimal_strategy(insights),
        planning_context=refined_context
    )
    
    return plan_v2
```

## Performance Considerations

- **Reasoning Intensity**: More complex reasoning strategies require more computation
- **Context Size**: Larger contexts provide better reasoning but take more time
- **Caching**: Results from reasoning engines are cached for repeated queries
- **Async Operation**: All reasoning operations are asynchronous for better performance

## Troubleshooting

### Common Issues

1. **Empty Reasoning Results**: Ensure context provides sufficient information
2. **Strategy Selection**: Use `analyze` command to get strategy recommendations
3. **Performance**: Start with simpler strategies for performance-critical applications
4. **Integration**: Check dependencies and import paths for reasoning modules

### Debug Mode

```python
# Enable detailed reasoning logs
planner = ReasoningEnhancedPlanDesigner(debug=True)

# Access detailed reasoning traces
plan = await planner.design_intelligent_plan(goal, strategy, context)
reasoning_trace = plan.metadata.get('reasoning_trace', {})
print(f"Reasoning steps taken: {reasoning_trace}")
```

## Conclusion

Feriq's reasoning-enhanced planning provides intelligent, optimized plans by leveraging comprehensive reasoning engines. The system adapts to different domains and challenges, providing sophisticated analysis and optimization capabilities that go far beyond traditional planning approaches.

For more examples and demonstrations, run:
```bash
feriq plan demo --type all
python -m feriq.demos.intelligent_planning_demo
```