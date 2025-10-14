"""
CLI Commands for Reasoning-Enhanced Planning

This module adds reasoning-enhanced planning capabilities to the Feriq CLI.
"""

import click
import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
from feriq.components.reasoning_plan_designer import (
    ReasoningEnhancedPlanDesigner, 
    ReasoningPlanningStrategy,
    ReasoningPlanContext
)
from feriq.core.goal import Goal, GoalType, GoalPriority


@click.group(name='plan')
@click.pass_context
def reasoning_planning_commands(ctx):
    """Reasoning-enhanced planning commands for intelligent plan generation."""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj['reasoning_planner'] = ReasoningEnhancedPlanDesigner()


@reasoning_planning_commands.command('strategies')
@click.pass_context
def list_reasoning_strategies(ctx):
    """List all available reasoning-enhanced planning strategies."""
    
    click.echo("🧠 Reasoning-Enhanced Planning Strategies:")
    click.echo("=" * 50)
    
    strategies = [
        ("causal_optimized", "🔗 Causal Optimized", "Uses causal reasoning to optimize task dependencies and effects"),
        ("probabilistic_risk", "🎲 Probabilistic Risk", "Uses probabilistic reasoning for risk assessment and mitigation"),
        ("temporal_sequenced", "⏰ Temporal Sequenced", "Uses temporal reasoning for optimal scheduling and sequencing"),
        ("spatial_distributed", "🗺️ Spatial Distributed", "Uses spatial reasoning for resource allocation and distribution"),
        ("collaborative_consensus", "🤝 Collaborative Consensus", "Uses collaborative reasoning for multi-stakeholder planning"),
        ("inductive_learned", "📚 Inductive Learned", "Uses inductive reasoning to learn from historical planning patterns"),
        ("hybrid_intelligent", "🚀 Hybrid Intelligent", "Uses multiple reasoning types adaptively for optimal planning")
    ]
    
    for strategy_id, name, description in strategies:
        click.echo(f"  {name}")
        click.echo(f"    ID: {strategy_id}")
        click.echo(f"    Description: {description}")
        click.echo()


@reasoning_planning_commands.command('create')
@click.option('--goal', '-g', required=True, help='Goal description for planning')
@click.option('--type', '-t', default='development', 
              type=click.Choice(['research', 'development', 'analysis', 'optimization', 'strategy']),
              help='Goal type')
@click.option('--priority', '-p', default='medium',
              type=click.Choice(['low', 'medium', 'high', 'critical']),
              help='Goal priority')
@click.option('--deadline', '-d', help='Deadline in days from now (e.g., 30)')
@click.option('--strategy', '-s', default='hybrid_intelligent',
              type=click.Choice(['causal_optimized', 'probabilistic_risk', 'temporal_sequenced', 
                               'spatial_distributed', 'collaborative_consensus', 'inductive_learned', 'hybrid_intelligent']),
              help='Reasoning strategy to use')
@click.option('--resources', '-r', multiple=True, help='Resource constraints (format: key:value)')
@click.option('--risks', multiple=True, help='Risk factors to consider')
@click.option('--stakeholders', multiple=True, help='Stakeholder preferences (format: stakeholder:preference)')
@click.pass_context
def create_reasoning_plan(ctx, goal: str, type: str, priority: str, deadline: Optional[str], 
                        strategy: str, resources: tuple, risks: tuple, stakeholders: tuple):
    """Create an intelligent plan using reasoning engines."""
    
    planner = ctx.obj['reasoning_planner']
    
    async def create_plan():
        # Parse goal parameters
        goal_type_map = {
            'research': GoalType.RESEARCH,
            'development': GoalType.DEVELOPMENT,
            'analysis': GoalType.ANALYSIS,
            'optimization': GoalType.OPTIMIZATION,
            'strategy': GoalType.STRATEGY
        }
        
        priority_map = {
            'low': GoalPriority.LOW,
            'medium': GoalPriority.MEDIUM,
            'high': GoalPriority.HIGH,
            'critical': GoalPriority.CRITICAL
        }
        
        strategy_map = {
            'causal_optimized': ReasoningPlanningStrategy.CAUSAL_OPTIMIZED,
            'probabilistic_risk': ReasoningPlanningStrategy.PROBABILISTIC_RISK,
            'temporal_sequenced': ReasoningPlanningStrategy.TEMPORAL_SEQUENCED,
            'spatial_distributed': ReasoningPlanningStrategy.SPATIAL_DISTRIBUTED,
            'collaborative_consensus': ReasoningPlanningStrategy.COLLABORATIVE_CONSENSUS,
            'inductive_learned': ReasoningPlanningStrategy.INDUCTIVE_LEARNED,
            'hybrid_intelligent': ReasoningPlanningStrategy.HYBRID_INTELLIGENT
        }
        
        # Create goal object
        goal_deadline = datetime.now() + timedelta(days=int(deadline) if deadline else 30)
        
        planning_goal = Goal(
            description=goal,
            goal_type=goal_type_map[type],
            priority=priority_map[priority],
            deadline=goal_deadline
        )
        
        # Parse context parameters
        resource_constraints = {}
        if resources:
            for resource in resources:
                if ':' in resource:
                    key, value = resource.split(':', 1)
                    resource_constraints[key.strip()] = value.strip()
        
        stakeholder_preferences = {}
        if stakeholders:
            for stakeholder in stakeholders:
                if ':' in stakeholder:
                    key, value = stakeholder.split(':', 1)
                    stakeholder_preferences[key.strip()] = value.strip()
        
        # Create planning context
        planning_context = ReasoningPlanContext(
            resource_constraints=resource_constraints,
            risk_factors=list(risks),
            stakeholder_preferences=stakeholder_preferences
        )
        
        # Generate intelligent plan
        click.echo(f"🧠 Creating intelligent plan using {strategy} strategy...")
        click.echo(f"🎯 Goal: {goal}")
        click.echo(f"📅 Deadline: {goal_deadline.strftime('%Y-%m-%d')}")
        
        plan = await planner.design_intelligent_plan(
            goal=planning_goal,
            reasoning_strategy=strategy_map[strategy],
            planning_context=planning_context
        )
        
        # Display results
        click.echo("\n📋 INTELLIGENT PLAN CREATED")
        click.echo("=" * 40)
        click.echo(f"Plan ID: {plan.id}")
        click.echo(f"Goal: {plan.goal.description}")
        click.echo(f"Strategy: {strategy}")
        click.echo(f"Created: {plan.created_at.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"Total Tasks: {len(plan.tasks)}")
        
        # Show reasoning insights
        if plan.metadata and 'reasoning_insights' in plan.metadata:
            insights = plan.metadata['reasoning_insights']
            click.echo(f"\n🧠 REASONING INSIGHTS:")
            if 'requirements' in insights:
                click.echo(f"Requirements identified: {len(insights['requirements'])}")
            if 'dependencies' in insights:
                click.echo(f"Dependencies found: {len(insights['dependencies'])}")
            if 'risk_assessment' in insights:
                click.echo(f"Risk factors assessed: {len(insights['risk_assessment'])}")
        
        # Show key tasks
        click.echo(f"\n📊 KEY TASKS:")
        for i, task in enumerate(plan.tasks[:5], 1):  # Show first 5 tasks
            click.echo(f"{i}. {task.name}")
            click.echo(f"   Description: {task.description[:80]}...")
            click.echo(f"   Duration: {task.estimated_duration}")
            click.echo(f"   Priority: {task.priority.value}")
            
            # Show reasoning enhancements
            if task.metadata:
                reasoning_types = task.metadata.get('reasoning_types_applied', [])
                if reasoning_types:
                    click.echo(f"   🧠 Reasoning Applied: {', '.join(reasoning_types)}")
            click.echo()
        
        if len(plan.tasks) > 5:
            click.echo(f"... and {len(plan.tasks) - 5} more tasks")
        
        click.echo(f"\n✅ Intelligent plan created successfully!")
        
        return plan
    
    asyncio.run(create_plan())


@reasoning_planning_commands.command('optimize')
@click.option('--plan-id', '-p', required=True, help='Plan ID to optimize')
@click.option('--strategy', '-s', required=True,
              type=click.Choice(['causal_optimized', 'probabilistic_risk', 'temporal_sequenced', 
                               'spatial_distributed', 'hybrid_intelligent']),
              help='Optimization strategy')
@click.option('--focus', '-f', multiple=True,
              type=click.Choice(['dependencies', 'risks', 'timing', 'resources', 'quality']),
              help='Optimization focus areas')
@click.pass_context
def optimize_plan(ctx, plan_id: str, strategy: str, focus: tuple):
    """Optimize an existing plan using reasoning engines."""
    
    click.echo(f"🚀 Optimizing plan {plan_id} using {strategy} strategy")
    click.echo(f"🎯 Focus areas: {', '.join(focus) if focus else 'All aspects'}")
    
    # This would integrate with actual plan storage/retrieval
    click.echo("⚠️  Plan optimization feature requires plan storage integration")
    click.echo("📝 This would apply reasoning-based optimizations to existing plans")


@reasoning_planning_commands.command('analyze')
@click.option('--goal', '-g', required=True, help='Goal to analyze for planning recommendations')
@click.option('--context', '-c', multiple=True, help='Additional context factors')
@click.pass_context
def analyze_planning_requirements(ctx, goal: str, context: tuple):
    """Analyze a goal and provide reasoning-based planning recommendations."""
    
    planner = ctx.obj['reasoning_planner']
    
    async def analyze_goal():
        # Create temporary goal for analysis
        analysis_goal = Goal(
            description=goal,
            goal_type=GoalType.ANALYSIS,  # Default for analysis
            priority=GoalPriority.MEDIUM,
            deadline=datetime.now() + timedelta(days=30)
        )
        
        click.echo(f"🔍 Analyzing goal: {goal}")
        click.echo("🧠 Generating reasoning-based recommendations...")
        
        # Get reasoning recommendations
        recommendations = await planner.get_reasoning_recommendations(analysis_goal)
        
        click.echo("\n📊 PLANNING ANALYSIS RESULTS")
        click.echo("=" * 40)
        
        recommended_strategy = recommendations['recommended_strategy']
        confidence = recommendations['reasoning_confidence']
        
        click.echo(f"🎯 Recommended Strategy: {recommended_strategy.value}")
        click.echo(f"🔬 Confidence Level: {confidence:.2f}")
        
        # Show reasoning breakdown
        reasoning_results = recommendations['recommendations']
        click.echo(f"\n🧠 REASONING BREAKDOWN:")
        
        for reasoning_type, result in reasoning_results.items():
            if result.success:
                click.echo(f"✅ {reasoning_type.value.upper()}:")
                for conclusion in result.conclusions[:2]:  # Show first 2 conclusions
                    click.echo(f"   • {conclusion.statement}")
                    click.echo(f"     Confidence: {conclusion.confidence:.2f}")
                click.echo()
        
        # Planning recommendations
        click.echo("💡 PLANNING RECOMMENDATIONS:")
        
        strategy_advice = {
            ReasoningPlanningStrategy.CAUSAL_OPTIMIZED: "Focus on understanding task dependencies and cause-effect relationships",
            ReasoningPlanningStrategy.PROBABILISTIC_RISK: "Emphasize risk assessment and uncertainty management",
            ReasoningPlanningStrategy.TEMPORAL_SEQUENCED: "Optimize timing and scheduling for efficiency",
            ReasoningPlanningStrategy.SPATIAL_DISTRIBUTED: "Consider geographic and resource distribution factors",
            ReasoningPlanningStrategy.COLLABORATIVE_CONSENSUS: "Involve multiple stakeholders in planning process",
            ReasoningPlanningStrategy.INDUCTIVE_LEARNED: "Learn from historical patterns and past experiences",
            ReasoningPlanningStrategy.HYBRID_INTELLIGENT: "Use adaptive approach with multiple reasoning types"
        }
        
        advice = strategy_advice.get(recommended_strategy, "Use comprehensive reasoning approach")
        click.echo(f"   {advice}")
        
        # Additional recommendations based on context
        if context:
            click.echo(f"\n📋 CONTEXT CONSIDERATIONS:")
            for ctx_item in context:
                click.echo(f"   • {ctx_item}")
        
        click.echo(f"\n✅ Goal analysis completed!")
        
        return recommendations
    
    asyncio.run(analyze_goal())


@reasoning_planning_commands.command('demo')
@click.option('--type', '-t', 
              type=click.Choice(['software', 'medical', 'supply-chain', 'strategic', 'learning', 'all']),
              default='all', help='Type of planning demo to run')
@click.pass_context
def run_planning_demo(ctx, type: str):
    """Run reasoning-enhanced planning demonstrations."""
    
    async def run_demo():
        if type == 'all':
            click.echo("🚀 Running all reasoning-enhanced planning demonstrations...")
            try:
                from examples.reasoning_planning_examples import run_reasoning_planning_examples
                await run_reasoning_planning_examples()
            except ImportError:
                click.echo("❌ Planning demo examples not found.")
        else:
            demo_map = {
                'software': 'example_software_development_planning',
                'medical': 'example_medical_research_planning', 
                'supply-chain': 'example_supply_chain_planning',
                'strategic': 'example_collaborative_strategic_planning',
                'learning': 'example_adaptive_learning_planning'
            }
            
            if type in demo_map:
                click.echo(f"🎯 Running {type} planning demonstration...")
                try:
                    from examples.reasoning_planning_examples import __dict__ as examples
                    demo_func = examples[demo_map[type]]
                    await demo_func()
                except (ImportError, KeyError):
                    click.echo(f"❌ {type} demo not found.")
    
    asyncio.run(run_demo())


# Integration function to add reasoning planning commands to main CLI
def add_reasoning_planning_commands(main_cli):
    """Add reasoning-enhanced planning commands to the main Feriq CLI."""
    main_cli.add_command(reasoning_planning_commands)