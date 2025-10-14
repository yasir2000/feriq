"""
CLI Commands for Feriq Reasoning System

This module adds reasoning capabilities to the Feriq CLI interface.
"""

import click
import asyncio
from typing import List, Optional
from feriq.reasoning import (
    ReasoningCoordinator, ReasoningType, ReasoningStrategy,
    ReasoningContext, Evidence, ReasoningPlan
)


@click.group(name='reason')
@click.pass_context
def reasoning_commands(ctx):
    """Reasoning system commands for advanced AI problem-solving."""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj['reasoning_coordinator'] = ReasoningCoordinator()


@reasoning_commands.command('types')
@click.pass_context
def list_reasoning_types(ctx):
    """List all available reasoning types."""
    coordinator = ctx.obj['reasoning_coordinator']
    available_types = coordinator.get_available_reasoning_types()
    
    click.echo("🧠 Available Reasoning Types:")
    click.echo("=" * 40)
    
    for reasoning_type in available_types:
        click.echo(f"  • {reasoning_type.value}")
    
    click.echo(f"\nTotal: {len(available_types)} reasoning types available")


@reasoning_commands.command('analyze')
@click.option('--problem', '-p', required=True, help='Problem to analyze')
@click.option('--evidence', '-e', multiple=True, help='Evidence to include (can be used multiple times)')
@click.option('--types', '-t', multiple=True, help='Specific reasoning types to use')
@click.option('--strategy', '-s', default='parallel', 
              type=click.Choice(['parallel', 'sequential', 'adaptive', 'hierarchical']),
              help='Reasoning strategy')
@click.pass_context
def analyze_problem(ctx, problem: str, evidence: tuple, types: tuple, strategy: str):
    """Analyze a problem using specified reasoning types."""
    coordinator = ctx.obj['reasoning_coordinator']
    
    async def run_analysis():
        # Convert evidence to Evidence objects
        evidence_objects = []
        for i, ev in enumerate(evidence):
            evidence_objects.append(Evidence(
                content=ev,
                source=f"cli_user",
                confidence=0.9
            ))
        
        # Create reasoning context
        context = ReasoningContext(
            problem=problem,
            evidence=evidence_objects,
            metadata={'source': 'cli', 'strategy': strategy}
        )
        
        # Determine reasoning types
        if types:
            reasoning_types = []
            for t in types:
                try:
                    reasoning_types.append(ReasoningType(t))
                except ValueError:
                    click.echo(f"Warning: Unknown reasoning type '{t}', skipping...")
        else:
            # Auto-suggest reasoning types
            reasoning_types = await coordinator.analyze_problem(problem)
            click.echo(f"🤖 Auto-suggested reasoning types: {[rt.value for rt in reasoning_types]}")
        
        if not reasoning_types:
            click.echo("❌ No valid reasoning types specified or suggested")
            return
        
        # Convert strategy string to enum
        strategy_map = {
            'parallel': ReasoningStrategy.PARALLEL,
            'sequential': ReasoningStrategy.SEQUENTIAL,
            'adaptive': ReasoningStrategy.ADAPTIVE,
            'hierarchical': ReasoningStrategy.HIERARCHICAL
        }
        
        # Perform reasoning
        click.echo(f"🧠 Analyzing problem using {strategy} strategy...")
        results = await coordinator.reason(
            context, 
            reasoning_types=reasoning_types,
            strategy=strategy_map[strategy]
        )
        
        # Display results
        click.echo("\n📊 REASONING RESULTS:")
        click.echo("=" * 50)
        
        for reasoning_type, result in results.items():
            click.echo(f"\n🔍 {reasoning_type.value.upper()} REASONING:")
            if result.success:
                click.echo(f"   ✅ Success (Confidence: {result.confidence:.2f})")
                for i, conclusion in enumerate(result.conclusions, 1):
                    click.echo(f"   {i}. {conclusion.statement}")
                    click.echo(f"      └─ Confidence: {conclusion.confidence:.2f}")
            else:
                click.echo(f"   ❌ Failed: {result.error_message}")
        
        # Summary
        successful_types = [rt.value for rt, result in results.items() if result.success]
        click.echo(f"\n📈 Summary: {len(successful_types)}/{len(results)} reasoning types successful")
        if successful_types:
            click.echo(f"✅ Successful: {', '.join(successful_types)}")
    
    asyncio.run(run_analysis())


@reasoning_commands.command('plan')
@click.option('--problem', '-p', required=True, help='Problem to create reasoning plan for')
@click.option('--strategy', '-s', default='adaptive', 
              type=click.Choice(['sequential', 'parallel', 'hierarchical', 'adaptive', 'pipeline']),
              help='Reasoning strategy for the plan')
@click.option('--save', '-save', help='Save plan to file')
@click.pass_context
def create_reasoning_plan(ctx, problem: str, strategy: str, save: Optional[str]):
    """Create a reasoning plan for complex problem-solving."""
    coordinator = ctx.obj['reasoning_coordinator']
    
    async def create_plan():
        # Analyze problem to suggest reasoning types
        suggested_types = await coordinator.analyze_problem(problem)
        
        click.echo(f"🎯 Creating reasoning plan for: {problem}")
        click.echo(f"📋 Strategy: {strategy}")
        click.echo(f"🧠 Suggested reasoning types: {[rt.value for rt in suggested_types]}")
        
        # Create plan based on strategy
        strategy_map = {
            'sequential': ReasoningStrategy.SEQUENTIAL,
            'parallel': ReasoningStrategy.PARALLEL,
            'hierarchical': ReasoningStrategy.HIERARCHICAL,
            'adaptive': ReasoningStrategy.ADAPTIVE,
            'pipeline': ReasoningStrategy.PIPELINE
        }
        
        if strategy == 'hierarchical':
            # Create hierarchical dependencies
            dependencies = {}
            if ReasoningType.INDUCTIVE in suggested_types and ReasoningType.PROBABILISTIC in suggested_types:
                dependencies[ReasoningType.PROBABILISTIC] = [ReasoningType.INDUCTIVE]
            if ReasoningType.PROBABILISTIC in suggested_types and ReasoningType.CAUSAL in suggested_types:
                dependencies[ReasoningType.CAUSAL] = [ReasoningType.PROBABILISTIC]
        else:
            dependencies = {}
        
        plan = coordinator.create_plan(
            strategy=strategy_map[strategy],
            reasoning_types=suggested_types,
            dependencies=dependencies
        )
        
        # Display plan
        click.echo("\n📋 REASONING PLAN:")
        click.echo("=" * 40)
        click.echo(f"Strategy: {plan.strategy.value}")
        click.echo(f"Reasoning Types: {[rt.value for rt in plan.reasoning_types]}")
        
        if plan.dependencies:
            click.echo("Dependencies:")
            for rt, deps in plan.dependencies.items():
                click.echo(f"  {rt.value} depends on: {[d.value for d in deps]}")
        
        if save:
            # Save plan to file (simplified)
            plan_data = {
                'strategy': plan.strategy.value,
                'reasoning_types': [rt.value for rt in plan.reasoning_types],
                'dependencies': {rt.value: [d.value for d in deps] for rt, deps in plan.dependencies.items()},
                'problem': problem
            }
            
            import json
            with open(save, 'w') as f:
                json.dump(plan_data, f, indent=2)
            
            click.echo(f"\n💾 Plan saved to: {save}")
    
    asyncio.run(create_plan())


@reasoning_commands.command('demo')
@click.option('--type', '-t', 
              type=click.Choice(['medical', 'business', 'research', 'spatial', 'all']),
              default='all', help='Type of demo to run')
@click.pass_context
def run_reasoning_demo(ctx, type: str):
    """Run reasoning system demonstrations."""
    
    async def run_demo():
        if type == 'all':
            click.echo("🚀 Running all reasoning demonstrations...")
            # Import and run the examples
            try:
                from examples.reasoning_integration import run_all_examples
                await run_all_examples()
            except ImportError:
                click.echo("❌ Demo examples not found. Please ensure examples/reasoning_integration.py exists.")
        
        elif type == 'medical':
            click.echo("🏥 Running medical diagnostic reasoning demo...")
            from examples.reasoning_integration import example_diagnostic_reasoning
            await example_diagnostic_reasoning()
        
        elif type == 'business':
            click.echo("💼 Running business strategy reasoning demo...")
            from examples.reasoning_integration import example_business_strategy_reasoning
            await example_business_strategy_reasoning()
        
        elif type == 'research':
            click.echo("🔬 Running research reasoning demo...")
            from examples.reasoning_integration import example_research_reasoning_task
            await example_research_reasoning_task()
        
        elif type == 'spatial':
            click.echo("🗺️ Running spatial reasoning demo...")
            from examples.reasoning_integration import example_spatial_reasoning
            await example_spatial_reasoning()
    
    asyncio.run(run_demo())


@reasoning_commands.command('collaborate')
@click.option('--problem', '-p', required=True, help='Problem for collaborative reasoning')
@click.option('--agents', '-a', multiple=True, help='Agent names for collaboration')
@click.option('--method', '-m', default='majority_vote',
              type=click.Choice(['majority_vote', 'weighted_average', 'delphi', 'argumentation']),
              help='Consensus method')
@click.pass_context
def collaborative_reasoning(ctx, problem: str, agents: tuple, method: str):
    """Run collaborative reasoning with multiple agents."""
    coordinator = ctx.obj['reasoning_coordinator']
    
    async def run_collaboration():
        if not agents:
            agents_list = ['Agent_Alpha', 'Agent_Beta', 'Agent_Gamma']
            click.echo(f"🤖 Using default agents: {', '.join(agents_list)}")
        else:
            agents_list = list(agents)
        
        click.echo(f"🤝 Collaborative reasoning: {problem}")
        click.echo(f"👥 Participants: {', '.join(agents_list)}")
        click.echo(f"📊 Method: {method}")
        
        # Create context for collaborative reasoning
        context = ReasoningContext(
            problem=problem,
            evidence=[Evidence(content=problem, source="collaborative_session")],
            metadata={
                'collaboration_type': 'consensus',
                'consensus_method': method,
                'participants': agents_list
            }
        )
        
        # Use collaborative reasoning
        results = await coordinator.reason(
            context,
            reasoning_types=[ReasoningType.COLLABORATIVE],
            strategy=ReasoningStrategy.PARALLEL
        )
        
        # Display results
        collab_result = results.get(ReasoningType.COLLABORATIVE)
        if collab_result and collab_result.success:
            click.echo("\n🎯 COLLABORATIVE REASONING RESULTS:")
            click.echo("=" * 50)
            
            for conclusion in collab_result.conclusions:
                click.echo(f"✅ {conclusion.statement}")
                click.echo(f"   Confidence: {conclusion.confidence:.2f}")
                
                # Display collaborative metadata if available
                if 'collaborative_result' in conclusion.metadata:
                    collab_data = conclusion.metadata['collaborative_result']
                    if 'consensus_strength' in collab_data:
                        click.echo(f"   Consensus Strength: {collab_data['consensus_strength']:.2f}")
        else:
            click.echo("❌ Collaborative reasoning failed")
    
    asyncio.run(run_collaboration())


# Integration function to add reasoning commands to main CLI
def add_reasoning_commands(main_cli):
    """Add reasoning commands to the main Feriq CLI."""
    main_cli.add_command(reasoning_commands)