"""
Intelligent Planning Demo - Showcasing Reasoning-Enhanced Planning

This demo demonstrates how Feriq's planner uses comprehensive reasoning engines
to create intelligent, optimized plans for various scenarios.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from feriq.components.reasoning_plan_designer import (
    ReasoningEnhancedPlanDesigner,
    ReasoningPlanningStrategy,
    ReasoningPlanContext
)
from feriq.core.goal import Goal, GoalType, GoalPriority
from feriq.core.plan import Plan


class IntelligentPlanningDemo:
    """Comprehensive demonstration of reasoning-enhanced planning capabilities."""
    
    def __init__(self):
        self.planner = ReasoningEnhancedPlanDesigner()
        self.demo_results = {}
    
    async def run_complete_demo(self):
        """Run all planning demonstrations."""
        print("🚀 FERIQ INTELLIGENT PLANNING DEMONSTRATION")
        print("=" * 60)
        print("Showcasing how reasoning engines enhance planning capabilities\n")
        
        demos = [
            ("Software Development Planning", self.demo_software_development),
            ("Medical Research Planning", self.demo_medical_research),
            ("Supply Chain Optimization", self.demo_supply_chain),
            ("Strategic Business Planning", self.demo_strategic_planning),
            ("Adaptive Learning Planning", self.demo_adaptive_learning),
            ("Cross-Strategy Comparison", self.demo_strategy_comparison)
        ]
        
        for demo_name, demo_func in demos:
            print(f"\n🎯 {demo_name}")
            print("-" * 40)
            await demo_func()
            await asyncio.sleep(1)  # Brief pause between demos
        
        print(f"\n✅ All demonstrations completed!")
        await self.show_demo_summary()
    
    async def demo_software_development(self):
        """Demonstrate causal reasoning for software development planning."""
        
        goal = Goal(
            description="Develop a microservices-based e-commerce platform with AI recommendations",
            goal_type=GoalType.DEVELOPMENT,
            priority=GoalPriority.HIGH,
            deadline=datetime.now() + timedelta(days=90)
        )
        
        context = ReasoningPlanContext(
            resource_constraints={
                "team_size": "6 developers",
                "budget": "$500,000",
                "technology_stack": "Python/FastAPI, React, PostgreSQL, Redis"
            },
            risk_factors=[
                "Integration complexity with legacy systems",
                "AI model performance requirements",
                "Scalability requirements for Black Friday traffic"
            ],
            stakeholder_preferences={
                "CTO": "Focus on scalability and maintainability",
                "Product Manager": "Prioritize user experience features",
                "DevOps": "Emphasize deployment automation"
            }
        )
        
        plan = await self.planner.design_intelligent_plan(
            goal=goal,
            reasoning_strategy=ReasoningPlanningStrategy.CAUSAL_OPTIMIZED,
            planning_context=context
        )
        
        print(f"📋 Plan Created: {plan.id}")
        print(f"🧠 Strategy: Causal Optimized (dependency analysis)")
        print(f"📊 Total Tasks: {len(plan.tasks)}")
        
        # Show reasoning insights
        insights = plan.metadata.get('reasoning_insights', {})
        if insights:
            print(f"🔗 Dependencies Identified: {len(insights.get('dependencies', []))}")
            print(f"⚠️  Risk Factors Assessed: {len(insights.get('risk_assessment', []))}")
        
        # Show key causal relationships discovered
        causal_chains = insights.get('causal_chains', [])
        if causal_chains:
            print(f"🔍 Key Causal Relationships:")
            for chain in causal_chains[:3]:
                print(f"   • {chain}")
        
        self.demo_results['software_development'] = {
            'plan_id': plan.id,
            'tasks': len(plan.tasks),
            'strategy': 'causal_optimized',
            'insights': insights
        }
    
    async def demo_medical_research(self):
        """Demonstrate probabilistic reasoning for medical research planning."""
        
        goal = Goal(
            description="Conduct clinical trial for novel cancer immunotherapy treatment",
            goal_type=GoalType.RESEARCH,
            priority=GoalPriority.CRITICAL,
            deadline=datetime.now() + timedelta(days=730)  # 2 years
        )
        
        context = ReasoningPlanContext(
            resource_constraints={
                "patient_recruitment": "200 participants",
                "budget": "$2.5M",
                "regulatory_timeline": "FDA approval required"
            },
            risk_factors=[
                "Patient recruitment challenges",
                "Adverse reaction probability",
                "Regulatory approval uncertainty",
                "Competitive treatment emergence"
            ],
            stakeholder_preferences={
                "Lead Researcher": "Focus on statistical significance",
                "Ethics Committee": "Prioritize patient safety",
                "Sponsor": "Manage budget and timeline risks"
            }
        )
        
        plan = await self.planner.design_intelligent_plan(
            goal=goal,
            reasoning_strategy=ReasoningPlanningStrategy.PROBABILISTIC_RISK,
            planning_context=context
        )
        
        print(f"📋 Plan Created: {plan.id}")
        print(f"🧠 Strategy: Probabilistic Risk (uncertainty management)")
        print(f"📊 Total Tasks: {len(plan.tasks)}")
        
        # Show risk analysis
        insights = plan.metadata.get('reasoning_insights', {})
        risk_analysis = insights.get('risk_assessment', {})
        if risk_analysis:
            print(f"🎲 Risk Probabilities Calculated:")
            for risk, probability in list(risk_analysis.items())[:3]:
                print(f"   • {risk}: {probability}")
        
        # Show mitigation strategies
        mitigation = insights.get('mitigation_strategies', [])
        if mitigation:
            print(f"🛡️  Mitigation Strategies:")
            for strategy in mitigation[:2]:
                print(f"   • {strategy}")
        
        self.demo_results['medical_research'] = {
            'plan_id': plan.id,
            'tasks': len(plan.tasks),
            'strategy': 'probabilistic_risk',
            'insights': insights
        }
    
    async def demo_supply_chain(self):
        """Demonstrate spatial reasoning for supply chain optimization."""
        
        goal = Goal(
            description="Optimize global supply chain for sustainable electronics manufacturing",
            goal_type=GoalType.OPTIMIZATION,
            priority=GoalPriority.HIGH,
            deadline=datetime.now() + timedelta(days=180)
        )
        
        context = ReasoningPlanContext(
            resource_constraints={
                "manufacturing_locations": "5 facilities across 3 continents",
                "transportation_budget": "$1.2M annually",
                "sustainability_target": "30% carbon reduction"
            },
            risk_factors=[
                "Geopolitical instability in key regions",
                "Supply disruption from natural disasters",
                "Raw material price volatility"
            ],
            stakeholder_preferences={
                "Operations": "Minimize transportation costs",
                "Sustainability": "Reduce carbon footprint",
                "Quality": "Maintain product quality standards"
            }
        )
        
        plan = await self.planner.design_intelligent_plan(
            goal=goal,
            reasoning_strategy=ReasoningPlanningStrategy.SPATIAL_DISTRIBUTED,
            planning_context=context
        )
        
        print(f"📋 Plan Created: {plan.id}")
        print(f"🧠 Strategy: Spatial Distributed (geographic optimization)")
        print(f"📊 Total Tasks: {len(plan.tasks)}")
        
        # Show spatial analysis
        insights = plan.metadata.get('reasoning_insights', {})
        spatial_analysis = insights.get('spatial_analysis', {})
        if spatial_analysis:
            print(f"🗺️  Spatial Optimizations:")
            for optimization in list(spatial_analysis.values())[:3]:
                print(f"   • {optimization}")
        
        # Show resource allocation
        resource_allocation = insights.get('resource_allocation', {})
        if resource_allocation:
            print(f"📦 Resource Distribution:")
            for location, allocation in list(resource_allocation.items())[:3]:
                print(f"   • {location}: {allocation}")
        
        self.demo_results['supply_chain'] = {
            'plan_id': plan.id,
            'tasks': len(plan.tasks),
            'strategy': 'spatial_distributed',
            'insights': insights
        }
    
    async def demo_strategic_planning(self):
        """Demonstrate collaborative reasoning for strategic planning."""
        
        goal = Goal(
            description="Launch AI-powered fintech startup in competitive market",
            goal_type=GoalType.STRATEGY,
            priority=GoalPriority.HIGH,
            deadline=datetime.now() + timedelta(days=365)
        )
        
        context = ReasoningPlanContext(
            resource_constraints={
                "funding": "$10M Series A",
                "team_size": "25 employees",
                "regulatory_compliance": "Multiple jurisdictions"
            },
            risk_factors=[
                "Intense market competition",
                "Regulatory changes",
                "Technical scalability challenges"
            ],
            stakeholder_preferences={
                "CEO": "Focus on market penetration",
                "CTO": "Prioritize technical excellence",
                "CFO": "Optimize cost efficiency",
                "Investors": "Achieve key milestones",
                "Legal": "Ensure regulatory compliance"
            }
        )
        
        plan = await self.planner.design_intelligent_plan(
            goal=goal,
            reasoning_strategy=ReasoningPlanningStrategy.COLLABORATIVE_CONSENSUS,
            planning_context=context
        )
        
        print(f"📋 Plan Created: {plan.id}")
        print(f"🧠 Strategy: Collaborative Consensus (stakeholder alignment)")
        print(f"📊 Total Tasks: {len(plan.tasks)}")
        
        # Show consensus analysis
        insights = plan.metadata.get('reasoning_insights', {})
        consensus_analysis = insights.get('stakeholder_consensus', {})
        if consensus_analysis:
            print(f"🤝 Stakeholder Alignment:")
            for stakeholder, alignment in list(consensus_analysis.items())[:4]:
                print(f"   • {stakeholder}: {alignment}")
        
        # Show conflict resolution
        conflict_resolution = insights.get('conflict_resolution', [])
        if conflict_resolution:
            print(f"⚖️  Conflict Resolutions:")
            for resolution in conflict_resolution[:2]:
                print(f"   • {resolution}")
        
        self.demo_results['strategic_planning'] = {
            'plan_id': plan.id,
            'tasks': len(plan.tasks),
            'strategy': 'collaborative_consensus',
            'insights': insights
        }
    
    async def demo_adaptive_learning(self):
        """Demonstrate inductive reasoning for adaptive learning planning."""
        
        goal = Goal(
            description="Create personalized AI tutoring system for STEM education",
            goal_type=GoalType.DEVELOPMENT,
            priority=GoalPriority.MEDIUM,
            deadline=datetime.now() + timedelta(days=120)
        )
        
        context = ReasoningPlanContext(
            resource_constraints={
                "development_team": "4 AI researchers + 3 developers",
                "training_data": "Educational content from 50,000 students",
                "compute_resources": "GPU cluster for model training"
            },
            risk_factors=[
                "Student privacy concerns",
                "Model bias in educational content",
                "Adaptation speed requirements"
            ],
            stakeholder_preferences={
                "Educators": "Focus on pedagogical effectiveness",
                "Students": "Prioritize engagement and usability",
                "Parents": "Ensure safety and progress tracking"
            }
        )
        
        plan = await self.planner.design_intelligent_plan(
            goal=goal,
            reasoning_strategy=ReasoningPlanningStrategy.INDUCTIVE_LEARNED,
            planning_context=context
        )
        
        print(f"📋 Plan Created: {plan.id}")
        print(f"🧠 Strategy: Inductive Learned (pattern-based optimization)")
        print(f"📊 Total Tasks: {len(plan.tasks)}")
        
        # Show pattern analysis
        insights = plan.metadata.get('reasoning_insights', {})
        patterns = insights.get('learned_patterns', [])
        if patterns:
            print(f"📚 Educational Patterns Identified:")
            for pattern in patterns[:3]:
                print(f"   • {pattern}")
        
        # Show adaptive strategies
        adaptations = insights.get('adaptive_strategies', [])
        if adaptations:
            print(f"🔄 Adaptive Learning Strategies:")
            for strategy in adaptations[:2]:
                print(f"   • {strategy}")
        
        self.demo_results['adaptive_learning'] = {
            'plan_id': plan.id,
            'tasks': len(plan.tasks),
            'strategy': 'inductive_learned',
            'insights': insights
        }
    
    async def demo_strategy_comparison(self):
        """Compare different reasoning strategies for the same goal."""
        
        print("📊 Comparing reasoning strategies for identical goal...")
        
        goal = Goal(
            description="Launch sustainable urban transportation solution",
            goal_type=GoalType.STRATEGY,
            priority=GoalPriority.HIGH,
            deadline=datetime.now() + timedelta(days=200)
        )
        
        context = ReasoningPlanContext(
            resource_constraints={"budget": "$5M", "team": "15 people"},
            risk_factors=["Regulatory approval", "Market competition"],
            stakeholder_preferences={"City Council": "Environmental impact", "Citizens": "Affordability"}
        )
        
        strategies = [
            ReasoningPlanningStrategy.CAUSAL_OPTIMIZED,
            ReasoningPlanningStrategy.PROBABILISTIC_RISK,
            ReasoningPlanningStrategy.HYBRID_INTELLIGENT
        ]
        
        comparison_results = {}
        
        for strategy in strategies:
            plan = await self.planner.design_intelligent_plan(goal, strategy, context)
            comparison_results[strategy.value] = {
                'tasks': len(plan.tasks),
                'reasoning_types': len(plan.metadata.get('reasoning_insights', {}))
            }
        
        print(f"🔍 Strategy Comparison Results:")
        for strategy, results in comparison_results.items():
            print(f"   • {strategy}: {results['tasks']} tasks, {results['reasoning_types']} insights")
        
        self.demo_results['strategy_comparison'] = comparison_results
    
    async def show_demo_summary(self):
        """Show summary of all demonstration results."""
        
        print("\n📈 DEMONSTRATION SUMMARY")
        print("=" * 40)
        
        total_plans = len([r for r in self.demo_results.values() if 'plan_id' in r])
        print(f"Total Plans Created: {total_plans}")
        
        strategies_used = set()
        total_tasks = 0
        
        for result in self.demo_results.values():
            if 'strategy' in result:
                strategies_used.add(result['strategy'])
                total_tasks += result.get('tasks', 0)
        
        print(f"Reasoning Strategies Demonstrated: {len(strategies_used)}")
        print(f"Total Tasks Generated: {total_tasks}")
        
        print(f"\n🧠 Reasoning Strategies Used:")
        for strategy in sorted(strategies_used):
            strategy_name = strategy.replace('_', ' ').title()
            print(f"   ✓ {strategy_name}")
        
        print(f"\n💡 Key Demonstrations:")
        print(f"   🔗 Causal dependency analysis for software development")
        print(f"   🎲 Probabilistic risk assessment for medical research")
        print(f"   🗺️  Spatial optimization for supply chain management")
        print(f"   🤝 Collaborative consensus for strategic planning")
        print(f"   📚 Inductive learning for adaptive systems")
        print(f"   📊 Multi-strategy comparison analysis")
        
        print(f"\n✅ Feriq's reasoning-enhanced planning successfully demonstrated!")
        print(f"🚀 Ready for intelligent, optimized project planning!")


async def run_intelligent_planning_demo():
    """Run the complete intelligent planning demonstration."""
    demo = IntelligentPlanningDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(run_intelligent_planning_demo())