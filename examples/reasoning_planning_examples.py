"""
Reasoning-Enhanced Planning Examples

This module demonstrates practical examples of how the Feriq planner can use
comprehensive reasoning engines for intelligent plan generation.
"""

import asyncio
from datetime import datetime, timedelta
from feriq.components.reasoning_plan_designer import (
    ReasoningEnhancedPlanDesigner, 
    ReasoningPlanningStrategy, 
    ReasoningPlanContext
)
from feriq.core.goal import Goal, GoalType, GoalPriority
from feriq.reasoning import ReasoningType


async def example_software_development_planning():
    """Example: Software development project planning using causal reasoning."""
    
    print("🛠️ SOFTWARE DEVELOPMENT PLANNING WITH REASONING")
    print("=" * 60)
    
    # Create reasoning-enhanced planner
    planner = ReasoningEnhancedPlanDesigner()
    
    # Define development goal
    dev_goal = Goal(
        description="Develop a customer relationship management (CRM) system",
        goal_type=GoalType.DEVELOPMENT,
        priority=GoalPriority.HIGH,
        deadline=datetime.now() + timedelta(days=90),
        success_criteria=[
            "User authentication system implemented",
            "Customer data management functionality",
            "Sales pipeline tracking",
            "Reporting dashboard",
            "Mobile responsive design"
        ]
    )
    
    # Create planning context with constraints and requirements
    planning_context = ReasoningPlanContext(
        resource_constraints={
            "developers": 4,
            "designers": 2,
            "testers": 2,
            "budget": "$150,000",
            "timeline": "90 days"
        },
        risk_factors=[
            "New technology stack",
            "Tight deadline",
            "Complex user requirements",
            "Third-party API dependencies"
        ],
        stakeholder_preferences={
            "client": "Prioritize user experience",
            "development_team": "Use modern tech stack",
            "project_manager": "Minimize risk and stay on schedule"
        },
        historical_plans=[
            {"project": "Previous CRM", "duration": "120 days", "success": True},
            {"project": "E-commerce system", "duration": "75 days", "success": True},
            {"project": "Mobile app", "duration": "60 days", "success": False}
        ]
    )
    
    # Generate intelligent plan using causal reasoning
    causal_plan = await planner.design_intelligent_plan(
        goal=dev_goal,
        reasoning_strategy=ReasoningPlanningStrategy.CAUSAL_OPTIMIZED,
        planning_context=planning_context
    )
    
    print(f"📋 Plan Created: {causal_plan.id}")
    print(f"🎯 Goal: {dev_goal.description}")
    print(f"🧠 Reasoning Strategy: Causal Optimization")
    print(f"📅 Timeline: {causal_plan.created_at} → {causal_plan.deadline}")
    print(f"📊 Total Tasks: {len(causal_plan.tasks)}")
    
    print("\n🔗 CAUSAL REASONING INSIGHTS:")
    for i, task in enumerate(causal_plan.tasks[:5], 1):  # Show first 5 tasks
        print(f"{i}. {task.name}")
        print(f"   Description: {task.description}")
        print(f"   Dependencies: {task.dependencies}")
        if task.metadata and 'causal_insights' in task.metadata:
            print(f"   Causal Insights: {task.metadata['causal_insights'][:2]}")  # First 2 insights
        print(f"   Duration: {task.estimated_duration}")
        print()
    
    return causal_plan


async def example_medical_research_planning():
    """Example: Medical research project using probabilistic risk assessment."""
    
    print("🏥 MEDICAL RESEARCH PLANNING WITH PROBABILISTIC REASONING")
    print("=" * 65)
    
    planner = ReasoningEnhancedPlanDesigner()
    
    # Define research goal
    research_goal = Goal(
        description="Conduct clinical trial for new diabetes treatment drug",
        goal_type=GoalType.RESEARCH,
        priority=GoalPriority.CRITICAL,
        deadline=datetime.now() + timedelta(days=365),
        success_criteria=[
            "IRB approval obtained",
            "Patient recruitment completed (200 participants)",
            "Phase II trial completed",
            "Safety and efficacy data collected",
            "Regulatory submission prepared"
        ]
    )
    
    # Research context with high uncertainty and risk
    research_context = ReasoningPlanContext(
        resource_constraints={
            "research_staff": 8,
            "clinical_sites": 3,
            "budget": "$2,500,000",
            "timeline": "12 months"
        },
        risk_factors=[
            "FDA regulatory requirements",
            "Patient recruitment challenges",
            "Adverse events possibility",
            "Competition from other trials",
            "COVID-19 impact on trials"
        ],
        stakeholder_preferences={
            "pharmaceutical_company": "Fast track to market",
            "FDA": "Rigorous safety protocols",
            "research_institution": "Scientific rigor",
            "patients": "Safe and effective treatment"
        },
        success_metrics=[
            "Patient safety score > 95%",
            "Primary endpoint achievement",
            "Regulatory timeline adherence",
            "Budget variance < 10%"
        ]
    )
    
    # Generate plan using probabilistic risk assessment
    risk_plan = await planner.design_intelligent_plan(
        goal=research_goal,
        reasoning_strategy=ReasoningPlanningStrategy.PROBABILISTIC_RISK,
        planning_context=research_context
    )
    
    print(f"📋 Research Plan: {risk_plan.id}")
    print(f"🎯 Study: {research_goal.description}")
    print(f"🧠 Strategy: Probabilistic Risk Assessment")
    print(f"⏱️ Duration: 12 months")
    print(f"📊 Tasks: {len(risk_plan.tasks)}")
    
    print("\n🎲 PROBABILISTIC RISK ANALYSIS:")
    high_risk_tasks = []
    for task in risk_plan.tasks:
        if task.metadata and 'risk_score' in task.metadata:
            risk_score = task.metadata['risk_score']
            print(f"📌 {task.name}")
            print(f"   Risk Score: {risk_score:.2f}")
            if risk_score > 0.7:
                high_risk_tasks.append(task)
                print(f"   ⚠️  HIGH RISK - Contingency: {task.metadata.get('contingency_plan', 'None')}")
            print(f"   Duration: {task.estimated_duration}")
        print()
    
    print(f"🚨 High Risk Tasks: {len(high_risk_tasks)}/{len(risk_plan.tasks)}")
    
    return risk_plan


async def example_supply_chain_planning():
    """Example: Supply chain optimization using spatial reasoning."""
    
    print("🚚 SUPPLY CHAIN OPTIMIZATION WITH SPATIAL REASONING")
    print("=" * 60)
    
    planner = ReasoningEnhancedPlanDesigner()
    
    # Supply chain optimization goal
    supply_goal = Goal(
        description="Optimize global supply chain network for electronics manufacturing",
        goal_type=GoalType.OPTIMIZATION,
        priority=GoalPriority.HIGH,
        deadline=datetime.now() + timedelta(days=180),
        success_criteria=[
            "Reduce shipping costs by 15%",
            "Improve delivery times by 20%",
            "Establish 3 new distribution centers",
            "Optimize inventory levels",
            "Ensure supply chain resilience"
        ]
    )
    
    # Spatial context with geographic considerations
    spatial_context = ReasoningPlanContext(
        resource_constraints={
            "distribution_centers": 5,
            "transportation_budget": "$5,000,000",
            "warehouse_capacity": "500,000 sq ft",
            "staff": 200
        },
        environmental_factors={
            "manufacturing_locations": ["China", "Vietnam", "Mexico"],
            "key_markets": ["North America", "Europe", "Asia-Pacific"],
            "shipping_routes": ["Pacific", "Atlantic", "Land-based"],
            "regulatory_zones": ["NAFTA", "EU", "ASEAN"]
        },
        risk_factors=[
            "Geopolitical tensions",
            "Natural disasters",
            "Port congestion",
            "Currency fluctuations"
        ]
    )
    
    # Generate spatially optimized plan
    spatial_plan = await planner.design_intelligent_plan(
        goal=supply_goal,
        reasoning_strategy=ReasoningPlanningStrategy.SPATIAL_DISTRIBUTED,
        planning_context=spatial_context
    )
    
    print(f"📋 Supply Chain Plan: {spatial_plan.id}")
    print(f"🎯 Objective: {supply_goal.description}")
    print(f"🧠 Strategy: Spatial Distribution Optimization")
    print(f"🌍 Scope: Global operations")
    print(f"📊 Optimization Tasks: {len(spatial_plan.tasks)}")
    
    print("\n🗺️ SPATIAL OPTIMIZATION INSIGHTS:")
    for task in spatial_plan.tasks[:4]:  # Show first 4 tasks
        print(f"📍 {task.name}")
        if task.metadata and 'resource_allocation' in task.metadata:
            print(f"   Resource Allocation: {task.metadata['resource_allocation']}")
        if task.metadata and 'spatial_insights' in task.metadata:
            insights = task.metadata['spatial_insights'][:2]  # First 2 insights
            for insight in insights:
                print(f"   🎯 {insight}")
        print()
    
    return spatial_plan


async def example_collaborative_strategic_planning():
    """Example: Multi-stakeholder strategic planning using collaborative reasoning."""
    
    print("🤝 COLLABORATIVE STRATEGIC PLANNING")
    print("=" * 45)
    
    planner = ReasoningEnhancedPlanDesigner()
    
    # Strategic planning goal
    strategy_goal = Goal(
        description="Develop 5-year digital transformation strategy",
        goal_type=GoalType.STRATEGY,
        priority=GoalPriority.CRITICAL,
        deadline=datetime.now() + timedelta(days=120),
        success_criteria=[
            "Digital roadmap defined",
            "Technology architecture planned",
            "Change management strategy",
            "Budget and resource allocation",
            "Stakeholder buy-in achieved"
        ]
    )
    
    # Multi-stakeholder context
    collaborative_context = ReasoningPlanContext(
        stakeholder_preferences={
            "CEO": "Revenue growth and competitive advantage",
            "CTO": "Technical excellence and innovation",
            "CFO": "Cost optimization and ROI",
            "CHRO": "Employee experience and skills development",
            "customers": "Better service and user experience",
            "shareholders": "Increased company value"
        },
        resource_constraints={
            "budget": "$10,000,000",
            "timeline": "5 years",
            "key_personnel": 50,
            "external_consultants": 20
        },
        success_metrics=[
            "Revenue increase > 25%",
            "Customer satisfaction > 90%",
            "Employee engagement > 85%",
            "Digital maturity score > 8/10"
        ]
    )
    
    # Generate collaborative plan
    collab_plan = await planner.design_intelligent_plan(
        goal=strategy_goal,
        reasoning_strategy=ReasoningPlanningStrategy.COLLABORATIVE_CONSENSUS,
        planning_context=collaborative_context
    )
    
    print(f"📋 Strategic Plan: {collab_plan.id}")
    print(f"🎯 Vision: {strategy_goal.description}")
    print(f"🧠 Approach: Collaborative Consensus Building")
    print(f"👥 Stakeholders: {len(collaborative_context.stakeholder_preferences)}")
    print(f"📊 Strategic Initiatives: {len(collab_plan.tasks)}")
    
    print("\n🤝 COLLABORATIVE CONSENSUS INSIGHTS:")
    consensus_tasks = []
    for task in collab_plan.tasks:
        if task.metadata and 'stakeholder_consensus' in task.metadata:
            consensus_tasks.append(task)
            print(f"✅ {task.name}")
            print(f"   Stakeholder Alignment: {task.metadata['stakeholder_consensus']}")
            if 'collaborative_insights' in task.metadata:
                insights = task.metadata['collaborative_insights'][:1]  # First insight
                for insight in insights:
                    print(f"   💡 {insight}")
        print()
    
    print(f"🎯 Consensus Achieved: {len(consensus_tasks)}/{len(collab_plan.tasks)} tasks")
    
    return collab_plan


async def example_adaptive_learning_planning():
    """Example: Project planning that learns from historical patterns."""
    
    print("🧠 ADAPTIVE LEARNING PLANNING")
    print("=" * 35)
    
    planner = ReasoningEnhancedPlanDesigner()
    
    # Learning-based planning goal
    learning_goal = Goal(
        description="Launch AI-powered customer service chatbot",
        goal_type=GoalType.DEVELOPMENT,
        priority=GoalPriority.MEDIUM,
        deadline=datetime.now() + timedelta(days=60),
        success_criteria=[
            "NLP model trained and deployed",
            "Integration with support system",
            "User interface developed",
            "Testing and validation completed",
            "Staff training provided"
        ]
    )
    
    # Context with rich historical data
    learning_context = ReasoningPlanContext(
        historical_plans=[
            {
                "project": "Previous chatbot v1.0",
                "duration": "45 days",
                "success": True,
                "lessons": "Underestimated NLP training time"
            },
            {
                "project": "Mobile app with AI",
                "duration": "80 days", 
                "success": True,
                "lessons": "Integration testing critical"
            },
            {
                "project": "Voice assistant",
                "duration": "120 days",
                "success": False,
                "lessons": "Insufficient user testing"
            },
            {
                "project": "Recommendation engine",
                "duration": "30 days",
                "success": True,
                "lessons": "Model validation important"
            }
        ],
        resource_constraints={
            "AI_engineers": 3,
            "frontend_developers": 2,
            "QA_engineers": 2,
            "budget": "$80,000"
        }
    )
    
    # Generate pattern-learned plan
    learned_plan = await planner.design_intelligent_plan(
        goal=learning_goal,
        reasoning_strategy=ReasoningPlanningStrategy.INDUCTIVE_LEARNED,
        planning_context=learning_context
    )
    
    print(f"📋 AI Project Plan: {learned_plan.id}")
    print(f"🎯 Project: {learning_goal.description}")
    print(f"🧠 Strategy: Inductive Pattern Learning")
    print(f"📚 Learning from: {len(learning_context.historical_plans)} past projects")
    print(f"📊 Optimized Tasks: {len(learned_plan.tasks)}")
    
    print("\n🔍 PATTERN LEARNING INSIGHTS:")
    for task in learned_plan.tasks[:3]:  # Show first 3 tasks
        print(f"📖 {task.name}")
        if task.metadata and 'learned_optimizations' in task.metadata:
            optimizations = task.metadata['learned_optimizations'][:2]
            for opt in optimizations:
                print(f"   💡 Learned: {opt}")
        print(f"   Estimated Duration: {task.estimated_duration}")
        print()
    
    return learned_plan


async def run_reasoning_planning_examples():
    """Run all reasoning-enhanced planning examples."""
    
    print("🚀 FERIQ REASONING-ENHANCED PLANNING DEMONSTRATIONS")
    print("=" * 70)
    print("Showcasing intelligent planning using comprehensive reasoning engines\n")
    
    # Run all planning examples
    examples = [
        ("Software Development", example_software_development_planning),
        ("Medical Research", example_medical_research_planning),
        ("Supply Chain", example_supply_chain_planning),
        ("Strategic Planning", example_collaborative_strategic_planning),
        ("Adaptive Learning", example_adaptive_learning_planning)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name.upper()} EXAMPLE {'='*20}")
            result = await example_func()
            results[name] = result
            print(f"✅ {name} planning completed successfully!")
        except Exception as e:
            print(f"❌ {name} planning failed: {e}")
            results[name] = None
        
        print("\n" + "="*70)
    
    # Summary
    print("\n🎯 REASONING-ENHANCED PLANNING SUMMARY")
    print("=" * 50)
    
    successful_plans = [name for name, result in results.items() if result is not None]
    
    print(f"📊 Total Examples: {len(examples)}")
    print(f"✅ Successful Plans: {len(successful_plans)}")
    print(f"🧠 Reasoning Types Demonstrated:")
    print("   • Causal Reasoning - Task dependency optimization")
    print("   • Probabilistic Reasoning - Risk assessment and mitigation")
    print("   • Spatial Reasoning - Geographic resource distribution")
    print("   • Collaborative Reasoning - Multi-stakeholder consensus")
    print("   • Inductive Reasoning - Pattern learning from history")
    
    print(f"\n🚀 All planning examples showcase how Feriq planners can leverage")
    print(f"   sophisticated reasoning engines for intelligent, optimized planning!")
    
    return results


if __name__ == "__main__":
    # Run all reasoning-enhanced planning examples
    asyncio.run(run_reasoning_planning_examples())