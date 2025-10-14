"""
Feriq Framework - Simple Example

This is a basic example showing how to use the Feriq framework
to create a simple multi-agent workflow.
"""

from datetime import timedelta
from feriq import (
    FeriqFramework,
    FeriqAgent,
    Goal,
    GoalType,
    Role,
    RoleCapability
)


def main():
    """Simple example of using Feriq framework."""
    print("🚀 Feriq Framework - Simple Example")
    
    # Initialize framework
    framework = FeriqFramework()
    
    # Create a simple goal
    goal = Goal(
        name="Write a Research Report",
        description="Research and write a comprehensive report on AI trends",
        goal_type=GoalType.RESEARCH,
        required_capabilities=["research", "writing", "analysis"],
        success_criteria=["Complete literature review", "Write 10-page report"],
        estimated_duration=timedelta(days=3)
    )
    
    # Create roles
    researcher_role = Role(
        name="Researcher",
        description="Conducts research and gathers information",
        capabilities=[
            RoleCapability("research", 0.9),
            RoleCapability("analysis", 0.8),
            RoleCapability("data_gathering", 0.85)
        ]
    )
    
    writer_role = Role(
        name="Writer",
        description="Creates written content and documentation",
        capabilities=[
            RoleCapability("writing", 0.9),
            RoleCapability("editing", 0.8),
            RoleCapability("documentation", 0.85)
        ]
    )
    
    # Create agents
    researcher_agent = FeriqAgent(
        name="ResearcherBot",
        role=researcher_role,
        capabilities=["research", "analysis", "data_gathering"]
    )
    
    writer_agent = FeriqAgent(
        name="WriterBot", 
        role=writer_role,
        capabilities=["writing", "editing", "documentation"]
    )
    
    # Add to framework
    framework.add_goal(goal)
    framework.add_role(researcher_role)
    framework.add_role(writer_role)
    framework.add_agent(researcher_agent)
    framework.add_agent(writer_agent)
    
    # Execute goal
    print(f"✅ Created goal: {goal.name}")
    print(f"✅ Created {len(framework.agents)} agents")
    print(f"✅ Framework ready with {len(framework.roles)} roles")
    
    # Show framework status
    status = framework.get_framework_status()
    print(f"\n📊 Framework Status:")
    print(f"   Goals: {status['goals']}")
    print(f"   Agents: {status['agents']}")
    print(f"   Roles: {status['roles']}")
    
    print("\n🎉 Simple example completed!")


if __name__ == "__main__":
    main()