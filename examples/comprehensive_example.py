"""
Feriq Framework - Comprehensive Example

This example demonstrates the full capabilities of the Feriq collaborative AI agents framework,
including dynamic role creation, task planning, agent coordination, and intelligent reasoning.

Scenario: Research and Development Project
- Goal: Develop a new AI-powered recommendation system
- Tasks: Research, design, implementation, testing, documentation
- Agents: Multiple agents with different roles and capabilities
- Coordination: Complex multi-agent collaboration with various patterns
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import Feriq framework components
from feriq import (
    FeriqFramework,
    FeriqAgent,
    Goal,
    GoalType,
    FeriqTask,
    TaskPriority,
    TaskComplexity,
    Plan,
    Role,
    RoleCapability,
    DynamicRoleDesigner,
    TaskDesigner,
    TaskAllocator,
    PlanDesigner,
    PlanObserver,
    WorkflowOrchestrator,
    Choreographer,
    Reasoner
)
from feriq.utils.logger import FeriqLogger
from feriq.utils.config import Config


class FeriqExample:
    """Comprehensive example demonstrating Feriq framework capabilities."""
    
    def __init__(self):
        """Initialize the example with Feriq framework."""
        # Create configuration
        self.config = Config()
        self.logger = FeriqLogger("FeriqExample", self.config)
        
        # Initialize framework
        self.framework = FeriqFramework(self.config)
        
        # Initialize components
        self.role_designer = DynamicRoleDesigner(self.config)
        self.task_designer = TaskDesigner(self.config)
        self.task_allocator = TaskAllocator(self.config)
        self.plan_designer = PlanDesigner(self.config)
        self.plan_observer = PlanObserver(self.config)
        self.orchestrator = WorkflowOrchestrator(self.config)
        self.choreographer = Choreographer(self.config)
        self.reasoner = Reasoner(self.config)
        
        # Connect components
        self.orchestrator.set_plan_observer(self.plan_observer)
        
        self.logger.info("Feriq Example initialized successfully")
    
    async def run_complete_example(self):
        """Run the complete example demonstrating all framework features."""
        print("🚀 Starting Feriq Framework Comprehensive Example")
        print("=" * 60)
        
        try:
            # Step 1: Create the main goal
            goal = await self.create_research_goal()
            print(f"✅ Created goal: {goal.name}")
            
            # Step 2: Design roles dynamically
            roles = await self.design_dynamic_roles(goal)
            print(f"✅ Designed {len(roles)} dynamic roles")
            
            # Step 3: Create and configure agents
            agents = await self.create_agents(roles)
            print(f"✅ Created {len(agents)} agents")
            
            # Step 4: Design tasks from goal
            tasks = await self.design_tasks(goal)
            print(f"✅ Designed {len(tasks)} tasks")
            
            # Step 5: Create execution plan
            plan = await self.create_execution_plan(goal, tasks, roles)
            print(f"✅ Created execution plan with {len(plan.tasks)} tasks")
            
            # Step 6: Set up agent coordination
            await self.setup_agent_coordination(agents)
            print("✅ Set up agent coordination patterns")
            
            # Step 7: Start monitoring and observation
            monitor_id = await self.start_monitoring(plan)
            print(f"✅ Started plan monitoring (ID: {monitor_id[:8]}...)")
            
            # Step 8: Execute the workflow
            workflow_id = await self.execute_workflow(plan, agents)
            print(f"✅ Started workflow execution (ID: {workflow_id[:8]}...)")
            
            # Step 9: Demonstrate reasoning capabilities
            await self.demonstrate_reasoning(goal, plan, agents)
            print("✅ Demonstrated reasoning capabilities")
            
            # Step 10: Monitor execution and show results
            await self.monitor_and_report(workflow_id, monitor_id)
            print("✅ Completed monitoring and reporting")
            
            # Step 11: Show final statistics
            await self.show_final_statistics()
            print("✅ Generated final statistics")
            
        except Exception as e:
            self.logger.error(f"Example execution failed: {e}")
            print(f"❌ Example failed: {e}")
        
        print("=" * 60)
        print("🎉 Feriq Framework Example Completed!")
    
    async def create_research_goal(self) -> Goal:
        """Create the main research and development goal."""
        goal = Goal(
            name="AI Recommendation System Development",
            description="Develop a comprehensive AI-powered recommendation system with research, design, implementation, and testing phases",
            goal_type=GoalType.DEVELOPMENT,
            priority=5,
            required_capabilities=[
                "machine_learning",
                "software_development",
                "data_analysis",
                "system_design",
                "testing",
                "documentation"
            ],
            success_criteria=[
                "Complete market research and competitive analysis",
                "Design system architecture and algorithms",
                "Implement core recommendation engine",
                "Conduct thorough testing and validation",
                "Deliver comprehensive documentation",
                "Achieve 95% accuracy in recommendations"
            ],
            estimated_duration=timedelta(days=30),
            dependencies=set()
        )
        
        # Add goal to framework
        self.framework.add_goal(goal)
        
        self.logger.info("Research goal created", goal_id=goal.goal_id)
        return goal
    
    async def design_dynamic_roles(self, goal: Goal) -> List[Role]:
        """Design roles dynamically based on the goal requirements."""
        roles = []
        
        # Analyze goal requirements
        analysis = self.role_designer.analyze_requirements(
            goal.required_capabilities,
            goal.estimated_duration,
            goal.description
        )
        
        # Design specific roles for this project
        role_specs = [
            {
                "name": "Research Specialist",
                "capabilities": ["research", "data_analysis", "market_analysis"],
                "description": "Conducts market research and competitive analysis",
                "complexity_handling": 0.8
            },
            {
                "name": "System Architect",
                "capabilities": ["system_design", "architecture", "scalability"],
                "description": "Designs system architecture and technical specifications",
                "complexity_handling": 0.9
            },
            {
                "name": "ML Engineer",
                "capabilities": ["machine_learning", "algorithm_design", "data_science"],
                "description": "Develops recommendation algorithms and ML models",
                "complexity_handling": 0.95
            },
            {
                "name": "Software Developer",
                "capabilities": ["software_development", "programming", "integration"],
                "description": "Implements the recommendation system",
                "complexity_handling": 0.85
            },
            {
                "name": "QA Engineer",
                "capabilities": ["testing", "quality_assurance", "validation"],
                "description": "Tests and validates the system",
                "complexity_handling": 0.7
            },
            {
                "name": "Technical Writer",
                "capabilities": ["documentation", "technical_writing", "communication"],
                "description": "Creates comprehensive documentation",
                "complexity_handling": 0.6
            }
        ]
        
        for spec in role_specs:
            role = self.role_designer.design_role(
                required_capabilities=spec["capabilities"],
                estimated_duration=timedelta(days=5),
                context={"goal_description": goal.description}
            )
            
            # Customize the role
            role.name = spec["name"]
            role.description = spec["description"]
            
            roles.append(role)
            self.framework.add_role(role)
        
        self.logger.info(f"Designed {len(roles)} dynamic roles")
        return roles
    
    async def create_agents(self, roles: List[Role]) -> List[FeriqAgent]:
        """Create agents with the designed roles."""
        agents = []
        
        for i, role in enumerate(roles):
            agent = FeriqAgent(
                name=f"{role.name.replace(' ', '')}Agent{i+1}",
                role=role,
                capabilities=role.capabilities[:],
                max_concurrent_tasks=2,
                learning_rate=0.1
            )
            
            # Add some experience and performance metrics
            agent.performance_metrics.update({
                "success_rate": 0.85 + (i * 0.02),  # Varying success rates
                "experience_level": 0.7 + (i * 0.05),
                "tasks_completed": 10 + (i * 5),
                "average_completion_time": 120 - (i * 10)  # minutes
            })
            
            agents.append(agent)
            self.framework.add_agent(agent)
            
            # Register agent with orchestrator and choreographer
            self.orchestrator.register_agent(agent)
            self.choreographer.register_agent(agent)
        
        self.logger.info(f"Created {len(agents)} agents")
        return agents
    
    async def design_tasks(self, goal: Goal) -> List[FeriqTask]:
        """Design tasks from the goal using the task designer."""
        # Use task designer to break down the goal
        tasks = self.task_designer.decompose_goal_into_tasks(
            goal,
            max_tasks=12,
            strategy="hybrid"
        )
        
        # Enhance tasks with specific details
        task_enhancements = [
            {
                "pattern": "research",
                "name": "Market Research and Analysis",
                "description": "Conduct comprehensive market research and competitive analysis for recommendation systems",
                "complexity": TaskComplexity(0.6),
                "priority": TaskPriority.HIGH
            },
            {
                "pattern": "design",
                "name": "System Architecture Design",
                "description": "Design the overall system architecture and technical specifications",
                "complexity": TaskComplexity(0.8),
                "priority": TaskPriority.HIGH
            },
            {
                "pattern": "algorithm",
                "name": "Recommendation Algorithm Development",
                "description": "Develop and optimize machine learning algorithms for recommendations",
                "complexity": TaskComplexity(0.9),
                "priority": TaskPriority.URGENT
            },
            {
                "pattern": "implementation",
                "name": "Core System Implementation",
                "description": "Implement the core recommendation engine and APIs",
                "complexity": TaskComplexity(0.85),
                "priority": TaskPriority.HIGH
            },
            {
                "pattern": "integration",
                "name": "System Integration",
                "description": "Integrate all components and establish data pipelines",
                "complexity": TaskComplexity(0.7),
                "priority": TaskPriority.MEDIUM
            },
            {
                "pattern": "testing",
                "name": "Comprehensive Testing",
                "description": "Conduct unit, integration, and performance testing",
                "complexity": TaskComplexity(0.6),
                "priority": TaskPriority.HIGH
            },
            {
                "pattern": "documentation",
                "name": "Technical Documentation",
                "description": "Create comprehensive technical and user documentation",
                "complexity": TaskComplexity(0.5),
                "priority": TaskPriority.MEDIUM
            }
        ]
        
        # Apply enhancements and set up dependencies
        enhanced_tasks = []
        for i, task in enumerate(tasks[:len(task_enhancements)]):
            enhancement = task_enhancements[i]
            
            task.name = enhancement["name"]
            task.description = enhancement["description"]
            task.complexity = enhancement["complexity"]
            task.priority = enhancement["priority"]
            task.estimated_duration = timedelta(
                days=2 + int(task.complexity_score * 3)
            )
            
            # Set up dependencies (simple sequential with some parallelism)
            if i > 0:
                # Most tasks depend on previous task
                task.dependencies.add(enhanced_tasks[i-1].task_id)
            
            if i > 2 and "testing" not in task.name.lower():
                # Testing can start after implementation begins
                pass
            
            enhanced_tasks.append(task)
            self.framework.add_task(task)
        
        self.logger.info(f"Designed {len(enhanced_tasks)} tasks with dependencies")
        return enhanced_tasks
    
    async def create_execution_plan(self, goal: Goal, tasks: List[FeriqTask], roles: List[Role]) -> Plan:
        """Create an execution plan using the plan designer."""
        from feriq.components.plan_designer import PlanningStrategy
        
        # Create plan using the plan designer
        plan = self.plan_designer.design_plan_from_goal(
            goal,
            roles,
            strategy=PlanningStrategy.HYBRID
        )
        
        # Update plan with our designed tasks
        plan.tasks = tasks
        
        # Regenerate milestones based on actual tasks
        plan.milestones = self.plan_designer._generate_milestones(goal, tasks, None)
        
        self.framework.add_plan(plan)
        
        self.logger.info(f"Created execution plan", plan_id=plan.plan_id)
        return plan
    
    async def setup_agent_coordination(self, agents: List[FeriqAgent]):
        """Set up coordination patterns between agents."""
        from feriq.components.choreographer import CoordinationPattern, MessageType
        
        # Set up research coordination (parallel information gathering)
        research_agents = [a for a in agents if "Research" in a.name or "ML" in a.name]
        if len(research_agents) >= 2:
            coordination_id = self.choreographer.coordinate_agents(
                CoordinationPattern.SCATTER_GATHER,
                [a.agent_id for a in research_agents],
                research_agents[0].agent_id,
                {"task_data": {"research_topics": ["market_analysis", "competitor_analysis", "technology_trends"]}}
            )
            self.logger.info(f"Set up research coordination: {coordination_id}")
        
        # Set up development pipeline (sequential development flow)
        dev_agents = [a for a in agents if any(role in a.name for role in ["Architect", "Developer", "QA"])]
        if len(dev_agents) >= 2:
            coordination_id = self.choreographer.coordinate_agents(
                CoordinationPattern.PIPELINE,
                [a.agent_id for a in dev_agents],
                dev_agents[0].agent_id
            )
            self.logger.info(f"Set up development pipeline: {coordination_id}")
        
        # Set up broadcast for project updates
        all_agent_ids = [a.agent_id for a in agents]
        self.choreographer.broadcast_message(
            "system",
            MessageType.INFORMATION_SHARE,
            {
                "message": "Project kick-off: AI Recommendation System Development",
                "priority": "high",
                "timeline": "30 days"
            },
            set(all_agent_ids)
        )
        
        self.logger.info("Agent coordination patterns established")
    
    async def start_monitoring(self, plan: Plan) -> str:
        """Start monitoring the plan execution."""
        monitor_id = self.plan_observer.start_observing_plan(plan)
        
        # Add some custom event handlers
        def handle_task_completion(observation):
            self.logger.info(f"Task completed: {observation.task_id}")
        
        def handle_performance_metrics(observation):
            if observation.data.get("efficiency", 1.0) < 0.7:
                self.logger.warning("Performance below threshold", 
                                  efficiency=observation.data.get("efficiency"))
        
        self.plan_observer.add_event_handler("task_progress", handle_task_completion)
        self.plan_observer.add_event_handler("performance_metric", handle_performance_metrics)
        
        self.logger.info(f"Started plan monitoring: {monitor_id}")
        return monitor_id
    
    async def execute_workflow(self, plan: Plan, agents: List[FeriqAgent]) -> str:
        """Execute the workflow using the orchestrator."""
        from feriq.components.orchestrator import ExecutionStrategy
        
        # Register resources
        self.orchestrator.register_resource("compute_nodes", 10)
        self.orchestrator.register_resource("memory_gb", 64)
        self.orchestrator.register_resource("storage_gb", 1000)
        
        # Start workflow execution
        workflow_id = self.orchestrator.start_workflow(
            plan,
            strategy=ExecutionStrategy.DYNAMIC
        )
        
        self.logger.info(f"Started workflow execution: {workflow_id}")
        return workflow_id
    
    async def demonstrate_reasoning(self, goal: Goal, plan: Plan, agents: List[FeriqAgent]):
        """Demonstrate reasoning capabilities."""
        # Reason about the goal
        goal_decision = self.reasoner.reason_about_goal(goal)
        print(f"🧠 Goal reasoning confidence: {goal_decision.confidence:.2f}")
        print(f"   Rationale: {goal_decision.rationale}")
        
        # Reason about task assignment
        if plan.tasks:
            task_decision = self.reasoner.reason_about_task_assignment(
                plan.tasks[0], agents
            )
            recommended_agent = task_decision.decision_data.get("recommended_agent")
            print(f"🤖 Task assignment recommendation: Agent {recommended_agent}")
        
        # Reason about plan optimization
        optimization_decision = self.reasoner.reason_about_plan_optimization(plan)
        recommendations = optimization_decision.decision_data.get("recommendations", [])
        print(f"📈 Plan optimization recommendations: {len(recommendations)} found")
        
        # Add some context for future reasoning
        from feriq.components.reasoner import ReasoningContext, ContextType
        
        context = ReasoningContext(
            context_id="project_context_1",
            context_type=ContextType.SYSTEM_WIDE,
            timestamp=datetime.now(),
            data={
                "project_phase": "execution",
                "team_size": len(agents),
                "complexity_level": "high",
                "timeline_pressure": "medium",
                "resource_availability": "adequate"
            },
            relevance_score=0.9
        )
        
        self.reasoner.add_context(context)
        
        self.logger.info("Reasoning demonstrations completed")
    
    async def monitor_and_report(self, workflow_id: str, monitor_id: str):
        """Monitor execution and provide reports."""
        # Wait a bit for some execution to occur
        await asyncio.sleep(5)
        
        # Get workflow status
        workflow_status = self.orchestrator.get_workflow_status(workflow_id)
        if workflow_status:
            print(f"📊 Workflow Progress: {workflow_status['progress']:.1%}")
            print(f"   Active tasks: {workflow_status['active_tasks']}")
            print(f"   Completed tasks: {workflow_status['completed_tasks']}")
        
        # Get plan status
        plan_status = self.plan_observer.get_plan_status(workflow_status['plan_id'])
        if plan_status:
            print(f"🔍 Plan monitoring active: {plan_status['is_monitoring']}")
            print(f"   High severity alerts: {plan_status['high_severity_alerts']}")
        
        # Get choreographer statistics
        choreo_stats = self.choreographer.get_choreographer_statistics()
        print(f"🎭 Agent coordination:")
        print(f"   Registered agents: {choreo_stats['registered_agents']}")
        print(f"   Active interactions: {choreo_stats['active_interactions']}")
        
        self.logger.info("Monitoring and reporting completed")
    
    async def show_final_statistics(self):
        """Show final statistics from all components."""
        print("\n📈 Final Framework Statistics:")
        print("-" * 40)
        
        # Framework statistics
        framework_stats = self.framework.get_framework_status()
        print(f"Framework - Goals: {framework_stats['goals']}, Agents: {framework_stats['agents']}")
        
        # Orchestrator statistics
        orch_stats = self.orchestrator.get_orchestrator_status()
        print(f"Orchestrator - Active workflows: {orch_stats['active_workflows']}")
        
        # Reasoning statistics
        reason_stats = self.reasoner.get_reasoning_statistics()
        print(f"Reasoner - Decisions made: {reason_stats['total_decisions']}")
        print(f"         - Success rate: {reason_stats['success_rate']:.1%}")
        
        # Observer statistics
        observer_stats = self.plan_observer.get_observer_statistics()
        print(f"Observer - Total observations: {observer_stats['total_observations']}")
        
        # Task allocator statistics
        allocator_stats = self.task_allocator.get_allocator_statistics()
        print(f"Allocator - Allocations made: {allocator_stats['total_allocations']}")
        
        self.logger.info("Final statistics displayed")


async def main():
    """Main function to run the Feriq example."""
    example = FeriqExample()
    await example.run_complete_example()


if __name__ == "__main__":
    print("Feriq Framework - Comprehensive Example")
    print("This example demonstrates collaborative AI agents working together")
    print("to develop an AI-powered recommendation system.\n")
    
    asyncio.run(main())