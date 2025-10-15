"""
Team Designer Component - Feriq Framework

Manages team creation, composition, collaboration, and goal-oriented coordination.
Enables multi-agent teams with specialized disciplines to work together on complex problems.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

class TeamType(Enum):
    """Types of teams based on collaboration patterns"""
    AUTONOMOUS = "autonomous"  # Self-organizing team
    HIERARCHICAL = "hierarchical"  # Leadership-based team
    SPECIALIST = "specialist"  # Domain-specific expertise team
    CROSS_FUNCTIONAL = "cross_functional"  # Multi-discipline team
    SWARM = "swarm"  # Distributed decision-making team

class TeamStatus(Enum):
    """Team operational status"""
    FORMING = "forming"  # Team being assembled
    ACTIVE = "active"  # Team actively working
    COLLABORATING = "collaborating"  # Team working with other teams
    PAUSED = "paused"  # Team temporarily inactive
    COMPLETED = "completed"  # Team goals accomplished
    DISBANDED = "disbanded"  # Team dissolved

class CollaborationMode(Enum):
    """How teams collaborate with each other"""
    INDEPENDENT = "independent"  # No collaboration
    COOPERATIVE = "cooperative"  # Share information and resources
    COORDINATED = "coordinated"  # Synchronized execution
    INTEGRATED = "integrated"  # Deep collaboration and shared goals

@dataclass
class TeamMember:
    """Individual team member representation"""
    role_id: str
    role_name: str
    specialization: str
    capabilities: List[str]
    contribution_level: float  # 0.0 to 1.0
    availability: bool = True
    joined_at: str = None
    
    def __post_init__(self):
        if self.joined_at is None:
            self.joined_at = datetime.now().isoformat()

@dataclass
class TeamGoal:
    """Goal that teams work towards"""
    id: str
    title: str
    description: str
    priority: str  # high, medium, low, critical
    complexity: float  # 0.0 to 1.0
    estimated_effort: int  # in hours
    deadline: Optional[str] = None
    status: str = "active"  # active, completed, blocked, deferred
    sub_goals: List[str] = None
    assigned_teams: List[str] = None
    progress: float = 0.0
    
    def __post_init__(self):
        if self.sub_goals is None:
            self.sub_goals = []
        if self.assigned_teams is None:
            self.assigned_teams = []

@dataclass
class TeamCollaboration:
    """Collaboration between teams"""
    id: str
    team_ids: List[str]
    collaboration_type: str
    shared_goals: List[str]
    communication_channels: List[str]
    coordination_rules: Dict[str, Any]
    status: str = "active"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class Team:
    """Core team representation"""
    id: str
    name: str
    description: str
    team_type: str
    discipline: str  # Domain of expertise
    members: List[TeamMember]
    goals: List[TeamGoal]
    capabilities: List[str]
    status: str = "forming"
    max_size: int = 10
    min_size: int = 1
    collaboration_mode: str = "cooperative"
    communication_protocols: List[str] = None
    decision_making_style: str = "consensus"  # consensus, democratic, authoritative
    performance_metrics: Dict[str, float] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.communication_protocols is None:
            self.communication_protocols = ["direct", "broadcast", "hierarchical"]
        if self.performance_metrics is None:
            self.performance_metrics = {
                "efficiency": 0.0,
                "collaboration_score": 0.0,
                "goal_completion_rate": 0.0,
                "adaptability": 0.0
            }
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()

class TeamDesigner:
    """
    Team Designer Component
    
    Manages team creation, composition, collaboration, and goal-oriented coordination.
    Enables multi-agent teams with specialized disciplines to work together autonomously.
    """
    
    def __init__(self, output_dir: str = "feriq/outputs/teams"):
        self.output_dir = output_dir
        self.teams: Dict[str, Team] = {}
        self.collaborations: Dict[str, TeamCollaboration] = {}
        self.goals: Dict[str, TeamGoal] = {}
        
    def create_team(self, 
                   name: str,
                   description: str,
                   discipline: str,
                   team_type: TeamType = TeamType.AUTONOMOUS,
                   max_size: int = 10,
                   min_size: int = 1,
                   capabilities: List[str] = None) -> Team:
        """Create a new team with specified parameters"""
        
        team_id = str(uuid.uuid4())
        
        if capabilities is None:
            capabilities = self._infer_capabilities_from_discipline(discipline)
        
        team = Team(
            id=team_id,
            name=name,
            description=description,
            team_type=team_type.value,
            discipline=discipline,
            members=[],
            goals=[],
            capabilities=capabilities,
            max_size=max_size,
            min_size=min_size
        )
        
        self.teams[team_id] = team
        self._save_team(team)
        
        return team
    
    def add_member_to_team(self, 
                          team_id: str, 
                          role_id: str, 
                          role_name: str,
                          specialization: str,
                          capabilities: List[str],
                          contribution_level: float = 1.0) -> bool:
        """Add a role as a member to a team"""
        
        if team_id not in self.teams:
            return False
        
        team = self.teams[team_id]
        
        # Check if already a member
        if any(member.role_id == role_id for member in team.members):
            return False
        
        # Check team capacity
        if len(team.members) >= team.max_size:
            return False
        
        member = TeamMember(
            role_id=role_id,
            role_name=role_name,
            specialization=specialization,
            capabilities=capabilities,
            contribution_level=contribution_level
        )
        
        team.members.append(member)
        team.updated_at = datetime.now().isoformat()
        
        # Update team capabilities
        self._update_team_capabilities(team)
        
        # Update team status if it has minimum members
        if len(team.members) >= team.min_size and team.status == "forming":
            team.status = "active"
        
        self._save_team(team)
        return True
    
    def create_team_goal(self,
                        title: str,
                        description: str,
                        priority: str = "medium",
                        complexity: float = 0.5,
                        estimated_effort: int = 40,
                        deadline: Optional[str] = None) -> TeamGoal:
        """Create a goal that teams can work towards"""
        
        goal_id = str(uuid.uuid4())
        
        goal = TeamGoal(
            id=goal_id,
            title=title,
            description=description,
            priority=priority,
            complexity=complexity,
            estimated_effort=estimated_effort,
            deadline=deadline
        )
        
        self.goals[goal_id] = goal
        self._save_goal(goal)
        
        return goal
    
    def assign_goal_to_team(self, goal_id: str, team_id: str) -> bool:
        """Assign a goal to a specific team"""
        
        if goal_id not in self.goals or team_id not in self.teams:
            return False
        
        goal = self.goals[goal_id]
        team = self.teams[team_id]
        
        # Add goal to team
        if goal not in team.goals:
            team.goals.append(goal)
        
        # Add team to goal's assigned teams
        if team_id not in goal.assigned_teams:
            goal.assigned_teams.append(team_id)
        
        team.updated_at = datetime.now().isoformat()
        
        self._save_team(team)
        self._save_goal(goal)
        
        return True
    
    def create_team_collaboration(self,
                                 team_ids: List[str],
                                 collaboration_type: CollaborationMode,
                                 shared_goals: List[str] = None,
                                 coordination_rules: Dict[str, Any] = None) -> TeamCollaboration:
        """Create collaboration between multiple teams"""
        
        collaboration_id = str(uuid.uuid4())
        
        if shared_goals is None:
            shared_goals = []
        
        if coordination_rules is None:
            coordination_rules = {
                "communication_frequency": "daily",
                "decision_making": "consensus",
                "resource_sharing": True,
                "conflict_resolution": "escalation"
            }
        
        collaboration = TeamCollaboration(
            id=collaboration_id,
            team_ids=team_ids,
            collaboration_type=collaboration_type.value,
            shared_goals=shared_goals,
            communication_channels=["team_chat", "status_updates", "shared_workspace"],
            coordination_rules=coordination_rules
        )
        
        self.collaborations[collaboration_id] = collaboration
        self._save_collaboration(collaboration)
        
        # Update team collaboration modes
        for team_id in team_ids:
            if team_id in self.teams:
                team = self.teams[team_id]
                team.status = "collaborating"
                team.updated_at = datetime.now().isoformat()
                self._save_team(team)
        
        return collaboration
    
    def extract_and_refine_goals(self, team_id: str, problem_description: str) -> List[TeamGoal]:
        """
        Extract and refine goals from a problem description using team intelligence
        This simulates AI-powered goal extraction and refinement
        """
        
        if team_id not in self.teams:
            return []
        
        team = self.teams[team_id]
        
        # Simulate intelligent goal extraction based on team discipline and capabilities
        extracted_goals = self._simulate_goal_extraction(team, problem_description)
        
        # Refine goals based on team expertise
        refined_goals = self._refine_goals_with_team_intelligence(team, extracted_goals)
        
        # Create and assign goals to team
        created_goals = []
        for goal_data in refined_goals:
            goal = self.create_team_goal(
                title=goal_data["title"],
                description=goal_data["description"],
                priority=goal_data["priority"],
                complexity=goal_data["complexity"],
                estimated_effort=goal_data["estimated_effort"]
            )
            
            self.assign_goal_to_team(goal.id, team_id)
            created_goals.append(goal)
        
        return created_goals
    
    def get_team_performance_metrics(self, team_id: str) -> Dict[str, float]:
        """Calculate performance metrics for a team"""
        
        if team_id not in self.teams:
            return {}
        
        team = self.teams[team_id]
        
        # Calculate various performance metrics
        efficiency = self._calculate_team_efficiency(team)
        collaboration_score = self._calculate_collaboration_score(team)
        goal_completion_rate = self._calculate_goal_completion_rate(team)
        adaptability = self._calculate_adaptability_score(team)
        
        metrics = {
            "efficiency": efficiency,
            "collaboration_score": collaboration_score,
            "goal_completion_rate": goal_completion_rate,
            "adaptability": adaptability,
            "overall_performance": (efficiency + collaboration_score + goal_completion_rate + adaptability) / 4
        }
        
        # Update team metrics
        team.performance_metrics.update(metrics)
        team.updated_at = datetime.now().isoformat()
        self._save_team(team)
        
        return metrics
    
    def simulate_autonomous_problem_solving(self, 
                                          team_id: str, 
                                          problem: str) -> Dict[str, Any]:
        """
        Simulate autonomous problem-solving by a team
        Teams can analyze problems, extract goals, assign tasks, and coordinate solutions
        """
        
        if team_id not in self.teams:
            return {"error": "Team not found"}
        
        team = self.teams[team_id]
        
        # Step 1: Problem analysis by team
        analysis = self._analyze_problem_with_team(team, problem)
        
        # Step 2: Extract and refine goals
        goals = self.extract_and_refine_goals(team_id, problem)
        
        # Step 3: Task breakdown and assignment
        tasks = self._break_down_goals_into_tasks(team, goals)
        
        # Step 4: Resource allocation
        resource_plan = self._allocate_team_resources(team, tasks)
        
        # Step 5: Execution planning
        execution_plan = self._create_team_execution_plan(team, tasks, resource_plan)
        
        # Step 6: Collaboration requirements
        collaboration_needs = self._identify_collaboration_needs(team, goals, tasks)
        
        solution = {
            "team_id": team_id,
            "team_name": team.name,
            "problem": problem,
            "analysis": analysis,
            "extracted_goals": [asdict(goal) for goal in goals],
            "task_breakdown": tasks,
            "resource_allocation": resource_plan,
            "execution_plan": execution_plan,
            "collaboration_requirements": collaboration_needs,
            "estimated_completion_time": execution_plan.get("total_time", 0),
            "confidence_score": analysis.get("confidence", 0.8),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save the solution
        self._save_solution(solution)
        
        return solution
    
    def get_teams_by_discipline(self, discipline: str) -> List[Team]:
        """Get all teams specialized in a specific discipline"""
        return [team for team in self.teams.values() if team.discipline == discipline]
    
    def get_available_teams(self) -> List[Team]:
        """Get all teams that are available for new work"""
        return [team for team in self.teams.values() 
                if team.status in ["active", "forming"] and len(team.goals) < 5]
    
    def get_collaborating_teams(self) -> List[TeamCollaboration]:
        """Get all active team collaborations"""
        return [collab for collab in self.collaborations.values() 
                if collab.status == "active"]
    
    # Private helper methods
    
    def _infer_capabilities_from_discipline(self, discipline: str) -> List[str]:
        """Infer team capabilities based on discipline"""
        
        discipline_capabilities = {
            "software_development": [
                "coding", "testing", "debugging", "architecture_design", 
                "code_review", "deployment", "documentation"
            ],
            "data_science": [
                "data_analysis", "machine_learning", "statistical_modeling",
                "data_visualization", "feature_engineering", "model_evaluation"
            ],
            "research": [
                "literature_review", "hypothesis_formation", "experimental_design",
                "data_collection", "analysis", "paper_writing", "peer_review"
            ],
            "marketing": [
                "market_research", "campaign_design", "content_creation",
                "social_media", "analytics", "customer_engagement"
            ],
            "finance": [
                "financial_analysis", "budgeting", "forecasting", "risk_assessment",
                "investment_planning", "compliance", "reporting"
            ],
            "design": [
                "ui_design", "ux_research", "prototyping", "user_testing",
                "visual_design", "interaction_design", "accessibility"
            ],
            "operations": [
                "process_optimization", "resource_management", "quality_assurance",
                "supply_chain", "logistics", "performance_monitoring"
            ]
        }
        
        return discipline_capabilities.get(discipline.lower(), ["problem_solving", "communication", "collaboration"])
    
    def _update_team_capabilities(self, team: Team):
        """Update team capabilities based on member capabilities"""
        all_capabilities = set(team.capabilities)
        
        for member in team.members:
            all_capabilities.update(member.capabilities)
        
        team.capabilities = list(all_capabilities)
    
    def _simulate_goal_extraction(self, team: Team, problem: str) -> List[Dict[str, Any]]:
        """Simulate AI-powered goal extraction from problem description"""
        
        # This would integrate with AI reasoning in a real implementation
        # For now, we simulate intelligent goal extraction
        
        base_goals = [
            {
                "title": f"Analyze {problem[:30]}...",
                "description": f"Conduct thorough analysis of the problem: {problem}",
                "priority": "high",
                "complexity": 0.6,
                "estimated_effort": 20
            },
            {
                "title": f"Design solution approach",
                "description": f"Design comprehensive solution approach using {team.discipline} expertise",
                "priority": "high",
                "complexity": 0.8,
                "estimated_effort": 40
            },
            {
                "title": f"Implement solution",
                "description": f"Execute the designed solution with team coordination",
                "priority": "medium",
                "complexity": 0.7,
                "estimated_effort": 60
            },
            {
                "title": f"Validate and refine",
                "description": f"Test, validate, and refine the solution based on results",
                "priority": "medium",
                "complexity": 0.5,
                "estimated_effort": 30
            }
        ]
        
        # Customize goals based on team discipline
        if team.discipline == "software_development":
            base_goals.extend([
                {
                    "title": "Code architecture design",
                    "description": "Design scalable and maintainable code architecture",
                    "priority": "high",
                    "complexity": 0.9,
                    "estimated_effort": 50
                },
                {
                    "title": "Testing strategy",
                    "description": "Develop comprehensive testing strategy and implementation",
                    "priority": "medium",
                    "complexity": 0.6,
                    "estimated_effort": 30
                }
            ])
        elif team.discipline == "data_science":
            base_goals.extend([
                {
                    "title": "Data exploration and preprocessing",
                    "description": "Explore, clean, and preprocess relevant data",
                    "priority": "high",
                    "complexity": 0.7,
                    "estimated_effort": 35
                },
                {
                    "title": "Model development and validation",
                    "description": "Develop and validate predictive/analytical models",
                    "priority": "high",
                    "complexity": 0.8,
                    "estimated_effort": 45
                }
            ])
        
        return base_goals
    
    def _refine_goals_with_team_intelligence(self, team: Team, goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refine goals using team intelligence and capabilities"""
        
        refined_goals = []
        
        for goal in goals:
            # Adjust complexity based on team capabilities
            capability_match = len(set(team.capabilities) & set(goal.get("required_capabilities", []))) / max(len(goal.get("required_capabilities", ["general"])), 1)
            goal["complexity"] = max(0.1, goal["complexity"] - (capability_match * 0.3))
            
            # Adjust effort based on team size and experience
            team_efficiency = min(1.0, len(team.members) / 5.0)  # Optimal team size around 5
            goal["estimated_effort"] = int(goal["estimated_effort"] * (1.0 - team_efficiency * 0.2))
            
            # Adjust priority based on team discipline alignment
            if team.discipline.lower() in goal["description"].lower():
                if goal["priority"] == "medium":
                    goal["priority"] = "high"
                elif goal["priority"] == "low":
                    goal["priority"] = "medium"
            
            refined_goals.append(goal)
        
        return refined_goals
    
    def _analyze_problem_with_team(self, team: Team, problem: str) -> Dict[str, Any]:
        """Simulate team-based problem analysis"""
        
        analysis = {
            "problem_complexity": min(1.0, len(problem) / 500.0),
            "team_capability_match": self._calculate_capability_match(team, problem),
            "estimated_duration": self._estimate_problem_duration(team, problem),
            "resource_requirements": self._estimate_resource_requirements(team, problem),
            "collaboration_needs": self._assess_collaboration_needs(team, problem),
            "confidence": 0.8,  # Base confidence
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Adjust confidence based on team capability match
        analysis["confidence"] *= analysis["team_capability_match"]
        
        return analysis
    
    def _break_down_goals_into_tasks(self, team: Team, goals: List[TeamGoal]) -> List[Dict[str, Any]]:
        """Break down team goals into specific tasks"""
        
        tasks = []
        
        for goal in goals:
            # Simulate intelligent task breakdown
            task_count = max(2, min(6, int(goal.complexity * 8)))
            
            for i in range(task_count):
                task = {
                    "id": str(uuid.uuid4()),
                    "goal_id": goal.id,
                    "title": f"Task {i+1} for {goal.title}",
                    "description": f"Subtask {i+1} contributing to goal: {goal.title}",
                    "assigned_member": self._assign_task_to_member(team, goal),
                    "estimated_effort": goal.estimated_effort // task_count,
                    "dependencies": [],
                    "priority": goal.priority,
                    "status": "pending"
                }
                tasks.append(task)
        
        return tasks
    
    def _allocate_team_resources(self, team: Team, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Allocate team resources to tasks"""
        
        resource_plan = {
            "total_effort_hours": sum(task["estimated_effort"] for task in tasks),
            "team_capacity": len(team.members) * 40,  # 40 hours per week per member
            "utilization_rate": 0.0,
            "member_assignments": {},
            "timeline": "TBD"
        }
        
        # Calculate utilization
        if resource_plan["team_capacity"] > 0:
            resource_plan["utilization_rate"] = min(1.0, resource_plan["total_effort_hours"] / resource_plan["team_capacity"])
        
        # Assign members to tasks
        for member in team.members:
            member_tasks = [task for task in tasks if task.get("assigned_member") == member.role_name]
            resource_plan["member_assignments"][member.role_name] = {
                "tasks": len(member_tasks),
                "effort_hours": sum(task["estimated_effort"] for task in member_tasks),
                "specialization_match": member.specialization
            }
        
        return resource_plan
    
    def _create_team_execution_plan(self, team: Team, tasks: List[Dict[str, Any]], resource_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan for team"""
        
        execution_plan = {
            "phases": [
                {
                    "name": "Planning and Setup",
                    "duration_hours": 16,
                    "tasks": [task for task in tasks if "plan" in task["title"].lower()]
                },
                {
                    "name": "Implementation",
                    "duration_hours": int(resource_plan["total_effort_hours"] * 0.7),
                    "tasks": [task for task in tasks if "implement" in task["title"].lower()]
                },
                {
                    "name": "Testing and Validation",
                    "duration_hours": int(resource_plan["total_effort_hours"] * 0.2),
                    "tasks": [task for task in tasks if "test" in task["title"].lower() or "validate" in task["title"].lower()]
                },
                {
                    "name": "Delivery and Handoff",
                    "duration_hours": 8,
                    "tasks": [task for task in tasks if "deliver" in task["title"].lower()]
                }
            ],
            "total_time": resource_plan["total_effort_hours"] + 24,  # Add overhead
            "parallel_execution": len(team.members) > 1,
            "milestones": [],
            "risk_factors": self._identify_execution_risks(team, tasks)
        }
        
        return execution_plan
    
    def _identify_collaboration_needs(self, team: Team, goals: List[TeamGoal], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify what collaboration this team might need with other teams"""
        
        collaboration_needs = {
            "external_expertise_needed": [],
            "resource_sharing_opportunities": [],
            "coordination_requirements": [],
            "recommended_collaborations": []
        }
        
        # Analyze goals and tasks for collaboration indicators
        for goal in goals:
            if goal.complexity > 0.8:
                collaboration_needs["external_expertise_needed"].append({
                    "goal": goal.title,
                    "expertise_type": "high_complexity_problem_solving",
                    "reason": "Complex goal requiring additional expertise"
                })
        
        # Check for cross-disciplinary needs
        team_capabilities = set(team.capabilities)
        all_required_capabilities = set()
        
        for task in tasks:
            task_requirements = task.get("required_capabilities", [])
            all_required_capabilities.update(task_requirements)
        
        missing_capabilities = all_required_capabilities - team_capabilities
        if missing_capabilities:
            collaboration_needs["recommended_collaborations"].append({
                "type": "capability_gap",
                "missing_capabilities": list(missing_capabilities),
                "suggested_disciplines": self._suggest_disciplines_for_capabilities(missing_capabilities)
            })
        
        return collaboration_needs
    
    def _calculate_team_efficiency(self, team: Team) -> float:
        """Calculate team efficiency score"""
        base_efficiency = 0.7
        
        # Adjust for team size (optimal around 5-7 members)
        size_factor = 1.0 - abs(len(team.members) - 6) * 0.05
        
        # Adjust for capability diversity
        diversity_factor = min(1.0, len(team.capabilities) / 10.0)
        
        # Adjust for member availability
        availability_factor = sum(1 for member in team.members if member.availability) / max(len(team.members), 1)
        
        efficiency = base_efficiency * size_factor * diversity_factor * availability_factor
        return min(1.0, max(0.0, efficiency))
    
    def _calculate_collaboration_score(self, team: Team) -> float:
        """Calculate how well the team collaborates"""
        base_score = 0.8
        
        # Higher scores for teams with diverse capabilities
        capability_diversity = len(team.capabilities) / 15.0
        
        # Communication protocol effectiveness
        protocol_score = len(team.communication_protocols) / 5.0
        
        collaboration_score = base_score * min(1.0, capability_diversity) * min(1.0, protocol_score)
        return min(1.0, max(0.0, collaboration_score))
    
    def _calculate_goal_completion_rate(self, team: Team) -> float:
        """Calculate goal completion rate for the team"""
        if not team.goals:
            return 0.5  # Neutral score for new teams
        
        completed_goals = sum(1 for goal in team.goals if goal.status == "completed")
        return completed_goals / len(team.goals)
    
    def _calculate_adaptability_score(self, team: Team) -> float:
        """Calculate team adaptability score"""
        base_adaptability = 0.7
        
        # Teams with diverse capabilities are more adaptable
        capability_factor = min(1.0, len(team.capabilities) / 12.0)
        
        # Autonomous teams tend to be more adaptable
        autonomy_factor = 1.0 if team.team_type == "autonomous" else 0.8
        
        adaptability = base_adaptability * capability_factor * autonomy_factor
        return min(1.0, max(0.0, adaptability))
    
    def _calculate_capability_match(self, team: Team, problem: str) -> float:
        """Calculate how well team capabilities match the problem"""
        # Simple keyword matching for simulation
        problem_keywords = problem.lower().split()
        capability_matches = sum(1 for cap in team.capabilities 
                               if any(keyword in cap.lower() for keyword in problem_keywords))
        
        if not team.capabilities:
            return 0.5
        
        return min(1.0, capability_matches / len(team.capabilities))
    
    def _estimate_problem_duration(self, team: Team, problem: str) -> int:
        """Estimate how long it will take the team to solve the problem"""
        base_duration = max(40, len(problem) // 10)  # Base hours
        
        # Adjust for team capability match
        capability_match = self._calculate_capability_match(team, problem)
        duration_factor = 2.0 - capability_match  # Better match = less time
        
        # Adjust for team size
        team_size_factor = max(0.5, 1.0 - (len(team.members) - 1) * 0.1)
        
        estimated_duration = int(base_duration * duration_factor * team_size_factor)
        return max(8, estimated_duration)  # Minimum 8 hours
    
    def _estimate_resource_requirements(self, team: Team, problem: str) -> Dict[str, Any]:
        """Estimate resource requirements for solving the problem"""
        duration = self._estimate_problem_duration(team, problem)
        
        return {
            "human_hours": duration,
            "team_members": len(team.members),
            "specialized_tools": self._identify_required_tools(team, problem),
            "external_resources": [],
            "budget_estimate": duration * 50  # $50/hour placeholder
        }
    
    def _assess_collaboration_needs(self, team: Team, problem: str) -> List[str]:
        """Assess what types of collaboration this team might need"""
        needs = []
        
        # Check problem complexity
        if len(problem) > 200:
            needs.append("cross_functional_expertise")
        
        # Check team capability gaps
        if len(team.capabilities) < 5:
            needs.append("additional_capabilities")
        
        # Check for specialized domains
        if any(term in problem.lower() for term in ["ai", "machine learning", "data", "algorithm"]):
            if "data_science" not in team.discipline:
                needs.append("data_science_expertise")
        
        return needs
    
    def _assign_task_to_member(self, team: Team, goal: TeamGoal) -> str:
        """Assign a task to the most suitable team member"""
        if not team.members:
            return "Unassigned"
        
        # Simple assignment based on specialization match
        best_member = team.members[0]
        for member in team.members:
            if member.specialization.lower() in goal.description.lower():
                best_member = member
                break
        
        return best_member.role_name
    
    def _identify_execution_risks(self, team: Team, tasks: List[Dict[str, Any]]) -> List[str]:
        """Identify potential risks in execution"""
        risks = []
        
        if len(team.members) < team.min_size:
            risks.append("Insufficient team members")
        
        if len(tasks) > len(team.members) * 3:
            risks.append("Task overload")
        
        capability_diversity = len(team.capabilities)
        if capability_diversity < 3:
            risks.append("Limited capability diversity")
        
        return risks
    
    def _suggest_disciplines_for_capabilities(self, capabilities: Set[str]) -> List[str]:
        """Suggest disciplines that might provide missing capabilities"""
        suggestions = []
        
        for capability in capabilities:
            if any(term in capability.lower() for term in ["code", "software", "programming"]):
                suggestions.append("software_development")
            elif any(term in capability.lower() for term in ["data", "analysis", "model"]):
                suggestions.append("data_science")
            elif any(term in capability.lower() for term in ["design", "visual", "user"]):
                suggestions.append("design")
            elif any(term in capability.lower() for term in ["research", "study", "investigation"]):
                suggestions.append("research")
        
        return list(set(suggestions))
    
    def _identify_required_tools(self, team: Team, problem: str) -> List[str]:
        """Identify tools that might be needed"""
        tools = ["communication_platform", "project_management"]
        
        if team.discipline == "software_development":
            tools.extend(["ide", "version_control", "testing_framework"])
        elif team.discipline == "data_science":
            tools.extend(["python", "jupyter", "data_visualization", "ml_frameworks"])
        elif team.discipline == "design":
            tools.extend(["design_software", "prototyping_tools", "collaboration_platform"])
        
        return tools
    
    # File I/O methods
    
    def _save_team(self, team: Team):
        """Save team to file"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        filename = f"{self.output_dir}/team_{team.id}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(team), f, indent=2, default=str)
    
    def _save_goal(self, goal: TeamGoal):
        """Save goal to file"""
        import os
        goals_dir = f"{self.output_dir}/goals"
        os.makedirs(goals_dir, exist_ok=True)
        
        filename = f"{goals_dir}/goal_{goal.id}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(goal), f, indent=2, default=str)
    
    def _save_collaboration(self, collaboration: TeamCollaboration):
        """Save collaboration to file"""
        import os
        collab_dir = f"{self.output_dir}/collaborations"
        os.makedirs(collab_dir, exist_ok=True)
        
        filename = f"{collab_dir}/collaboration_{collaboration.id}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(collaboration), f, indent=2, default=str)
    
    def _save_solution(self, solution: Dict[str, Any]):
        """Save autonomous solution to file"""
        import os
        solutions_dir = f"{self.output_dir}/solutions"
        os.makedirs(solutions_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{solutions_dir}/solution_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(solution, f, indent=2, default=str)