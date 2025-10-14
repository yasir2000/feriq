"""Dynamic Role Designer component for creating and assigning roles based on task requirements."""

from typing import Dict, List, Any, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime
import json

from ..core.role import Role, RoleType, RoleCapability
from ..core.task import FeriqTask, TaskType, TaskComplexity
from ..core.goal import Goal


class RoleTemplate(BaseModel):
    """Template for creating new roles."""
    name: str = Field(..., description="Template name")
    role_type: RoleType = Field(..., description="Role type")
    base_capabilities: List[RoleCapability] = Field(default_factory=list,
                                                   description="Base capabilities")
    responsibilities_template: List[str] = Field(default_factory=list,
                                               description="Template responsibilities")
    constraints_template: List[str] = Field(default_factory=list,
                                          description="Template constraints")
    adaptable_attributes: List[str] = Field(default_factory=list,
                                          description="Attributes that can be adapted")


class RoleRequirement(BaseModel):
    """Represents requirements for a role."""
    required_capabilities: Dict[str, float] = Field(default_factory=dict,
                                                   description="Required capabilities and levels")
    preferred_role_types: List[RoleType] = Field(default_factory=list,
                                               description="Preferred role types")
    task_complexity: Optional[TaskComplexity] = Field(default=None,
                                                     description="Task complexity level")
    collaboration_needs: bool = Field(default=False,
                                     description="Whether role needs collaboration")
    autonomy_requirement: float = Field(default=0.5, ge=0.0, le=1.0,
                                       description="Required autonomy level")
    specialization_depth: float = Field(default=0.5, ge=0.0, le=1.0,
                                       description="Required specialization depth")


class DynamicRoleDesigner(BaseModel):
    """
    Dynamic Role Designer that creates and assigns roles based on
    task requirements, agent capabilities, and contextual needs.
    """
    
    # Component identification
    name: str = Field(default="DynamicRoleDesigner", description="Component name")
    version: str = Field(default="1.0", description="Component version")
    
    # Templates and patterns
    role_templates: Dict[str, RoleTemplate] = Field(default_factory=dict,
                                                   description="Available role templates")
    role_patterns: Dict[str, Any] = Field(default_factory=dict,
                                         description="Common role patterns")
    
    # Knowledge base
    capability_ontology: Dict[str, Any] = Field(default_factory=dict,
                                               description="Capability relationships and hierarchy")
    domain_knowledge: Dict[str, Any] = Field(default_factory=dict,
                                           description="Domain-specific knowledge")
    
    # Adaptation and learning
    role_performance_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict,
                                                                     description="Role performance tracking")
    adaptation_rules: List[Dict[str, Any]] = Field(default_factory=list,
                                                  description="Rules for role adaptation")
    
    # Framework reference
    framework: Optional[Any] = Field(default=None, description="Reference to main framework")
    
    # Configuration
    auto_create_roles: bool = Field(default=True, description="Whether to auto-create roles")
    max_roles_per_agent: int = Field(default=5, description="Maximum roles per agent")
    role_adaptation_threshold: float = Field(default=0.7, description="Threshold for role adaptation")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize the role designer with default templates."""
        super().__init__(**kwargs)
        self._initialize_default_templates()
        self._initialize_capability_ontology()
    
    def _initialize_default_templates(self) -> None:
        """Initialize default role templates."""
        # Research Specialist Template
        research_template = RoleTemplate(
            name="Research Specialist",
            role_type=RoleType.RESEARCHER,
            base_capabilities=[
                RoleCapability(name="information_gathering", description="Ability to gather information", proficiency_level=0.8),
                RoleCapability(name="data_analysis", description="Ability to analyze data", proficiency_level=0.7),
                RoleCapability(name="critical_thinking", description="Critical thinking skills", proficiency_level=0.9),
                RoleCapability(name="documentation", description="Documentation skills", proficiency_level=0.8)
            ],
            responsibilities_template=[
                "Conduct thorough research on assigned topics",
                "Analyze and synthesize information from multiple sources",
                "Provide evidence-based recommendations",
                "Document findings and methodology"
            ],
            constraints_template=[
                "Must verify information accuracy",
                "Should use reliable sources",
                "Must document source citations"
            ],
            adaptable_attributes=["proficiency_level", "specialization_area", "research_methods"]
        )
        
        # Data Analyst Template
        analyst_template = RoleTemplate(
            name="Data Analyst",
            role_type=RoleType.ANALYST,
            base_capabilities=[
                RoleCapability(name="data_analysis", description="Data analysis expertise", proficiency_level=0.9),
                RoleCapability(name="statistical_modeling", description="Statistical modeling", proficiency_level=0.8),
                RoleCapability(name="visualization", description="Data visualization", proficiency_level=0.7),
                RoleCapability(name="pattern_recognition", description="Pattern recognition", proficiency_level=0.8)
            ],
            responsibilities_template=[
                "Analyze complex datasets",
                "Identify patterns and trends",
                "Create data visualizations",
                "Provide data-driven insights"
            ],
            constraints_template=[
                "Must ensure data quality",
                "Should validate analytical methods",
                "Must maintain data privacy"
            ],
            adaptable_attributes=["analytical_methods", "domain_expertise", "tool_proficiency"]
        )
        
        # Project Coordinator Template
        coordinator_template = RoleTemplate(
            name="Project Coordinator",
            role_type=RoleType.COORDINATOR,
            base_capabilities=[
                RoleCapability(name="project_management", description="Project management skills", proficiency_level=0.9),
                RoleCapability(name="communication", description="Communication skills", proficiency_level=0.9),
                RoleCapability(name="team_coordination", description="Team coordination", proficiency_level=0.8),
                RoleCapability(name="planning", description="Planning and scheduling", proficiency_level=0.8)
            ],
            responsibilities_template=[
                "Coordinate team activities",
                "Manage project timelines",
                "Facilitate communication",
                "Monitor progress and deliverables"
            ],
            constraints_template=[
                "Must maintain clear communication",
                "Should track project milestones",
                "Must resolve conflicts promptly"
            ],
            adaptable_attributes=["team_size", "project_complexity", "communication_style"]
        )
        
        # Store templates
        self.role_templates = {
            "researcher": research_template,
            "analyst": analyst_template,
            "coordinator": coordinator_template
        }
    
    def _initialize_capability_ontology(self) -> None:
        """Initialize the capability ontology with relationships."""
        self.capability_ontology = {
            "information_gathering": {
                "parent": None,
                "children": ["web_research", "database_querying", "interviewing"],
                "related": ["data_analysis", "critical_thinking"],
                "prerequisites": ["basic_research_skills"]
            },
            "data_analysis": {
                "parent": None,
                "children": ["statistical_analysis", "qualitative_analysis", "quantitative_analysis"],
                "related": ["pattern_recognition", "visualization"],
                "prerequisites": ["mathematical_reasoning"]
            },
            "project_management": {
                "parent": None,
                "children": ["planning", "scheduling", "resource_allocation"],
                "related": ["communication", "leadership"],
                "prerequisites": ["organizational_skills"]
            },
            "communication": {
                "parent": None,
                "children": ["written_communication", "verbal_communication", "presentation"],
                "related": ["leadership", "negotiation"],
                "prerequisites": ["language_skills"]
            }
        }
    
    def analyze_requirements(self, context: Dict[str, Any]) -> RoleRequirement:
        """Analyze context to determine role requirements."""
        requirements = RoleRequirement()
        
        # Extract task information
        if "task" in context:
            task = context["task"]
            if isinstance(task, dict):
                # Handle task dictionary
                complexity = task.get("complexity", TaskComplexity.MODERATE)
                task_type = task.get("task_type", TaskType.CUSTOM)
                required_capabilities = task.get("required_capabilities", [])
            elif hasattr(task, "complexity"):
                # Handle FeriqTask object
                complexity = task.complexity
                task_type = task.task_type
                required_capabilities = task.required_capabilities
            else:
                # Default values
                complexity = TaskComplexity.MODERATE
                task_type = TaskType.CUSTOM
                required_capabilities = []
            
            requirements.task_complexity = complexity
            
            # Map task type to preferred role types
            task_to_role_mapping = {
                TaskType.RESEARCH: [RoleType.RESEARCHER],
                TaskType.ANALYSIS: [RoleType.ANALYST],
                TaskType.PLANNING: [RoleType.PLANNER],
                TaskType.EXECUTION: [RoleType.EXECUTOR],
                TaskType.COORDINATION: [RoleType.COORDINATOR],
                TaskType.REVIEW: [RoleType.REVIEWER]
            }
            requirements.preferred_role_types = task_to_role_mapping.get(task_type, [RoleType.SPECIALIST])
            
            # Set capability requirements
            for capability in required_capabilities:
                level = self._determine_capability_level(capability, complexity)
                requirements.required_capabilities[capability] = level
        
        # Extract goal information
        if "goal" in context:
            goal = context["goal"]
            if hasattr(goal, "required_capabilities"):
                for capability in goal.required_capabilities:
                    if capability not in requirements.required_capabilities:
                        requirements.required_capabilities[capability] = 0.7
        
        # Determine collaboration needs
        if "collaboration" in context:
            requirements.collaboration_needs = bool(context["collaboration"])
        
        # Set autonomy requirements based on complexity
        autonomy_mapping = {
            TaskComplexity.TRIVIAL: 0.3,
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MODERATE: 0.6,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.EXPERT: 0.9
        }
        requirements.autonomy_requirement = autonomy_mapping.get(
            requirements.task_complexity, 0.5
        )
        
        return requirements
    
    def _determine_capability_level(self, capability: str, complexity: TaskComplexity) -> float:
        """Determine required capability level based on task complexity."""
        base_levels = {
            TaskComplexity.TRIVIAL: 0.3,
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MODERATE: 0.7,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.EXPERT: 0.9
        }
        return base_levels.get(complexity, 0.7)
    
    def design_role(self, requirements: Dict[str, Any]) -> Optional[Role]:
        """Design a role based on requirements."""
        # Analyze requirements
        role_req = self.analyze_requirements(requirements)
        
        # Try to find existing suitable role
        existing_role = self._find_suitable_existing_role(role_req)
        if existing_role:
            return existing_role
        
        # Create new role if auto-creation is enabled
        if self.auto_create_roles:
            return self._create_new_role(role_req, requirements)
        
        return None
    
    def _find_suitable_existing_role(self, requirements: RoleRequirement) -> Optional[Role]:
        """Find an existing role that meets the requirements."""
        if not self.framework or not self.framework.roles:
            return None
        
        best_role = None
        best_score = 0.0
        
        for role in self.framework.roles.values():
            score = self._calculate_role_suitability(role, requirements)
            if score > best_score and score >= self.role_adaptation_threshold:
                best_score = score
                best_role = role
        
        return best_role
    
    def _calculate_role_suitability(self, role: Role, requirements: RoleRequirement) -> float:
        """Calculate how suitable a role is for the requirements."""
        score = 0.0
        weight_sum = 0.0
        
        # Check capability match
        for req_cap, req_level in requirements.required_capabilities.items():
            role_level = role.get_capability_level(req_cap)
            
            if role_level >= req_level:
                score += 1.0 * req_level  # Perfect match
            else:
                score += (role_level / req_level) * req_level if req_level > 0 else 0.0
            
            weight_sum += req_level
        
        # Check role type preference
        if requirements.preferred_role_types and role.role_type in requirements.preferred_role_types:
            score += 0.5
            weight_sum += 0.5
        
        # Check autonomy match
        autonomy_diff = abs(role.autonomy_level - requirements.autonomy_requirement)
        autonomy_score = 1.0 - autonomy_diff
        score += autonomy_score * 0.3
        weight_sum += 0.3
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _create_new_role(self, requirements: RoleRequirement, context: Dict[str, Any]) -> Role:
        """Create a new role based on requirements."""
        # Select best template
        template = self._select_best_template(requirements)
        
        # Generate role name
        role_name = self._generate_role_name(requirements, context)
        
        # Create capabilities based on requirements
        capabilities = []
        for cap_name, req_level in requirements.required_capabilities.items():
            capability = RoleCapability(
                name=cap_name,
                description=f"Capability for {cap_name}",
                proficiency_level=req_level
            )
            capabilities.append(capability)
        
        # Add template capabilities that aren't already included
        if template:
            for template_cap in template.base_capabilities:
                if not any(cap.name == template_cap.name for cap in capabilities):
                    capabilities.append(template_cap)
        
        # Create the role
        role = Role(
            name=role_name,
            role_type=requirements.preferred_role_types[0] if requirements.preferred_role_types else RoleType.SPECIALIST,
            description=f"Dynamically created role for {role_name}",
            capabilities=capabilities,
            autonomy_level=requirements.autonomy_requirement,
            collaboration_preference=0.8 if requirements.collaboration_needs else 0.3,
            created_at=datetime.now().isoformat(),
            version="1.0",
            tags=["dynamic", "auto-generated"]
        )
        
        # Add template responsibilities if available
        if template:
            role.responsibilities = template.responsibilities_template.copy()
            role.constraints = template.constraints_template.copy()
        
        # Store the role in framework
        if self.framework:
            self.framework.roles[role.name] = role
        
        return role
    
    def _select_best_template(self, requirements: RoleRequirement) -> Optional[RoleTemplate]:
        """Select the best template for the requirements."""
        if not requirements.preferred_role_types:
            return None
        
        # Map role types to templates
        type_to_template = {
            RoleType.RESEARCHER: "researcher",
            RoleType.ANALYST: "analyst",
            RoleType.COORDINATOR: "coordinator"
        }
        
        for role_type in requirements.preferred_role_types:
            template_key = type_to_template.get(role_type)
            if template_key and template_key in self.role_templates:
                return self.role_templates[template_key]
        
        # Return first available template as fallback
        return next(iter(self.role_templates.values())) if self.role_templates else None
    
    def _generate_role_name(self, requirements: RoleRequirement, context: Dict[str, Any]) -> str:
        """Generate a descriptive name for the new role."""
        # Extract domain or specialization from context
        domain = context.get("domain", "General")
        
        # Use preferred role type
        if requirements.preferred_role_types:
            base_type = requirements.preferred_role_types[0].value.title()
        else:
            base_type = "Specialist"
        
        # Create specialized name
        if domain != "General":
            return f"{domain} {base_type}"
        else:
            # Use main capability if available
            if requirements.required_capabilities:
                main_cap = max(requirements.required_capabilities.items(), key=lambda x: x[1])
                cap_name = main_cap[0].replace("_", " ").title()
                return f"{cap_name} {base_type}"
        
        return f"Dynamic {base_type}"
    
    def adapt_role(self, role: Role, performance_data: Dict[str, Any]) -> Role:
        """Adapt an existing role based on performance data."""
        if not self.framework:
            return role
        
        # Store performance history
        if role.name not in self.role_performance_history:
            self.role_performance_history[role.name] = []
        
        self.role_performance_history[role.name].append({
            "timestamp": datetime.now().isoformat(),
            "performance": performance_data
        })
        
        # Analyze performance and adapt
        if len(self.role_performance_history[role.name]) >= 3:
            adaptations = self._analyze_performance_for_adaptation(role.name)
            self._apply_adaptations(role, adaptations)
        
        return role
    
    def _analyze_performance_for_adaptation(self, role_name: str) -> List[Dict[str, Any]]:
        """Analyze performance history to suggest adaptations."""
        history = self.role_performance_history[role_name]
        adaptations = []
        
        # Simple adaptation rules based on performance trends
        recent_performances = history[-3:]
        
        # Check for capability improvements needed
        for performance in recent_performances:
            if "failed_capabilities" in performance["performance"]:
                for cap in performance["performance"]["failed_capabilities"]:
                    adaptations.append({
                        "type": "increase_capability",
                        "capability": cap,
                        "increase": 0.1
                    })
        
        # Check for autonomy adjustments
        autonomy_scores = [p["performance"].get("autonomy_score", 0.5) for p in recent_performances]
        avg_autonomy = sum(autonomy_scores) / len(autonomy_scores)
        
        if avg_autonomy < 0.3:
            adaptations.append({
                "type": "decrease_autonomy",
                "amount": 0.1
            })
        elif avg_autonomy > 0.9:
            adaptations.append({
                "type": "increase_autonomy", 
                "amount": 0.1
            })
        
        return adaptations
    
    def _apply_adaptations(self, role: Role, adaptations: List[Dict[str, Any]]) -> None:
        """Apply adaptations to a role."""
        for adaptation in adaptations:
            if adaptation["type"] == "increase_capability":
                cap_name = adaptation["capability"]
                increase = adaptation["increase"]
                
                # Find and update capability
                for capability in role.capabilities:
                    if capability.name == cap_name:
                        capability.proficiency_level = min(1.0, capability.proficiency_level + increase)
                        break
                else:
                    # Add new capability if not found
                    new_capability = RoleCapability(
                        name=cap_name,
                        description=f"Adapted capability for {cap_name}",
                        proficiency_level=increase
                    )
                    role.capabilities.append(new_capability)
            
            elif adaptation["type"] == "decrease_autonomy":
                role.autonomy_level = max(0.0, role.autonomy_level - adaptation["amount"])
            
            elif adaptation["type"] == "increase_autonomy":
                role.autonomy_level = min(1.0, role.autonomy_level + adaptation["amount"])
        
        # Update role metadata
        role.updated_at = datetime.now().isoformat()
        role.version = str(float(role.version) + 0.1)
    
    def get_role_recommendations(self, agent_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get role recommendations for an agent given context."""
        if not self.framework:
            return []
        
        agent = self.framework.get_agent(agent_id)
        if not agent:
            return []
        
        # Analyze requirements
        requirements = self.analyze_requirements(context)
        
        # Score all available roles
        recommendations = []
        for role in self.framework.roles.values():
            suitability_score = self._calculate_role_suitability(role, requirements)
            
            # Check if agent can assume this role
            can_assume = agent._can_assume_role(role) if hasattr(agent, '_can_assume_role') else True
            
            recommendations.append({
                "role": role,
                "suitability_score": suitability_score,
                "can_assume": can_assume,
                "confidence": min(1.0, suitability_score * 1.2)
            })
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def create_role_from_template(self, template_name: str, customizations: Dict[str, Any] = None) -> Optional[Role]:
        """Create a role from a template with optional customizations."""
        template = self.role_templates.get(template_name)
        if not template:
            return None
        
        customizations = customizations or {}
        
        # Create base role from template
        role = Role(
            name=customizations.get("name", template.name),
            role_type=template.role_type,
            description=customizations.get("description", f"Role based on {template.name} template"),
            capabilities=template.base_capabilities.copy(),
            responsibilities=template.responsibilities_template.copy(),
            constraints=template.constraints_template.copy(),
            created_at=datetime.now().isoformat(),
            version="1.0",
            tags=["template-based", template_name]
        )
        
        # Apply customizations
        if "capabilities" in customizations:
            for cap_name, level in customizations["capabilities"].items():
                # Update existing capability or add new one
                for capability in role.capabilities:
                    if capability.name == cap_name:
                        capability.proficiency_level = level
                        break
                else:
                    new_cap = RoleCapability(
                        name=cap_name,
                        description=f"Custom capability: {cap_name}",
                        proficiency_level=level
                    )
                    role.capabilities.append(new_cap)
        
        # Store in framework
        if self.framework:
            self.framework.roles[role.name] = role
        
        return role
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get the current status of the role designer component."""
        return {
            "name": self.name,
            "version": self.version,
            "templates_available": len(self.role_templates),
            "roles_created": len(self.framework.roles) if self.framework else 0,
            "performance_tracked_roles": len(self.role_performance_history),
            "auto_create_enabled": self.auto_create_roles,
            "adaptation_threshold": self.role_adaptation_threshold
        }