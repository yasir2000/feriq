"""Base Role class for the Feriq framework."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class RoleType(str, Enum):
    """Enumeration of available role types."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    PLANNER = "planner"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    REVIEWER = "reviewer"
    FACILITATOR = "facilitator"


class RoleCapability(BaseModel):
    """Represents a capability that a role can have."""
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of what this capability enables")
    proficiency_level: float = Field(default=0.5, ge=0.0, le=1.0, 
                                   description="Proficiency level from 0.0 to 1.0")
    prerequisites: List[str] = Field(default_factory=list, 
                                   description="Required capabilities or conditions")


class Role(BaseModel):
    """
    Represents a role that can be assigned to agents in the Feriq framework.
    
    Roles define the responsibilities, capabilities, and constraints
    that guide agent behavior and task assignment.
    """
    
    name: str = Field(..., description="Unique name for this role")
    role_type: RoleType = Field(..., description="Type categorization of this role")
    description: str = Field(..., description="Detailed description of the role")
    
    # Core attributes
    responsibilities: List[str] = Field(default_factory=list,
                                      description="Primary responsibilities of this role")
    capabilities: List[RoleCapability] = Field(default_factory=list,
                                             description="Capabilities this role possesses")
    constraints: List[str] = Field(default_factory=list,
                                 description="Limitations or constraints on this role")
    
    # Behavioral attributes
    autonomy_level: float = Field(default=0.5, ge=0.0, le=1.0,
                                description="Level of autonomous decision-making")
    collaboration_preference: float = Field(default=0.5, ge=0.0, le=1.0,
                                          description="Preference for collaborative work")
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0,
                                description="Tolerance for uncertain outcomes")
    
    # Dynamic attributes
    current_workload: float = Field(default=0.0, ge=0.0,
                                  description="Current workload as a ratio")
    performance_metrics: Dict[str, float] = Field(default_factory=dict,
                                                description="Performance tracking metrics")
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0,
                               description="Rate of capability improvement")
    
    # Metadata
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")
    version: str = Field(default="1.0", description="Role version")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        use_enum_values = True
    
    def add_capability(self, capability: RoleCapability) -> None:
        """Add a new capability to this role."""
        # Check if capability already exists
        for existing in self.capabilities:
            if existing.name == capability.name:
                # Update existing capability
                existing.proficiency_level = max(existing.proficiency_level, 
                                               capability.proficiency_level)
                return
        
        # Add new capability
        self.capabilities.append(capability)
    
    def get_capability_level(self, capability_name: str) -> float:
        """Get the proficiency level for a specific capability."""
        for capability in self.capabilities:
            if capability.name == capability_name:
                return capability.proficiency_level
        return 0.0
    
    def can_handle_task(self, task_requirements: List[str]) -> bool:
        """Check if this role can handle tasks with given requirements."""
        role_capabilities = {cap.name for cap in self.capabilities}
        return all(req in role_capabilities for req in task_requirements)
    
    def calculate_suitability_score(self, task_requirements: Dict[str, float]) -> float:
        """
        Calculate how suitable this role is for a task.
        
        Args:
            task_requirements: Dict mapping capability names to required proficiency levels
            
        Returns:
            Suitability score between 0.0 and 1.0
        """
        if not task_requirements:
            return 0.5  # Neutral score for tasks with no specific requirements
        
        total_score = 0.0
        total_weight = 0.0
        
        for req_capability, req_level in task_requirements.items():
            role_level = self.get_capability_level(req_capability)
            
            # Calculate score for this requirement
            if role_level >= req_level:
                # Role exceeds requirement - positive score
                score = 1.0
            else:
                # Role doesn't meet requirement - proportional penalty
                score = role_level / req_level if req_level > 0 else 0.0
            
            total_score += score * req_level  # Weight by requirement importance
            total_weight += req_level
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def update_workload(self, workload_change: float) -> None:
        """Update the current workload of this role."""
        self.current_workload = max(0.0, self.current_workload + workload_change)
    
    def is_available(self, max_workload: float = 1.0) -> bool:
        """Check if this role is available for new tasks."""
        return self.current_workload < max_workload
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert role to dictionary representation."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Role":
        """Create role from dictionary representation."""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of the role."""
        return f"Role(name='{self.name}', type='{self.role_type}', workload={self.current_workload:.2f})"
    
    def __repr__(self) -> str:
        """Developer representation of the role."""
        return self.__str__()