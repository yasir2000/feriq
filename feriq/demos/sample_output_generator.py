"""
Sample Output Generator for Feriq Framework Components

Generates sample outputs for all framework components to demonstrate listing capabilities.
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any


class SampleOutputGenerator:
    """Generates sample outputs for all Feriq framework components."""
    
    def __init__(self, project_path: Path = None):
        self.project_path = project_path or Path.cwd()
        self.outputs_dir = self.project_path / 'outputs'
        self.timestamp = datetime.now().isoformat()
    
    def generate_all_samples(self):
        """Generate sample outputs for all components."""
        print("🎭 Generating sample outputs for Feriq framework components...")
        
        # Create outputs directory structure
        self.create_output_directories()
        
        # Generate samples for each component
        self.generate_role_designer_outputs()
        self.generate_task_designer_outputs()
        self.generate_plan_designer_outputs()
        self.generate_plan_observer_outputs()
        self.generate_agent_outputs()
        self.generate_workflow_orchestrator_outputs()
        self.generate_choreographer_outputs()
        self.generate_reasoner_outputs()
        self.generate_actions_log()
        
        print("✅ All sample outputs generated successfully!")
        print(f"📁 Outputs stored in: {self.outputs_dir}")
        print("💡 Use 'feriq list components' to see all available listings")
    
    def create_output_directories(self):
        """Create directory structure for outputs."""
        component_dirs = [
            'roles', 'tasks', 'plans', 'observations', 'agents',
            'workflows', 'interactions', 'reasoning', 'actions'
        ]
        
        for dir_name in component_dirs:
            (self.outputs_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def generate_role_designer_outputs(self):
        """Generate sample outputs for Dynamic Role Designer."""
        
        # Role definitions
        role_definitions = {
            'research_specialist': {
                'type': 'research',
                'status': 'active',
                'description': 'Conducts thorough research and analysis on complex topics',
                'capabilities': ['data_gathering', 'analysis', 'report_generation'],
                'requirements': ['domain_expertise', 'analytical_thinking'],
                'created_at': self.timestamp
            },
            'project_manager': {
                'type': 'coordination',
                'status': 'active', 
                'description': 'Manages project timelines, resources, and stakeholder communication',
                'capabilities': ['planning', 'coordination', 'communication'],
                'requirements': ['leadership', 'organization'],
                'created_at': self.timestamp
            },
            'technical_lead': {
                'type': 'technical',
                'status': 'active',
                'description': 'Provides technical guidance and architecture decisions',
                'capabilities': ['architecture_design', 'code_review', 'mentoring'],
                'requirements': ['technical_expertise', 'leadership'],
                'created_at': self.timestamp
            }
        }
        
        # Role assignments
        role_assignments = {
            'agent_001': 'research_specialist',
            'agent_002': 'project_manager',
            'agent_003': 'technical_lead',
            'agent_004': 'research_specialist'
        }
        
        # Role templates
        role_templates = {
            'data_analyst': {
                'description': 'Analyzes data and generates insights',
                'template_vars': ['domain', 'data_sources', 'analysis_types'],
                'capabilities_template': ['data_processing', 'visualization', 'reporting']
            },
            'quality_assurance': {
                'description': 'Ensures quality standards and testing',
                'template_vars': ['testing_types', 'quality_metrics'],
                'capabilities_template': ['testing', 'quality_control', 'documentation']
            }
        }
        
        # Dynamic roles
        dynamic_roles = [
            {
                'role_id': 'dynamic_001',
                'generated_for': 'AI model training project',
                'role_type': 'ml_engineer',
                'reasoning': 'Project requires ML expertise for model development',
                'created_at': self.timestamp
            }
        ]
        
        # Save outputs
        self.save_yaml(self.outputs_dir / 'roles' / 'role_definitions.yaml', role_definitions)
        self.save_json(self.outputs_dir / 'roles' / 'role_assignments.json', role_assignments)
        self.save_yaml(self.outputs_dir / 'roles' / 'role_templates.yaml', role_templates)
        self.save_json(self.outputs_dir / 'roles' / 'dynamic_roles.json', dynamic_roles)
    
    def generate_task_designer_outputs(self):
        """Generate sample outputs for Task Designer & Allocator."""
        
        # Task breakdowns
        task_breakdowns = {
            'task_001': {
                'name': 'Literature Review',
                'description': 'Comprehensive review of existing research in AI reasoning',
                'status': 'in_progress',
                'priority': 'high',
                'assigned_agent': 'agent_001',
                'estimated_duration': '5 days',
                'dependencies': [],
                'created_at': self.timestamp
            },
            'task_002': {
                'name': 'Data Collection',
                'description': 'Gather datasets for training reasoning models',
                'status': 'pending',
                'priority': 'medium',
                'assigned_agent': 'agent_004',
                'estimated_duration': '3 days',
                'dependencies': ['task_001'],
                'created_at': self.timestamp
            },
            'task_003': {
                'name': 'Architecture Design',
                'description': 'Design system architecture for reasoning framework',
                'status': 'completed',
                'priority': 'critical',
                'assigned_agent': 'agent_003',
                'estimated_duration': '7 days',
                'dependencies': [],
                'created_at': self.timestamp
            }
        }
        
        # Task assignments
        task_assignments = {
            'agent_001': ['task_001'],
            'agent_002': ['task_005', 'task_006'],
            'agent_003': ['task_003'],
            'agent_004': ['task_002', 'task_004']
        }
        
        # Allocation reports
        allocation_reports = {
            'allocation_summary': [
                'Optimal allocation achieved with 95% efficiency',
                'Agent workload balanced across skill sets',
                'Critical path identified with 3 bottleneck tasks'
            ],
            'optimization_metrics': {
                'efficiency_score': 0.95,
                'load_balance_score': 0.88,
                'skill_match_score': 0.92
            }
        }
        
        # Task dependencies
        task_dependencies = {
            'dependency_graph': {
                'task_001': [],
                'task_002': ['task_001'],
                'task_003': [],
                'task_004': ['task_002', 'task_003']
            },
            'critical_path': ['task_001', 'task_002', 'task_004']
        }
        
        # Save outputs
        self.save_json(self.outputs_dir / 'tasks' / 'task_breakdowns.json', task_breakdowns)
        self.save_json(self.outputs_dir / 'tasks' / 'task_assignments.json', task_assignments)
        self.save_yaml(self.outputs_dir / 'tasks' / 'allocation_reports.yaml', allocation_reports)
        self.save_json(self.outputs_dir / 'tasks' / 'task_dependencies.json', task_dependencies)
    
    def generate_plan_designer_outputs(self):
        """Generate sample outputs for Plan Designer."""
        
        # Execution plans
        execution_plans = {
            'plan_001': {
                'name': 'AI Reasoning System Development',
                'description': 'Complete development of advanced reasoning system',
                'status': 'active',
                'progress': 65,
                'start_date': self.timestamp,
                'estimated_completion': (datetime.now() + timedelta(days=30)).isoformat(),
                'tasks': ['task_001', 'task_002', 'task_003'],
                'milestones': [
                    {'name': 'Research Complete', 'date': '2025-10-20', 'status': 'completed'},
                    {'name': 'Architecture Approved', 'date': '2025-10-25', 'status': 'in_progress'},
                    {'name': 'Implementation Ready', 'date': '2025-11-01', 'status': 'pending'}
                ]
            },
            'plan_002': {
                'name': 'CLI Enhancement Project',
                'description': 'Enhance CLI with comprehensive listing capabilities',
                'status': 'active',
                'progress': 80,
                'start_date': self.timestamp,
                'estimated_completion': (datetime.now() + timedelta(days=7)).isoformat(),
                'tasks': ['task_004', 'task_005'],
                'milestones': [
                    {'name': 'Commands Implemented', 'date': '2025-10-16', 'status': 'completed'},
                    {'name': 'Testing Complete', 'date': '2025-10-18', 'status': 'pending'}
                ]
            }
        }
        
        # Resource allocations
        resource_allocations = {
            'human_resources': '4 developers, 1 project manager',
            'computational_resources': '2 GPU servers, 1 database server',
            'budget_allocation': '$150,000 development budget',
            'time_allocation': '90 days total project duration'
        }
        
        # Timeline schedules
        timeline_schedules = {
            'project_timeline': [
                {'phase': 'Research', 'start': '2025-10-01', 'end': '2025-10-15', 'status': 'completed'},
                {'phase': 'Design', 'start': '2025-10-10', 'end': '2025-10-25', 'status': 'in_progress'},
                {'phase': 'Implementation', 'start': '2025-10-20', 'end': '2025-11-15', 'status': 'pending'},
                {'phase': 'Testing', 'start': '2025-11-10', 'end': '2025-11-25', 'status': 'pending'}
            ]
        }
        
        # Plan templates
        plan_templates = {
            'software_development': {
                'phases': ['requirements', 'design', 'implementation', 'testing', 'deployment'],
                'default_duration': 90,
                'required_roles': ['technical_lead', 'developers', 'tester']
            },
            'research_project': {
                'phases': ['literature_review', 'methodology', 'data_collection', 'analysis', 'publication'],
                'default_duration': 180,
                'required_roles': ['research_specialist', 'data_analyst']
            }
        }
        
        # Save outputs
        self.save_json(self.outputs_dir / 'plans' / 'execution_plans.json', execution_plans)
        self.save_yaml(self.outputs_dir / 'plans' / 'resource_allocations.yaml', resource_allocations)
        self.save_json(self.outputs_dir / 'plans' / 'timeline_schedules.json', timeline_schedules)
        self.save_yaml(self.outputs_dir / 'plans' / 'plan_templates.yaml', plan_templates)
    
    def generate_plan_observer_outputs(self):
        """Generate sample outputs for Plan Observer."""
        
        # Execution logs
        execution_logs = [
            {
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'level': 'info',
                'component': 'task_allocator',
                'message': 'Task task_001 assigned to agent_001',
                'context': {'task_id': 'task_001', 'agent_id': 'agent_001'}
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'level': 'warning',
                'component': 'plan_executor',
                'message': 'Task task_002 delayed due to dependency wait',
                'context': {'task_id': 'task_002', 'delay_reason': 'dependency_wait'}
            },
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'success',
                'component': 'workflow_orchestrator',
                'message': 'Milestone "Research Complete" achieved',
                'context': {'milestone': 'Research Complete', 'plan_id': 'plan_001'}
            }
        ]
        
        # Performance metrics
        performance_metrics = {
            'task_completion_rate': '78%',
            'average_task_duration': '4.2 days',
            'agent_utilization': '85%',
            'plan_adherence_score': '0.92',
            'resource_efficiency': '88%',
            'milestone_hit_rate': '90%'
        }
        
        # Status reports
        status_reports = {
            'overall_status': 'On Track',
            'active_plans': 2,
            'completed_tasks': 15,
            'pending_tasks': 8,
            'issues_identified': 3,
            'risks_mitigated': 5,
            'last_updated': self.timestamp
        }
        
        # Alerts
        alerts = [
            {
                'id': 'alert_001',
                'severity': 'warning',
                'message': 'Task task_002 approaching deadline',
                'status': 'active',
                'created_at': self.timestamp,
                'context': {'task_id': 'task_002', 'deadline': '2025-10-20'}
            },
            {
                'id': 'alert_002',
                'severity': 'info',
                'message': 'New agent agent_005 added to project',
                'status': 'resolved',
                'created_at': (datetime.now() - timedelta(hours=6)).isoformat(),
                'context': {'agent_id': 'agent_005'}
            }
        ]
        
        # Save outputs
        self.save_json(self.outputs_dir / 'observations' / 'execution_logs.json', execution_logs)
        self.save_json(self.outputs_dir / 'observations' / 'performance_metrics.json', performance_metrics)
        self.save_yaml(self.outputs_dir / 'observations' / 'status_reports.yaml', status_reports)
        self.save_json(self.outputs_dir / 'observations' / 'alerts.json', alerts)
    
    def generate_agent_outputs(self):
        """Generate sample outputs for Goal-Oriented Agents."""
        
        # Agent configurations
        agent_configs = {
            'agent_001': {
                'name': 'Research Agent Alpha',
                'role': 'research_specialist',
                'status': 'active',
                'goals': ['goal_001', 'goal_002'],
                'capabilities': ['research', 'analysis', 'reporting'],
                'last_active': (datetime.now() - timedelta(minutes=30)).isoformat(),
                'performance_score': 0.92
            },
            'agent_002': {
                'name': 'Project Coordinator',
                'role': 'project_manager',
                'status': 'busy',
                'goals': ['goal_003'],
                'capabilities': ['planning', 'coordination', 'communication'],
                'last_active': datetime.now().isoformat(),
                'performance_score': 0.88
            },
            'agent_003': {
                'name': 'Tech Lead Beta',
                'role': 'technical_lead',
                'status': 'idle',
                'goals': ['goal_004'],
                'capabilities': ['architecture', 'coding', 'review'],
                'last_active': (datetime.now() - timedelta(hours=2)).isoformat(),
                'performance_score': 0.95
            }
        }
        
        # Goal progress
        goal_progress = {
            'agent_001': [
                {'goal_id': 'goal_001', 'progress': 85, 'status': 'in_progress'},
                {'goal_id': 'goal_002', 'progress': 45, 'status': 'in_progress'}
            ],
            'agent_002': [
                {'goal_id': 'goal_003', 'progress': 70, 'status': 'in_progress'}
            ],
            'agent_003': [
                {'goal_id': 'goal_004', 'progress': 100, 'status': 'completed'}
            ]
        }
        
        # Learning logs
        learning_logs = [
            {
                'agent_id': 'agent_001',
                'timestamp': self.timestamp,
                'learning_type': 'performance_improvement',
                'description': 'Improved research efficiency by optimizing query strategies',
                'impact_score': 0.15
            },
            {
                'agent_id': 'agent_002',
                'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
                'learning_type': 'skill_acquisition',
                'description': 'Learned new stakeholder communication patterns',
                'impact_score': 0.08
            }
        ]
        
        # Adaptations
        adaptations = {
            'recent': [
                'agent_001 adapted search strategy for better research results',
                'agent_002 modified communication frequency based on stakeholder feedback',
                'agent_003 updated code review criteria for better quality'
            ],
            'adaptation_history': [
                {'agent': 'agent_001', 'adaptation': 'search_optimization', 'date': self.timestamp},
                {'agent': 'agent_002', 'adaptation': 'communication_tuning', 'date': self.timestamp}
            ]
        }
        
        # Save outputs
        self.save_yaml(self.outputs_dir / 'agents' / 'agent_configs.yaml', agent_configs)
        self.save_json(self.outputs_dir / 'agents' / 'goal_progress.json', goal_progress)
        self.save_json(self.outputs_dir / 'agents' / 'learning_logs.json', learning_logs)
        self.save_yaml(self.outputs_dir / 'agents' / 'adaptations.yaml', adaptations)
    
    def generate_workflow_orchestrator_outputs(self):
        """Generate sample outputs for Workflow Orchestrator."""
        
        # Workflow definitions
        workflow_definitions = {
            'workflow_001': {
                'name': 'Research and Development Pipeline',
                'description': 'End-to-end R&D workflow with quality gates',
                'status': 'running',
                'progress': 60,
                'agents': ['agent_001', 'agent_002', 'agent_003'],
                'stages': ['research', 'design', 'implementation', 'testing'],
                'current_stage': 'implementation'
            },
            'workflow_002': {
                'name': 'Documentation Generation',
                'description': 'Automated documentation creation and review',
                'status': 'completed',
                'progress': 100,
                'agents': ['agent_001', 'agent_004'],
                'stages': ['content_creation', 'review', 'publication'],
                'current_stage': 'publication'
            }
        }
        
        # Execution results
        execution_results = {
            'workflow_001': {
                'execution_id': 'exec_001',
                'start_time': (datetime.now() - timedelta(days=10)).isoformat(),
                'status': 'running',
                'completed_stages': ['research', 'design'],
                'current_stage': 'implementation',
                'metrics': {
                    'tasks_completed': 12,
                    'tasks_remaining': 8,
                    'efficiency_score': 0.87
                }
            },
            'workflow_002': {
                'execution_id': 'exec_002',
                'start_time': (datetime.now() - timedelta(days=5)).isoformat(),
                'end_time': datetime.now().isoformat(),
                'status': 'completed',
                'completed_stages': ['content_creation', 'review', 'publication'],
                'metrics': {
                    'tasks_completed': 6,
                    'tasks_remaining': 0,
                    'efficiency_score': 0.95
                }
            }
        }
        
        # Resource usage
        resource_usage = {
            'cpu_utilization': '68%',
            'memory_usage': '4.2GB',
            'network_bandwidth': '125 Mbps',
            'storage_usage': '2.8TB',
            'agent_hours': '240 hours total'
        }
        
        # Coordination logs
        coordination_logs = [
            {
                'timestamp': self.timestamp,
                'event': 'workflow_stage_transition',
                'workflow_id': 'workflow_001',
                'from_stage': 'design',
                'to_stage': 'implementation',
                'triggered_by': 'milestone_completion'
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=4)).isoformat(),
                'event': 'resource_reallocation',
                'workflow_id': 'workflow_001',
                'resource': 'agent_003',
                'action': 'assigned_to_critical_task'
            }
        ]
        
        # Save outputs
        self.save_yaml(self.outputs_dir / 'workflows' / 'workflow_definitions.yaml', workflow_definitions)
        self.save_json(self.outputs_dir / 'workflows' / 'execution_results.json', execution_results)
        self.save_json(self.outputs_dir / 'workflows' / 'resource_usage.json', resource_usage)
        self.save_json(self.outputs_dir / 'workflows' / 'coordination_logs.json', coordination_logs)
    
    def generate_choreographer_outputs(self):
        """Generate sample outputs for Choreographer."""
        
        # Interaction patterns
        interaction_patterns = {
            'research_collaboration': {
                'description': 'Pattern for collaborative research activities',
                'participants': ['research_specialist', 'data_analyst'],
                'communication_frequency': 'high',
                'coordination_style': 'synchronous'
            },
            'project_reporting': {
                'description': 'Pattern for project status reporting',
                'participants': ['project_manager', 'stakeholders'],
                'communication_frequency': 'weekly',
                'coordination_style': 'asynchronous'
            },
            'code_review': {
                'description': 'Pattern for technical code review process',
                'participants': ['technical_lead', 'developers'],
                'communication_frequency': 'per_commit',
                'coordination_style': 'asynchronous'
            }
        }
        
        # Communication logs
        communication_logs = [
            {
                'timestamp': datetime.now().isoformat(),
                'sender': 'agent_001',
                'receiver': 'agent_002',
                'type': 'progress_update',
                'pattern': 'research_collaboration',
                'message_id': 'msg_001'
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
                'sender': 'agent_002',
                'receiver': 'agent_003',
                'type': 'task_assignment',
                'pattern': 'project_coordination',
                'message_id': 'msg_002'
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'sender': 'agent_003',
                'receiver': 'agent_001',
                'type': 'review_request',
                'pattern': 'code_review',
                'message_id': 'msg_003'
            }
        ]
        
        # Coordination matrices
        coordination_matrices = {
            'agent_interaction_matrix': {
                'agent_001': {'agent_002': 15, 'agent_003': 8, 'agent_004': 5},
                'agent_002': {'agent_001': 15, 'agent_003': 12, 'agent_004': 3},
                'agent_003': {'agent_001': 8, 'agent_002': 12, 'agent_004': 7}
            },
            'communication_efficiency': {
                'overall_score': 0.89,
                'pattern_effectiveness': {
                    'research_collaboration': 0.92,
                    'project_reporting': 0.85,
                    'code_review': 0.91
                }
            }
        }
        
        # Save outputs
        self.save_yaml(self.outputs_dir / 'interactions' / 'interaction_patterns.yaml', interaction_patterns)
        self.save_json(self.outputs_dir / 'interactions' / 'communication_logs.json', communication_logs)
        self.save_json(self.outputs_dir / 'interactions' / 'coordination_matrices.json', coordination_matrices)
    
    def generate_reasoner_outputs(self):
        """Generate sample outputs for Reasoner."""
        
        # Reasoning results
        reasoning_results = [
            {
                'reasoning_id': 'reason_001',
                'reasoning_type': 'causal',
                'query': 'Why did task_002 experience delays?',
                'conclusion': 'Task delay caused by dependency wait on task_001 completion',
                'confidence': 0.89,
                'evidence': ['task_001 completion delayed by 2 days', 'task_002 has strict dependency'],
                'timestamp': self.timestamp
            },
            {
                'reasoning_id': 'reason_002',
                'reasoning_type': 'probabilistic',
                'query': 'What is the probability of meeting project deadline?',
                'conclusion': 'High probability (0.82) of meeting deadline with current resource allocation',
                'confidence': 0.82,
                'evidence': ['current progress rate', 'remaining task complexity', 'resource availability'],
                'timestamp': self.timestamp
            },
            {
                'reasoning_id': 'reason_003',
                'reasoning_type': 'inductive',
                'query': 'What patterns emerge from agent performance data?',
                'conclusion': 'Agents perform 15% better on tasks matching their specialized skills',
                'confidence': 0.91,
                'evidence': ['historical performance data', 'skill-task matching analysis'],
                'timestamp': self.timestamp
            }
        ]
        
        # Decision trees
        decision_trees = {
            'resource_allocation_tree': {
                'root': 'Is agent available?',
                'branches': {
                    'yes': {
                        'question': 'Does agent have required skills?',
                        'branches': {
                            'yes': 'Assign task to agent',
                            'no': 'Provide training or find alternative'
                        }
                    },
                    'no': 'Queue task or reallocate resources'
                }
            },
            'risk_assessment_tree': {
                'root': 'Is risk probability > 0.7?',
                'branches': {
                    'yes': 'Implement immediate mitigation',
                    'no': {
                        'question': 'Is impact severity high?',
                        'branches': {
                            'yes': 'Monitor closely and prepare mitigation',
                            'no': 'Add to risk register for periodic review'
                        }
                    }
                }
            }
        }
        
        # Strategic recommendations
        strategic_recommendations = [
            {
                'recommendation': 'Increase parallel task execution to accelerate project timeline',
                'priority': 'high',
                'reasoning_type': 'temporal',
                'expected_impact': '20% timeline reduction',
                'confidence': 0.85
            },
            {
                'recommendation': 'Implement cross-training program to improve agent versatility',
                'priority': 'medium',
                'reasoning_type': 'inductive',
                'expected_impact': 'Improved resource flexibility',
                'confidence': 0.78
            },
            {
                'recommendation': 'Establish automated quality gates to prevent defect propagation',
                'priority': 'high',
                'reasoning_type': 'causal',
                'expected_impact': '40% reduction in downstream issues',
                'confidence': 0.92
            }
        ]
        
        # Problem solutions
        problem_solutions = [
            {
                'problem_id': 'prob_001',
                'description': 'Agent workload imbalance causing bottlenecks',
                'solution': 'Implement dynamic task reallocation based on real-time capacity',
                'reasoning_approach': 'optimization + collaborative',
                'implementation_steps': [
                    'Monitor agent capacity in real-time',
                    'Implement task reallocation algorithm',
                    'Test with pilot workload'
                ]
            },
            {
                'problem_id': 'prob_002',
                'description': 'Communication gaps between distributed team members',
                'solution': 'Establish structured communication patterns with regular checkpoints',
                'reasoning_approach': 'collaborative + temporal',
                'implementation_steps': [
                    'Define communication templates',
                    'Schedule regular sync meetings',
                    'Implement communication tracking'
                ]
            }
        ]
        
        # Save outputs
        self.save_json(self.outputs_dir / 'reasoning' / 'reasoning_results.json', reasoning_results)
        self.save_yaml(self.outputs_dir / 'reasoning' / 'decision_trees.yaml', decision_trees)
        self.save_json(self.outputs_dir / 'reasoning' / 'strategic_recommendations.json', strategic_recommendations)
        self.save_json(self.outputs_dir / 'reasoning' / 'problem_solutions.json', problem_solutions)
    
    def generate_actions_log(self):
        """Generate sample actions log across all components."""
        
        # Action history
        action_history = [
            {
                'timestamp': (datetime.now() - timedelta(hours=5)).isoformat(),
                'component': 'role_designer',
                'action': 'create_role',
                'status': 'success',
                'details': {'role_name': 'research_specialist'}
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=4)).isoformat(),
                'component': 'task_designer',
                'action': 'break_down_goal',
                'status': 'success',
                'details': {'goal_id': 'goal_001', 'tasks_created': 5}
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=3)).isoformat(),
                'component': 'plan_designer',
                'action': 'create_execution_plan',
                'status': 'success',
                'details': {'plan_id': 'plan_001', 'tasks_included': 15}
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'component': 'plan_observer',
                'action': 'monitor_execution',
                'status': 'success',
                'details': {'plans_monitored': 2, 'alerts_generated': 1}
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'component': 'workflow_orchestrator',
                'action': 'coordinate_execution',
                'status': 'success',
                'details': {'workflows_active': 2, 'agents_coordinated': 4}
            },
            {
                'timestamp': datetime.now().isoformat(),
                'component': 'reasoner',
                'action': 'generate_recommendations',
                'status': 'success',
                'details': {'reasoning_type': 'strategic', 'recommendations': 3}
            }
        ]
        
        # Component actions summary
        component_actions = {
            'role_designer': 8,
            'task_designer': 12,
            'plan_designer': 6,
            'plan_observer': 25,
            'workflow_orchestrator': 15,
            'choreographer': 18,
            'reasoner': 10
        }
        
        # System events
        system_events = [
            {
                'timestamp': self.timestamp,
                'event_type': 'component_integration',
                'description': 'All framework components successfully integrated',
                'severity': 'info'
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(),
                'event_type': 'performance_optimization',
                'description': 'Reasoning engine performance improved by 25%',
                'severity': 'info'
            }
        ]
        
        # Save outputs
        self.save_json(self.outputs_dir / 'actions' / 'action_history.json', action_history)
        self.save_json(self.outputs_dir / 'actions' / 'component_actions.json', component_actions)
        self.save_json(self.outputs_dir / 'actions' / 'system_events.json', system_events)
    
    def save_json(self, filepath: Path, data: Any):
        """Save data as JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_yaml(self, filepath: Path, data: Any):
        """Save data as YAML file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def generate_sample_outputs():
    """Generate sample outputs for demonstration."""
    generator = SampleOutputGenerator()
    generator.generate_all_samples()


if __name__ == "__main__":
    generate_sample_outputs()