"""
Plan Observer Component

This module implements the monitoring and observation system for tracking plan execution,
progress measurement, and adaptive plan modification based on real-time feedback.
"""

from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import asyncio
from collections import deque, defaultdict
from pydantic import BaseModel, Field
import threading
import time
from statistics import mean, stdev

from ..core.goal import Goal, GoalStatus
from ..core.task import FeriqTask, TaskStatus
from ..core.plan import Plan, PlanStatus, Milestone
from ..core.agent import FeriqAgent
from ..utils.logger import FeriqLogger
from ..utils.config import Config


class ObservationType(str, Enum):
    """Types of observations that can be made."""
    TASK_PROGRESS = "task_progress"
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"
    MILESTONE_COMPLETION = "milestone_completion"
    AGENT_ACTIVITY = "agent_activity"
    PLAN_DEVIATION = "plan_deviation"
    BOTTLENECK_DETECTION = "bottleneck_detection"


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Observation:
    """Represents an observation made during plan execution."""
    observation_id: str
    observation_type: ObservationType
    timestamp: datetime
    plan_id: str
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.LOW
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "timestamp": self.timestamp.isoformat(),
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "data": self.data,
            "severity": self.severity,
            "message": self.message
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for plan execution."""
    plan_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    in_progress_tasks: int
    average_task_duration: float
    resource_utilization: Dict[str, float]
    bottlenecks: List[str]
    efficiency_score: float
    estimated_completion_time: Optional[datetime] = None
    actual_completion_time: Optional[datetime] = None
    
    @property
    def completion_rate(self) -> float:
        return self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
    
    @property
    def failure_rate(self) -> float:
        return self.failed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0


class AlertSystem:
    """System for generating and managing alerts based on observations."""
    
    def __init__(self, logger: FeriqLogger):
        self.logger = logger
        self.alert_rules: Dict[ObservationType, List[Callable]] = defaultdict(list)
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        # Task progress alerts
        self.add_alert_rule(
            ObservationType.TASK_PROGRESS,
            lambda obs: AlertSeverity.HIGH if obs.data.get("overdue_hours", 0) > 24 else AlertSeverity.LOW
        )
        
        # Resource usage alerts
        self.add_alert_rule(
            ObservationType.RESOURCE_USAGE,
            lambda obs: AlertSeverity.CRITICAL if obs.data.get("utilization", 0) > 0.9 else AlertSeverity.LOW
        )
        
        # Performance alerts
        self.add_alert_rule(
            ObservationType.PERFORMANCE_METRIC,
            lambda obs: AlertSeverity.HIGH if obs.data.get("efficiency", 1.0) < 0.5 else AlertSeverity.LOW
        )
    
    def add_alert_rule(self, observation_type: ObservationType, rule_func: Callable[[Observation], AlertSeverity]):
        """Add a custom alert rule."""
        self.alert_rules[observation_type].append(rule_func)
    
    def add_alert_handler(self, severity: AlertSeverity, handler_func: Callable[[Observation], None]):
        """Add a custom alert handler."""
        self.alert_handlers[severity].append(handler_func)
    
    def process_observation(self, observation: Observation) -> AlertSeverity:
        """Process observation and determine alert severity."""
        max_severity = AlertSeverity.LOW
        
        for rule_func in self.alert_rules[observation.observation_type]:
            try:
                severity = rule_func(observation)
                if self._severity_value(severity) > self._severity_value(max_severity):
                    max_severity = severity
            except Exception as e:
                self.logger.error(f"Error in alert rule: {e}")
        
        observation.severity = max_severity
        
        # Trigger alert handlers
        for handler_func in self.alert_handlers[max_severity]:
            try:
                handler_func(observation)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        return max_severity
    
    def _severity_value(self, severity: AlertSeverity) -> int:
        """Convert severity to numeric value for comparison."""
        return {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4
        }[severity]


class PlanMonitor:
    """Real-time monitoring system for plan execution."""
    
    def __init__(self, plan: Plan, logger: FeriqLogger):
        self.plan = plan
        self.logger = logger
        self.observations: deque = deque(maxlen=1000)
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = datetime.now()
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 30  # seconds
        
        # Tracking state
        self.task_start_times: Dict[str, datetime] = {}
        self.task_end_times: Dict[str, datetime] = {}
        self.resource_usage_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Plan monitoring started", plan_id=self.plan.plan_id)
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Plan monitoring stopped", plan_id=self.plan.plan_id)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._collect_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        current_metrics = self._calculate_current_metrics()
        self.metrics_history.append(current_metrics)
        
        # Create performance observation
        observation = Observation(
            observation_id=str(uuid.uuid4()),
            observation_type=ObservationType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            plan_id=self.plan.plan_id,
            data={
                "completion_rate": current_metrics.completion_rate,
                "efficiency": current_metrics.efficiency_score,
                "resource_utilization": current_metrics.resource_utilization
            },
            message=f"Performance metrics collected: {current_metrics.completion_rate:.2%} complete"
        )
        
        self.observations.append(observation)
    
    def _calculate_current_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        completed_tasks = sum(1 for task in self.plan.tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in self.plan.tasks if task.status == TaskStatus.FAILED)
        in_progress_tasks = sum(1 for task in self.plan.tasks if task.status == TaskStatus.IN_PROGRESS)
        
        # Calculate average task duration for completed tasks
        durations = []
        for task in self.plan.tasks:
            if task.status == TaskStatus.COMPLETED and task.task_id in self.task_end_times:
                start_time = self.task_start_times.get(task.task_id)
                end_time = self.task_end_times[task.task_id]
                if start_time:
                    duration = (end_time - start_time).total_seconds()
                    durations.append(duration)
        
        avg_duration = mean(durations) if durations else 0.0
        
        # Calculate efficiency score (completed vs estimated time)
        efficiency_score = 1.0
        if durations and len(durations) > 0:
            estimated_durations = [
                task.estimated_duration.total_seconds() 
                for task in self.plan.tasks 
                if task.status == TaskStatus.COMPLETED and task.estimated_duration
            ]
            if estimated_durations:
                avg_estimated = mean(estimated_durations)
                efficiency_score = min(avg_estimated / avg_duration, 1.0) if avg_duration > 0 else 1.0
        
        return PerformanceMetrics(
            plan_id=self.plan.plan_id,
            total_tasks=len(self.plan.tasks),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            in_progress_tasks=in_progress_tasks,
            average_task_duration=avg_duration,
            resource_utilization={},  # Would be filled by actual resource monitoring
            bottlenecks=[],  # Would be detected by bottleneck analysis
            efficiency_score=efficiency_score
        )
    
    def record_task_start(self, task_id: str):
        """Record when a task starts."""
        self.task_start_times[task_id] = datetime.now()
        
        observation = Observation(
            observation_id=str(uuid.uuid4()),
            observation_type=ObservationType.TASK_PROGRESS,
            timestamp=datetime.now(),
            plan_id=self.plan.plan_id,
            task_id=task_id,
            message=f"Task {task_id} started"
        )
        self.observations.append(observation)
    
    def record_task_completion(self, task_id: str):
        """Record when a task completes."""
        self.task_end_times[task_id] = datetime.now()
        
        observation = Observation(
            observation_id=str(uuid.uuid4()),
            observation_type=ObservationType.TASK_PROGRESS,
            timestamp=datetime.now(),
            plan_id=self.plan.plan_id,
            task_id=task_id,
            message=f"Task {task_id} completed"
        )
        self.observations.append(observation)
    
    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the latest performance metrics."""
        return self.metrics_history[-1] if self.metrics_history else None


class PlanObserver:
    """
    Plan Observer component that monitors plan execution and provides real-time feedback.
    
    This component provides:
    - Real-time monitoring of plan execution
    - Performance metrics and progress tracking
    - Alert system for deviations and issues
    - Adaptive recommendations for plan optimization
    - Historical analysis and reporting
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = FeriqLogger("PlanObserver", self.config)
        self.alert_system = AlertSystem(self.logger)
        
        # Active monitors
        self.active_monitors: Dict[str, PlanMonitor] = {}
        
        # Observation storage
        self.all_observations: deque = deque(maxlen=5000)
        self.plan_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        self.logger.info("PlanObserver initialized successfully")
    
    def start_observing_plan(self, plan: Plan) -> str:
        """
        Start observing a plan execution.
        
        Args:
            plan: The plan to observe
            
        Returns:
            Monitor ID for the observation session
        """
        monitor_id = str(uuid.uuid4())
        monitor = PlanMonitor(plan, self.logger)
        
        self.active_monitors[monitor_id] = monitor
        monitor.start_monitoring()
        
        self.logger.info(
            "Started observing plan",
            plan_id=plan.plan_id,
            monitor_id=monitor_id
        )
        
        return monitor_id
    
    def stop_observing_plan(self, monitor_id: str):
        """Stop observing a plan."""
        if monitor_id in self.active_monitors:
            monitor = self.active_monitors[monitor_id]
            monitor.stop_monitoring()
            
            # Store final metrics
            final_metrics = monitor.get_latest_metrics()
            if final_metrics:
                self.plan_metrics[monitor.plan.plan_id].append(final_metrics)
            
            # Store observations
            self.all_observations.extend(monitor.observations)
            
            del self.active_monitors[monitor_id]
            
            self.logger.info(
                "Stopped observing plan",
                plan_id=monitor.plan.plan_id,
                monitor_id=monitor_id
            )
    
    def record_observation(self, observation: Observation):
        """Record a new observation."""
        # Process through alert system
        severity = self.alert_system.process_observation(observation)
        
        # Store observation
        self.all_observations.append(observation)
        
        # Trigger event handlers
        self._trigger_event_handlers(observation)
        
        self.logger.info(
            "Observation recorded",
            observation_id=observation.observation_id,
            observation_type=observation.observation_type,
            severity=severity
        )
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a plan."""
        # Find active monitor
        monitor = None
        for mon in self.active_monitors.values():
            if mon.plan.plan_id == plan_id:
                monitor = mon
                break
        
        if not monitor:
            return {"error": "Plan not being monitored"}
        
        current_metrics = monitor.get_latest_metrics()
        recent_observations = [
            obs for obs in monitor.observations 
            if obs.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        return {
            "plan_id": plan_id,
            "status": monitor.plan.status,
            "is_monitoring": monitor.is_monitoring,
            "start_time": monitor.start_time.isoformat(),
            "current_metrics": current_metrics.__dict__ if current_metrics else None,
            "recent_observations_count": len(recent_observations),
            "high_severity_alerts": sum(
                1 for obs in recent_observations 
                if obs.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            )
        }
    
    def analyze_plan_performance(self, plan_id: str) -> Dict[str, Any]:
        """Analyze performance trends for a plan."""
        metrics_list = self.plan_metrics.get(plan_id, [])
        
        if not metrics_list:
            return {"error": "No metrics available for plan"}
        
        # Calculate trends
        completion_rates = [m.completion_rate for m in metrics_list]
        efficiency_scores = [m.efficiency_score for m in metrics_list]
        
        analysis = {
            "plan_id": plan_id,
            "metrics_count": len(metrics_list),
            "completion_trend": {
                "current": completion_rates[-1] if completion_rates else 0,
                "average": mean(completion_rates) if completion_rates else 0,
                "variance": stdev(completion_rates) if len(completion_rates) > 1 else 0
            },
            "efficiency_trend": {
                "current": efficiency_scores[-1] if efficiency_scores else 0,
                "average": mean(efficiency_scores) if efficiency_scores else 0,
                "variance": stdev(efficiency_scores) if len(efficiency_scores) > 1 else 0
            },
            "recommendations": self._generate_recommendations(metrics_list)
        }
        
        return analysis
    
    def _generate_recommendations(self, metrics_list: List[PerformanceMetrics]) -> List[str]:
        """Generate recommendations based on performance metrics."""
        recommendations = []
        
        if not metrics_list:
            return recommendations
        
        latest = metrics_list[-1]
        
        # Completion rate recommendations
        if latest.completion_rate < 0.5:
            recommendations.append("Consider reviewing task allocation and dependencies")
        
        # Efficiency recommendations
        if latest.efficiency_score < 0.7:
            recommendations.append("Review task complexity estimates and resource allocation")
        
        # Failure rate recommendations
        if latest.failure_rate > 0.1:
            recommendations.append("Investigate task failures and improve error handling")
        
        # Resource utilization recommendations
        for resource, utilization in latest.resource_utilization.items():
            if utilization > 0.9:
                recommendations.append(f"Consider scaling up {resource} resources")
            elif utilization < 0.3:
                recommendations.append(f"Consider reducing {resource} allocation")
        
        return recommendations
    
    def add_event_handler(self, event_type: str, handler: Callable[[Observation], None]):
        """Add an event handler for specific observation types."""
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event_handlers(self, observation: Observation):
        """Trigger event handlers for an observation."""
        event_type = observation.observation_type.value
        
        for handler in self.event_handlers[event_type]:
            try:
                handler(observation)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    def get_observation_history(
        self,
        plan_id: Optional[str] = None,
        observation_types: Optional[List[ObservationType]] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get filtered observation history."""
        observations = list(self.all_observations)
        
        # Apply filters
        if plan_id:
            observations = [obs for obs in observations if obs.plan_id == plan_id]
        
        if observation_types:
            observations = [obs for obs in observations if obs.observation_type in observation_types]
        
        if since:
            observations = [obs for obs in observations if obs.timestamp >= since]
        
        # Sort by timestamp (newest first) and limit
        observations.sort(key=lambda x: x.timestamp, reverse=True)
        observations = observations[:limit]
        
        return [obs.to_dict() for obs in observations]
    
    def export_metrics(self, plan_id: str, format: str = "json") -> str:
        """Export metrics for a plan in specified format."""
        metrics_list = self.plan_metrics.get(plan_id, [])
        
        if format == "json":
            import json
            return json.dumps([m.__dict__ for m in metrics_list], default=str, indent=2)
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if metrics_list:
                writer = csv.DictWriter(output, fieldnames=metrics_list[0].__dict__.keys())
                writer.writeheader()
                for metrics in metrics_list:
                    writer.writerow(metrics.__dict__)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_observer_statistics(self) -> Dict[str, Any]:
        """Get statistics about the observer's activity."""
        return {
            "active_monitors": len(self.active_monitors),
            "total_observations": len(self.all_observations),
            "monitored_plans": len(self.plan_metrics),
            "observation_types": {
                obs_type.value: sum(1 for obs in self.all_observations if obs.observation_type == obs_type)
                for obs_type in ObservationType
            },
            "alert_severities": {
                severity.value: sum(1 for obs in self.all_observations if obs.severity == severity)
                for severity in AlertSeverity
            }
        }