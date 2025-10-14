"""
Temporal Reasoning Module

Implements time-based logic, sequence analysis, and temporal pattern recognition.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from .base import BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType, Evidence, Hypothesis, Conclusion


@dataclass
class TemporalEvent:
    """Represents a temporal event."""
    id: str
    description: str
    timestamp: datetime
    duration: Optional[timedelta] = None
    event_type: str = "instant"


class SequenceAnalyzer:
    """Analyzes temporal sequences and patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def analyze_sequence(self, events: List[TemporalEvent]) -> Dict[str, Any]:
        """Analyze a sequence of temporal events."""
        if len(events) < 2:
            return {'patterns': [], 'duration': timedelta(0)}
        
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        patterns = []
        
        # Look for regular intervals
        intervals = []
        for i in range(1, len(sorted_events)):
            interval = sorted_events[i].timestamp - sorted_events[i-1].timestamp
            intervals.append(interval.total_seconds())
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            if all(abs(interval - avg_interval) < avg_interval * 0.1 for interval in intervals):
                patterns.append({
                    'type': 'regular_interval',
                    'interval': avg_interval,
                    'confidence': 0.8
                })
        
        total_duration = sorted_events[-1].timestamp - sorted_events[0].timestamp
        
        return {
            'patterns': patterns,
            'duration': total_duration,
            'event_count': len(events)
        }


class EventCorrelator:
    """Correlates events across time."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def find_correlations(self, event_sequences: List[List[TemporalEvent]]) -> List[Dict[str, Any]]:
        """Find correlations between event sequences."""
        correlations = []
        
        for i, seq1 in enumerate(event_sequences):
            for j, seq2 in enumerate(event_sequences[i+1:], i+1):
                correlation = await self._calculate_correlation(seq1, seq2)
                if correlation['strength'] > 0.5:
                    correlations.append({
                        'sequence1_index': i,
                        'sequence2_index': j,
                        'correlation': correlation
                    })
        
        return correlations
    
    async def _calculate_correlation(self, seq1: List[TemporalEvent], seq2: List[TemporalEvent]) -> Dict[str, Any]:
        """Calculate correlation between two sequences."""
        if not seq1 or not seq2:
            return {'strength': 0.0, 'type': 'none'}
        
        # Simple temporal proximity correlation
        matches = 0
        total_comparisons = 0
        
        for event1 in seq1:
            for event2 in seq2:
                total_comparisons += 1
                time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
                if time_diff < 3600:  # Within 1 hour
                    matches += 1
        
        strength = matches / total_comparisons if total_comparisons > 0 else 0.0
        
        return {
            'strength': strength,
            'type': 'temporal_proximity',
            'matches': matches
        }


class TimeSeriesReasoner:
    """Reasons about time series data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def analyze_trends(self, time_series: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        if len(time_series) < 2:
            return {'trend': 'insufficient_data'}
        
        values = [point[1] for point in time_series]
        
        # Simple trend analysis
        if values[-1] > values[0]:
            trend = 'increasing'
        elif values[-1] < values[0]:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate rate of change
        time_span = (time_series[-1][0] - time_series[0][0]).total_seconds()
        value_change = values[-1] - values[0]
        rate = value_change / time_span if time_span > 0 else 0
        
        return {
            'trend': trend,
            'rate_of_change': rate,
            'start_value': values[0],
            'end_value': values[-1]
        }


class TemporalReasoner(BaseReasoner):
    """Main temporal reasoning engine."""
    
    def __init__(self, name: str = "TemporalReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.TEMPORAL, config)
        self.sequence_analyzer = SequenceAnalyzer(config)
        self.event_correlator = EventCorrelator(config)
        self.time_series_reasoner = TimeSeriesReasoner(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform temporal reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Extract temporal events from evidence
            events = await self._extract_temporal_events(context.available_evidence)
            
            if events:
                # Analyze sequence
                sequence_analysis = await self.sequence_analyzer.analyze_sequence(events)
                result.reasoning_trace.append(f"Analyzed sequence of {len(events)} events")
                
                for pattern in sequence_analysis['patterns']:
                    conclusion = Conclusion(
                        statement=f"Temporal pattern detected: {pattern['type']}",
                        confidence=pattern['confidence'],
                        reasoning_type=self.reasoning_type,
                        reasoning_chain=["Temporal sequence analysis"],
                        metadata={'pattern': pattern}
                    )
                    result.conclusions.append(conclusion)
                
                result.confidence = 0.7 if sequence_analysis['patterns'] else 0.3
                result.success = True
            else:
                result.success = False
                result.error_message = "No temporal events found in evidence"
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        return result
    
    async def _extract_temporal_events(self, evidence_list: List[Evidence]) -> List[TemporalEvent]:
        """Extract temporal events from evidence."""
        events = []
        
        for evidence in evidence_list:
            if hasattr(evidence, 'timestamp') and evidence.timestamp:
                event = TemporalEvent(
                    id=evidence.id,
                    description=str(evidence.content),
                    timestamp=evidence.timestamp
                )
                events.append(event)
        
        return events