"""
Spatial Reasoning Module

Implements geometric reasoning, topological analysis, and spatial relationship processing.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import asyncio
import math
from dataclasses import dataclass
from .base import BaseReasoner, ReasoningContext, ReasoningResult, ReasoningType, Evidence, Hypothesis, Conclusion


@dataclass
class Point:
    """Represents a point in 2D/3D space."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class SpatialRegion:
    """Represents a spatial region."""
    id: str
    center: Point
    radius: float
    region_type: str = "circle"


class GeometricReasoner:
    """Performs geometric calculations and reasoning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def calculate_distance(self, point1: Point, point2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(
            (point1.x - point2.x) ** 2 + 
            (point1.y - point2.y) ** 2 + 
            (point1.z - point2.z) ** 2
        )
    
    async def find_nearest_neighbors(self, target: Point, points: List[Point], k: int = 3) -> List[Tuple[Point, float]]:
        """Find k nearest neighbors to target point."""
        distances = []
        for point in points:
            distance = await self.calculate_distance(target, point)
            distances.append((point, distance))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    async def analyze_spatial_distribution(self, points: List[Point]) -> Dict[str, Any]:
        """Analyze spatial distribution of points."""
        if len(points) < 2:
            return {'distribution': 'insufficient_data'}
        
        # Calculate centroid
        centroid = Point(
            x=sum(p.x for p in points) / len(points),
            y=sum(p.y for p in points) / len(points),
            z=sum(p.z for p in points) / len(points)
        )
        
        # Calculate spread
        distances_from_centroid = []
        for point in points:
            distance = await self.calculate_distance(point, centroid)
            distances_from_centroid.append(distance)
        
        avg_distance = sum(distances_from_centroid) / len(distances_from_centroid)
        
        # Determine distribution type
        if avg_distance < 1.0:
            distribution = 'clustered'
        elif avg_distance > 5.0:
            distribution = 'dispersed'
        else:
            distribution = 'random'
        
        return {
            'distribution': distribution,
            'centroid': centroid,
            'average_distance_from_centroid': avg_distance,
            'point_count': len(points)
        }


class TopologicalReasoner:
    """Performs topological reasoning about spatial relationships."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def analyze_containment(self, regions: List[SpatialRegion]) -> List[Dict[str, Any]]:
        """Analyze containment relationships between regions."""
        containment_relations = []
        
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i != j:
                    relation = await self._check_containment(region1, region2)
                    if relation:
                        containment_relations.append(relation)
        
        return containment_relations
    
    async def _check_containment(self, region1: SpatialRegion, region2: SpatialRegion) -> Optional[Dict[str, Any]]:
        """Check if one region contains another."""
        # Simple containment check for circular regions
        if region1.region_type == "circle" and region2.region_type == "circle":
            distance = math.sqrt(
                (region1.center.x - region2.center.x) ** 2 + 
                (region1.center.y - region2.center.y) ** 2
            )
            
            if distance + region2.radius <= region1.radius:
                return {
                    'type': 'contains',
                    'container': region1.id,
                    'contained': region2.id,
                    'confidence': 1.0
                }
            elif distance <= region1.radius + region2.radius:
                return {
                    'type': 'overlaps',
                    'region1': region1.id,
                    'region2': region2.id,
                    'confidence': 0.8
                }
        
        return None


class RegionAnalyzer:
    """Analyzes spatial regions and their properties."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def analyze_region_properties(self, region: SpatialRegion) -> Dict[str, Any]:
        """Analyze properties of a spatial region."""
        properties = {
            'id': region.id,
            'center': region.center,
            'type': region.region_type
        }
        
        if region.region_type == "circle":
            properties['area'] = math.pi * region.radius ** 2
            properties['circumference'] = 2 * math.pi * region.radius
        
        return properties


class SpatialReasoner(BaseReasoner):
    """Main spatial reasoning engine."""
    
    def __init__(self, name: str = "SpatialReasoner", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, ReasoningType.SPATIAL, config)
        self.geometric_reasoner = GeometricReasoner(config)
        self.topological_reasoner = TopologicalReasoner(config)
        self.region_analyzer = RegionAnalyzer(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform spatial reasoning."""
        result = ReasoningResult(reasoning_type=self.reasoning_type)
        
        try:
            # Extract spatial data from context
            points, regions = await self._extract_spatial_data(context)
            
            if points:
                # Analyze spatial distribution
                distribution_analysis = await self.geometric_reasoner.analyze_spatial_distribution(points)
                result.reasoning_trace.append(f"Analyzed distribution of {len(points)} points")
                
                conclusion = Conclusion(
                    statement=f"Spatial distribution: {distribution_analysis['distribution']}",
                    confidence=0.8,
                    reasoning_type=self.reasoning_type,
                    reasoning_chain=["Geometric distribution analysis"],
                    metadata={'analysis': distribution_analysis}
                )
                result.conclusions.append(conclusion)
            
            if regions:
                # Analyze topological relationships
                containment_relations = await self.topological_reasoner.analyze_containment(regions)
                result.reasoning_trace.append(f"Found {len(containment_relations)} spatial relationships")
                
                for relation in containment_relations:
                    conclusion = Conclusion(
                        statement=f"Spatial relationship: {relation['type']} between regions",
                        confidence=relation['confidence'],
                        reasoning_type=self.reasoning_type,
                        reasoning_chain=["Topological analysis"],
                        metadata={'relation': relation}
                    )
                    result.conclusions.append(conclusion)
            
            result.confidence = 0.7 if (points or regions) else 0.0
            result.success = len(result.conclusions) > 0
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        return result
    
    async def _extract_spatial_data(self, context: ReasoningContext) -> Tuple[List[Point], List[SpatialRegion]]:
        """Extract spatial data from context."""
        points = []
        regions = []
        
        # Extract from metadata
        if 'spatial_points' in context.metadata:
            for point_data in context.metadata['spatial_points']:
                point = Point(
                    x=point_data.get('x', 0),
                    y=point_data.get('y', 0),
                    z=point_data.get('z', 0)
                )
                points.append(point)
        
        if 'spatial_regions' in context.metadata:
            for region_data in context.metadata['spatial_regions']:
                center = Point(
                    x=region_data.get('center_x', 0),
                    y=region_data.get('center_y', 0),
                    z=region_data.get('center_z', 0)
                )
                region = SpatialRegion(
                    id=region_data.get('id', f"region_{len(regions)}"),
                    center=center,
                    radius=region_data.get('radius', 1.0),
                    region_type=region_data.get('type', 'circle')
                )
                regions.append(region)
        
        return points, regions