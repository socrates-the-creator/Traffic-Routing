"""
A* routing algorithm implementation for road networks.
This module provides A* pathfinding with traffic-aware cost functions.
"""

import heapq
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
import logging
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RouteResult:
    """Result of a routing operation."""
    path: List[str]
    total_distance: float
    total_time: float
    total_cost: float
    algorithm: str
    timestamp: datetime

class AStarRouter:
    """
    A* routing algorithm implementation for road networks.
    """
    
    def __init__(self, graph: nx.Graph, traffic_data: Dict = None):
        """
        Initialize A* router.
        
        Args:
            graph: NetworkX graph representing the road network
            traffic_data: Dictionary containing traffic information
        """
        self.graph = graph
        self.traffic_data = traffic_data or {}
        
    def find_route(self, 
                   start_node: str, 
                   end_node: str,
                   cost_function: str = 'time',
                   traffic_weight: float = 1.0) -> RouteResult:
        """
        Find optimal route using A* algorithm.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            cost_function: Cost function type ('time', 'distance', 'combined')
            traffic_weight: Weight for traffic in cost calculation
            
        Returns:
            RouteResult object with path and metrics
        """
        if start_node not in self.graph or end_node not in self.graph:
            raise ValueError(f"Start or end node not found in graph")
        
        # A* algorithm implementation
        open_set = [(0, start_node)]
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, end_node)}
        visited = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == end_node:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                return self._calculate_route_metrics(path, cost_function, traffic_weight)
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                
                # Calculate tentative g_score
                edge_cost = self._get_edge_cost(current, neighbor, cost_function, traffic_weight)
                tentative_g_score = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        raise ValueError(f"No path found from {start_node} to {end_node}")
    
    def _heuristic(self, node1: str, node2: str) -> float:
        """
        Calculate heuristic distance between two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            Heuristic distance
        """
        # Get coordinates
        lat1, lon1 = self.graph.nodes[node1]['lat'], self.graph.nodes[node1]['lon']
        lat2, lon2 = self.graph.nodes[node2]['lat'], self.graph.nodes[node2]['lon']
        
        # Haversine distance (simplified)
        return self._haversine_distance(lat1, lon1, lat2, lon2)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Haversine distance between two points.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def _get_edge_cost(self, 
                      source: str, 
                      target: str, 
                      cost_function: str,
                      traffic_weight: float) -> float:
        """
        Calculate edge cost based on specified cost function.
        
        Args:
            source: Source node
            target: Target node
            cost_function: Type of cost function
            traffic_weight: Weight for traffic
            
        Returns:
            Edge cost
        """
        edge_data = self.graph[source][target]
        
        if cost_function == 'distance':
            return edge_data.get('length', 100)
        
        elif cost_function == 'time':
            # Calculate travel time
            length = edge_data.get('length', 100)
            max_speed = self._get_max_speed(edge_data)
            current_speed = self._get_current_speed(source, target)
            
            # Apply traffic factor
            speed = current_speed * (1 - traffic_weight * self._get_traffic_factor(source, target))
            speed = max(speed, 5)  # Minimum speed of 5 km/h
            
            return length / (speed / 3.6)  # Convert to seconds
        
        elif cost_function == 'combined':
            # Combine distance and time
            distance_cost = edge_data.get('length', 100)
            time_cost = self._get_edge_cost(source, target, 'time', traffic_weight)
            
            # Normalize and combine
            return 0.3 * distance_cost + 0.7 * time_cost
        
        else:
            return edge_data.get('length', 100)
    
    def _get_max_speed(self, edge_data: Dict) -> float:
        """Get maximum speed for an edge."""
        try:
            return float(edge_data.get('maxspeed', 50))
        except:
            return 50.0
    
    def _get_current_speed(self, source: str, target: str) -> float:
        """Get current speed for an edge."""
        edge_key = f"{source}_{target}"
        if edge_key in self.traffic_data:
            return self.traffic_data[edge_key].get('current_speed', 50)
        return 50.0
    
    def _get_traffic_factor(self, source: str, target: str) -> float:
        """Get traffic factor for an edge."""
        edge_key = f"{source}_{target}"
        if edge_key in self.traffic_data:
            return self.traffic_data[edge_key].get('jam_factor', 0)
        return 0.0
    
    def _reconstruct_path(self, came_from: Dict, current: str) -> List[str]:
        """
        Reconstruct path from came_from dictionary.
        
        Args:
            came_from: Dictionary mapping nodes to their predecessors
            current: Current node
            
        Returns:
            List of nodes in the path
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _calculate_route_metrics(self, 
                               path: List[str], 
                               cost_function: str,
                               traffic_weight: float) -> RouteResult:
        """
        Calculate route metrics.
        
        Args:
            path: List of nodes in the path
            cost_function: Cost function used
            traffic_weight: Traffic weight used
            
        Returns:
            RouteResult with calculated metrics
        """
        total_distance = 0
        total_time = 0
        total_cost = 0
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            edge_data = self.graph[source][target]
            length = edge_data.get('length', 100)
            
            # Distance
            total_distance += length
            
            # Time
            max_speed = self._get_max_speed(edge_data)
            current_speed = self._get_current_speed(source, target)
            speed = current_speed * (1 - traffic_weight * self._get_traffic_factor(source, target))
            speed = max(speed, 5)
            time = length / (speed / 3.6)
            total_time += time
            
            # Cost
            cost = self._get_edge_cost(source, target, cost_function, traffic_weight)
            total_cost += cost
        
        return RouteResult(
            path=path,
            total_distance=total_distance,
            total_time=total_time,
            total_cost=total_cost,
            algorithm='A*',
            timestamp=datetime.now()
        )
    
    def find_multiple_routes(self, 
                           start_node: str, 
                           end_node: str,
                           num_routes: int = 3,
                           cost_function: str = 'time',
                           traffic_weight: float = 1.0) -> List[RouteResult]:
        """
        Find multiple alternative routes.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            num_routes: Number of routes to find
            cost_function: Cost function type
            traffic_weight: Traffic weight
            
        Returns:
            List of RouteResult objects
        """
        routes = []
        
        # Find primary route
        try:
            primary_route = self.find_route(start_node, end_node, cost_function, traffic_weight)
            routes.append(primary_route)
        except ValueError:
            return routes
        
        # Find alternative routes by penalizing used edges
        for i in range(num_routes - 1):
            try:
                # Create modified graph with penalized edges
                modified_graph = self._create_modified_graph(routes)
                modified_router = AStarRouter(modified_graph, self.traffic_data)
                
                alt_route = modified_router.find_route(start_node, end_node, cost_function, traffic_weight)
                routes.append(alt_route)
                
            except ValueError:
                break
        
        return routes
    
    def _create_modified_graph(self, existing_routes: List[RouteResult]) -> nx.Graph:
        """
        Create modified graph with penalized edges from existing routes.
        
        Args:
            existing_routes: List of existing routes
            
        Returns:
            Modified NetworkX graph
        """
        modified_graph = self.graph.copy()
        
        # Penalize edges used in existing routes
        for route in existing_routes:
            for i in range(len(route.path) - 1):
                source = route.path[i]
                target = route.path[i + 1]
                
                if modified_graph.has_edge(source, target):
                    # Increase edge cost
                    current_length = modified_graph[source][target].get('length', 100)
                    modified_graph[source][target]['length'] = current_length * 1.5
        
        return modified_graph

class DijkstraRouter:
    """
    Dijkstra's algorithm implementation for comparison with A*.
    """
    
    def __init__(self, graph: nx.Graph, traffic_data: Dict = None):
        """
        Initialize Dijkstra router.
        
        Args:
            graph: NetworkX graph
            traffic_data: Traffic data dictionary
        """
        self.graph = graph
        self.traffic_data = traffic_data or {}
    
    def find_route(self, 
                   start_node: str, 
                   end_node: str,
                   cost_function: str = 'time',
                   traffic_weight: float = 1.0) -> RouteResult:
        """
        Find route using Dijkstra's algorithm.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            cost_function: Cost function type
            traffic_weight: Traffic weight
            
        Returns:
            RouteResult object
        """
        if start_node not in self.graph or end_node not in self.graph:
            raise ValueError(f"Start or end node not found in graph")
        
        # Dijkstra's algorithm
        distances = {start_node: 0}
        previous = {}
        unvisited = set(self.graph.nodes())
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda node: distances.get(node, float('inf')))
            
            if current == end_node:
                break
            
            unvisited.remove(current)
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in unvisited:
                    edge_cost = self._get_edge_cost(current, neighbor, cost_function, traffic_weight)
                    new_distance = distances[current] + edge_cost
                    
                    if new_distance < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        if end_node not in previous and end_node != start_node:
            raise ValueError(f"No path found from {start_node} to {end_node}")
        
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = previous.get(current)
        path.reverse()
        
        return self._calculate_route_metrics(path, cost_function, traffic_weight)
    
    def _get_edge_cost(self, source: str, target: str, cost_function: str, traffic_weight: float) -> float:
        """Get edge cost (same as A* implementation)."""
        edge_data = self.graph[source][target]
        
        if cost_function == 'distance':
            return edge_data.get('length', 100)
        
        elif cost_function == 'time':
            length = edge_data.get('length', 100)
            max_speed = self._get_max_speed(edge_data)
            current_speed = self._get_current_speed(source, target)
            speed = current_speed * (1 - traffic_weight * self._get_traffic_factor(source, target))
            speed = max(speed, 5)
            return length / (speed / 3.6)
        
        else:
            return edge_data.get('length', 100)
    
    def _get_max_speed(self, edge_data: Dict) -> float:
        """Get maximum speed for an edge."""
        try:
            return float(edge_data.get('maxspeed', 50))
        except:
            return 50.0
    
    def _get_current_speed(self, source: str, target: str) -> float:
        """Get current speed for an edge."""
        edge_key = f"{source}_{target}"
        if edge_key in self.traffic_data:
            return self.traffic_data[edge_key].get('current_speed', 50)
        return 50.0
    
    def _get_traffic_factor(self, source: str, target: str) -> float:
        """Get traffic factor for an edge."""
        edge_key = f"{source}_{target}"
        if edge_key in self.traffic_data:
            return self.traffic_data[edge_key].get('jam_factor', 0)
        return 0.0
    
    def _calculate_route_metrics(self, path: List[str], cost_function: str, traffic_weight: float) -> RouteResult:
        """Calculate route metrics (same as A* implementation)."""
        total_distance = 0
        total_time = 0
        total_cost = 0
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            edge_data = self.graph[source][target]
            length = edge_data.get('length', 100)
            
            total_distance += length
            
            max_speed = self._get_max_speed(edge_data)
            current_speed = self._get_current_speed(source, target)
            speed = current_speed * (1 - traffic_weight * self._get_traffic_factor(source, target))
            speed = max(speed, 5)
            time = length / (speed / 3.6)
            total_time += time
            
            cost = self._get_edge_cost(source, target, cost_function, traffic_weight)
            total_cost += cost
        
        return RouteResult(
            path=path,
            total_distance=total_distance,
            total_time=total_time,
            total_cost=total_cost,
            algorithm='Dijkstra',
            timestamp=datetime.now()
        )

def main():
    """
    Example usage of routing algorithms.
    """
    # Create a simple test graph
    G = nx.Graph()
    
    # Add nodes with coordinates
    nodes = [
        ('A', {'lat': 19.0760, 'lon': 72.8777}),
        ('B', {'lat': 19.0760, 'lon': 72.8778}),
        ('C', {'lat': 19.0761, 'lon': 72.8777}),
        ('D', {'lat': 19.0761, 'lon': 72.8778}),
    ]
    
    G.add_nodes_from(nodes)
    
    # Add edges
    edges = [
        ('A', 'B', {'length': 100, 'maxspeed': '50'}),
        ('A', 'C', {'length': 150, 'maxspeed': '40'}),
        ('B', 'D', {'length': 150, 'maxspeed': '40'}),
        ('C', 'D', {'length': 100, 'maxspeed': '50'}),
    ]
    
    G.add_edges_from(edges)
    
    # Create traffic data
    traffic_data = {
        'A_B': {'current_speed': 30, 'jam_factor': 0.4},
        'A_C': {'current_speed': 45, 'jam_factor': 0.1},
        'B_D': {'current_speed': 45, 'jam_factor': 0.1},
        'C_D': {'current_speed': 25, 'jam_factor': 0.5},
    }
    
    # Test A* routing
    print("Testing A* routing...")
    astar_router = AStarRouter(G, traffic_data)
    
    try:
        route = astar_router.find_route('A', 'D', cost_function='time', traffic_weight=1.0)
        print(f"A* Route: {route.path}")
        print(f"Distance: {route.total_distance:.2f}m")
        print(f"Time: {route.total_time:.2f}s")
        print(f"Cost: {route.total_cost:.2f}")
    except ValueError as e:
        print(f"A* Error: {e}")
    
    # Test Dijkstra routing
    print("\nTesting Dijkstra routing...")
    dijkstra_router = DijkstraRouter(G, traffic_data)
    
    try:
        route = dijkstra_router.find_route('A', 'D', cost_function='time', traffic_weight=1.0)
        print(f"Dijkstra Route: {route.path}")
        print(f"Distance: {route.total_distance:.2f}m")
        print(f"Time: {route.total_time:.2f}s")
        print(f"Cost: {route.total_cost:.2f}")
    except ValueError as e:
        print(f"Dijkstra Error: {e}")

if __name__ == "__main__":
    main()
