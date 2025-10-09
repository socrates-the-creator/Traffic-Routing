"""
GNN-based routing algorithm for intelligent pathfinding.
This module uses trained GNN models to predict optimal routes.
"""

import numpy as np
import networkx as nx
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import pickle
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import TrafficGNNModel, TrafficPredictor
from routing.a_star import RouteResult, AStarRouter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNNRouter:
    """
    GNN-based routing algorithm that uses trained models for intelligent pathfinding.
    """
    
    def __init__(self, 
                 graph: nx.Graph, 
                 model_path: str = None,
                 dataset_path: str = None,
                 traffic_data: Dict = None):
        """
        Initialize GNN router.
        
        Args:
            graph: NetworkX graph representing the road network
            model_path: Path to trained GNN model
            dataset_path: Path to processed dataset
            traffic_data: Dictionary containing traffic information
        """
        self.graph = graph
        self.traffic_data = traffic_data or {}
        self.model = None
        self.dataset = None
        self.traffic_predictor = None
        
        # Load model and dataset if paths provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if dataset_path and os.path.exists(dataset_path):
            self.load_dataset(dataset_path)
        
        # Initialize traffic predictor
        if self.model and self.dataset:
            self.traffic_predictor = TrafficPredictor()
            self.traffic_predictor.model = self.model
            self.traffic_predictor.dataset = self.dataset
    
    def load_model(self, model_path: str):
        """
        Load trained GNN model.
        
        Args:
            model_path: Path to the model
        """
        try:
            # Load model architecture and weights
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded GNN model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def load_dataset(self, dataset_path: str):
        """
        Load processed dataset.
        
        Args:
            dataset_path: Path to the dataset
        """
        try:
            with open(dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
            logger.info(f"Loaded dataset from {dataset_path}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.dataset = None
    
    def predict_traffic_conditions(self) -> Dict:
        """
        Predict traffic conditions using the GNN model.
        
        Returns:
            Dictionary with traffic predictions
        """
        if not self.traffic_predictor:
            logger.warning("Traffic predictor not initialized")
            return {}
        
        try:
            predictions = self.traffic_predictor.predict_current_traffic()
            return predictions
        except Exception as e:
            logger.error(f"Error predicting traffic: {e}")
            return {}
    
    def find_route(self, 
                   start_node: str, 
                   end_node: str,
                   use_gnn_predictions: bool = True,
                   cost_function: str = 'time',
                   traffic_weight: float = 1.0) -> RouteResult:
        """
        Find optimal route using GNN-enhanced routing.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            use_gnn_predictions: Whether to use GNN predictions
            cost_function: Cost function type
            traffic_weight: Traffic weight
            
        Returns:
            RouteResult object with path and metrics
        """
        if start_node not in self.graph or end_node not in self.graph:
            raise ValueError(f"Start or end node not found in graph")
        
        # Get GNN predictions if available
        gnn_traffic_data = {}
        if use_gnn_predictions and self.traffic_predictor:
            gnn_predictions = self.predict_traffic_conditions()
            gnn_traffic_data = self._convert_predictions_to_traffic_data(gnn_predictions)
        
        # Combine with real-time traffic data
        combined_traffic_data = {**self.traffic_data, **gnn_traffic_data}
        
        # Use A* with enhanced traffic data
        astar_router = AStarRouter(self.graph, combined_traffic_data)
        
        try:
            route = astar_router.find_route(start_node, end_node, cost_function, traffic_weight)
            route.algorithm = 'GNN-Enhanced A*'
            return route
        except ValueError as e:
            # Fallback to basic A* if GNN routing fails
            logger.warning(f"GNN routing failed, falling back to basic A*: {e}")
            basic_router = AStarRouter(self.graph, self.traffic_data)
            route = basic_router.find_route(start_node, end_node, cost_function, traffic_weight)
            route.algorithm = 'Fallback A*'
            return route
    
    def _convert_predictions_to_traffic_data(self, predictions: Dict) -> Dict:
        """
        Convert GNN predictions to traffic data format.
        
        Args:
            predictions: GNN predictions dictionary
            
        Returns:
            Traffic data dictionary
        """
        traffic_data = {}
        
        if 'node_predictions' in predictions and 'node_mapping' in predictions:
            node_predictions = predictions['node_predictions']
            node_mapping = predictions['node_mapping']
            
            # Create reverse mapping
            reverse_mapping = {v: k for k, v in node_mapping.items()}
            
            # Convert predictions to traffic data
            for i, prediction in enumerate(node_predictions):
                if i in reverse_mapping:
                    node_id = reverse_mapping[i]
                    
                    # Find connected edges for this node
                    for neighbor in self.graph.neighbors(node_id):
                        edge_key = f"{node_id}_{neighbor}"
                        
                        # Convert prediction to traffic metrics
                        predicted_speed = max(prediction * 50, 5)  # Scale prediction
                        jam_factor = max(0, 1 - (predicted_speed / 50))
                        
                        traffic_data[edge_key] = {
                            'current_speed': predicted_speed,
                            'jam_factor': jam_factor,
                            'confidence': 0.8,  # GNN confidence
                            'source': 'gnn_prediction'
                        }
        
        return traffic_data
    
    def find_multiple_routes(self, 
                           start_node: str, 
                           end_node: str,
                           num_routes: int = 3,
                           use_gnn_predictions: bool = True,
                           cost_function: str = 'time',
                           traffic_weight: float = 1.0) -> List[RouteResult]:
        """
        Find multiple alternative routes using GNN predictions.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            num_routes: Number of routes to find
            use_gnn_predictions: Whether to use GNN predictions
            cost_function: Cost function type
            traffic_weight: Traffic weight
            
        Returns:
            List of RouteResult objects
        """
        routes = []
        
        # Get GNN predictions
        gnn_traffic_data = {}
        if use_gnn_predictions and self.traffic_predictor:
            gnn_predictions = self.predict_traffic_conditions()
            gnn_traffic_data = self._convert_predictions_to_traffic_data(gnn_predictions)
        
        # Combine traffic data
        combined_traffic_data = {**self.traffic_data, **gnn_traffic_data}
        
        # Use A* with enhanced traffic data
        astar_router = AStarRouter(self.graph, combined_traffic_data)
        
        try:
            routes = astar_router.find_multiple_routes(
                start_node, end_node, num_routes, cost_function, traffic_weight
            )
            
            # Update algorithm names
            for i, route in enumerate(routes):
                if i == 0:
                    route.algorithm = 'GNN-Enhanced A* (Primary)'
                else:
                    route.algorithm = f'GNN-Enhanced A* (Alternative {i})'
            
        except ValueError as e:
            logger.warning(f"GNN multi-route failed: {e}")
            # Fallback to basic A*
            basic_router = AStarRouter(self.graph, self.traffic_data)
            routes = basic_router.find_multiple_routes(
                start_node, end_node, num_routes, cost_function, traffic_weight
            )
        
        return routes
    
    def get_route_recommendations(self, 
                                start_node: str, 
                                end_node: str,
                                preferences: Dict = None) -> Dict:
        """
        Get intelligent route recommendations based on GNN predictions.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            preferences: User preferences (fastest, shortest, scenic, etc.)
            
        Returns:
            Dictionary with route recommendations
        """
        preferences = preferences or {'priority': 'fastest'}
        
        # Get GNN predictions
        gnn_predictions = {}
        if self.traffic_predictor:
            gnn_predictions = self.predict_traffic_conditions()
        
        # Find multiple routes
        routes = self.find_multiple_routes(start_node, end_node, num_routes=3)
        
        # Analyze routes based on preferences
        recommendations = self._analyze_routes(routes, preferences, gnn_predictions)
        
        return recommendations
    
    def _analyze_routes(self, 
                       routes: List[RouteResult], 
                       preferences: Dict,
                       gnn_predictions: Dict) -> Dict:
        """
        Analyze routes and provide recommendations.
        
        Args:
            routes: List of routes
            preferences: User preferences
            gnn_predictions: GNN predictions
            
        Returns:
            Analysis and recommendations
        """
        if not routes:
            return {'error': 'No routes found'}
        
        # Sort routes based on preferences
        if preferences.get('priority') == 'fastest':
            routes.sort(key=lambda x: x.total_time)
        elif preferences.get('priority') == 'shortest':
            routes.sort(key=lambda x: x.total_distance)
        else:
            routes.sort(key=lambda x: x.total_cost)
        
        # Analyze traffic conditions
        traffic_analysis = self._analyze_traffic_conditions(routes, gnn_predictions)
        
        # Create recommendations
        recommendations = {
            'best_route': routes[0],
            'alternatives': routes[1:] if len(routes) > 1 else [],
            'traffic_analysis': traffic_analysis,
            'recommendations': self._generate_recommendations(routes, traffic_analysis),
            'timestamp': datetime.now().isoformat()
        }
        
        return recommendations
    
    def _analyze_traffic_conditions(self, 
                                  routes: List[RouteResult], 
                                  gnn_predictions: Dict) -> Dict:
        """
        Analyze traffic conditions along routes.
        
        Args:
            routes: List of routes
            gnn_predictions: GNN predictions
            
        Returns:
            Traffic analysis
        """
        analysis = {
            'overall_condition': 'unknown',
            'congestion_level': 0,
            'predicted_delays': [],
            'route_conditions': []
        }
        
        for route in routes:
            route_condition = {
                'route_id': route.algorithm,
                'congestion_score': 0,
                'delay_risk': 'low',
                'traffic_incidents': 0
            }
            
            # Analyze each segment
            for i in range(len(route.path) - 1):
                source = route.path[i]
                target = route.path[i + 1]
                edge_key = f"{source}_{target}"
                
                # Get traffic data
                traffic_info = self.traffic_data.get(edge_key, {})
                jam_factor = traffic_info.get('jam_factor', 0)
                
                route_condition['congestion_score'] += jam_factor
            
            # Average congestion score
            if len(route.path) > 1:
                route_condition['congestion_score'] /= (len(route.path) - 1)
            
            # Determine delay risk
            if route_condition['congestion_score'] > 0.7:
                route_condition['delay_risk'] = 'high'
            elif route_condition['congestion_score'] > 0.4:
                route_condition['delay_risk'] = 'medium'
            else:
                route_condition['delay_risk'] = 'low'
            
            analysis['route_conditions'].append(route_condition)
        
        # Overall condition
        avg_congestion = np.mean([r['congestion_score'] for r in analysis['route_conditions']])
        analysis['congestion_level'] = avg_congestion
        
        if avg_congestion > 0.7:
            analysis['overall_condition'] = 'heavy_congestion'
        elif avg_congestion > 0.4:
            analysis['overall_condition'] = 'moderate_congestion'
        else:
            analysis['overall_condition'] = 'light_traffic'
        
        return analysis
    
    def _generate_recommendations(self, 
                                routes: List[RouteResult], 
                                traffic_analysis: Dict) -> List[str]:
        """
        Generate route recommendations.
        
        Args:
            routes: List of routes
            traffic_analysis: Traffic analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Overall traffic condition
        condition = traffic_analysis['overall_condition']
        if condition == 'heavy_congestion':
            recommendations.append("Heavy traffic detected. Consider alternative routes or delay travel.")
        elif condition == 'moderate_congestion':
            recommendations.append("Moderate traffic. Primary route recommended with minor delays expected.")
        else:
            recommendations.append("Light traffic conditions. All routes should have minimal delays.")
        
        # Route-specific recommendations
        for route_condition in traffic_analysis['route_conditions']:
            if route_condition['delay_risk'] == 'high':
                recommendations.append(f"Route {route_condition['route_id']} has high delay risk.")
            elif route_condition['delay_risk'] == 'medium':
                recommendations.append(f"Route {route_condition['route_id']} has moderate delay risk.")
        
        # Time-based recommendations
        if routes:
            best_route = routes[0]
            if best_route.total_time > 1800:  # More than 30 minutes
                recommendations.append("Long travel time expected. Consider public transport alternatives.")
        
        return recommendations
    
    def update_traffic_data(self, new_traffic_data: Dict):
        """
        Update traffic data with new information.
        
        Args:
            new_traffic_data: New traffic data dictionary
        """
        self.traffic_data.update(new_traffic_data)
        logger.info(f"Updated traffic data with {len(new_traffic_data)} new entries")
    
    def get_traffic_insights(self) -> Dict:
        """
        Get traffic insights from GNN predictions.
        
        Returns:
            Dictionary with traffic insights
        """
        if not self.traffic_predictor:
            return {'error': 'Traffic predictor not available'}
        
        try:
            predictions = self.predict_traffic_conditions()
            
            insights = {
                'prediction_confidence': 0.8,  # GNN confidence
                'predicted_congestion_hotspots': [],
                'recommended_avoidance_areas': [],
                'optimal_travel_times': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Analyze predictions for insights
            if 'node_predictions' in predictions:
                node_predictions = predictions['node_predictions']
                
                # Find congestion hotspots (low predicted speeds)
                congestion_threshold = 0.3  # Below 30% of normal speed
                for i, prediction in enumerate(node_predictions):
                    if prediction < congestion_threshold:
                        insights['predicted_congestion_hotspots'].append({
                            'node_index': i,
                            'predicted_speed_ratio': prediction,
                            'severity': 'high' if prediction < 0.2 else 'medium'
                        })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting traffic insights: {e}")
            return {'error': str(e)}

def main():
    """
    Example usage of GNN router.
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
    
    # Initialize GNN router (without model for demo)
    print("Testing GNN router...")
    gnn_router = GNNRouter(G, traffic_data=traffic_data)
    
    try:
        # Find route
        route = gnn_router.find_route('A', 'D', use_gnn_predictions=False)
        print(f"GNN Route: {route.path}")
        print(f"Algorithm: {route.algorithm}")
        print(f"Distance: {route.total_distance:.2f}m")
        print(f"Time: {route.total_time:.2f}s")
        
        # Get recommendations
        recommendations = gnn_router.get_route_recommendations('A', 'D')
        print(f"\nRecommendations: {recommendations['recommendations']}")
        
    except ValueError as e:
        print(f"GNN Router Error: {e}")

if __name__ == "__main__":
    main()
