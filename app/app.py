"""
Flask web application for the Mumbai Navigation System.
This module provides the main web interface for the navigation system.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest.tomtom_fetch import TomTomTrafficFetcher
from ingest.osm_overpass import OSMDataFetcher
from preprocess.build_graph import RoadNetworkProcessor
from models.model import TrafficGNNModel, TrafficPredictor
from routing.a_star import AStarRouter, DijkstraRouter
from routing.gnn_router import GNNRouter
from routing.route_compare import RouteComparator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mumbai_navigation_system_2024'

# Global variables for caching
cached_graph = None
cached_traffic_data = None
cached_gnn_model = None
cached_gnn_dataset = None
cached_routers = {}

def initialize_system():
    """Initialize the navigation system components."""
    global cached_graph, cached_traffic_data, cached_gnn_model, cached_gnn_dataset, cached_routers
    
    logger.info("Initializing navigation system...")
    
    try:
        # Load or create graph
        if cached_graph is None:
            cached_graph = load_or_create_graph()
        
        # Load traffic data
        if cached_traffic_data is None:
            cached_traffic_data = load_traffic_data()
        
        # Load GNN model and dataset
        if cached_gnn_model is None or cached_gnn_dataset is None:
            cached_gnn_model, cached_gnn_dataset = load_gnn_components()
        
        # Initialize routers
        if not cached_routers:
            cached_routers = initialize_routers()
        
        logger.info("Navigation system initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")

def load_or_create_graph():
    """Load existing graph or create new one."""
    graph_file = "data/processed/mumbai_graph.pkl"
    
    if os.path.exists(graph_file):
        logger.info("Loading existing graph...")
        with open(graph_file, 'rb') as f:
            return pickle.load(f)
    else:
        logger.info("Creating new graph...")
        return create_new_graph()

def create_new_graph():
    """Create new graph from OSM data."""
    # Fetch OSM data
    osm_fetcher = OSMDataFetcher()
    roads = osm_fetcher.get_road_network()
    
    # Process into graph
    processor = RoadNetworkProcessor()
    graph = processor.build_networkx_graph(roads)
    
    # Save graph
    os.makedirs(os.path.dirname("data/processed/mumbai_graph.pkl"), exist_ok=True)
    with open("data/processed/mumbai_graph.pkl", 'wb') as f:
        pickle.dump(graph, f)
    
    return graph

def load_traffic_data():
    """Load traffic data."""
    traffic_file = "data/raw/current_traffic.csv"
    
    if os.path.exists(traffic_file):
        logger.info("Loading existing traffic data...")
        return pd.read_csv(traffic_file).to_dict('records')
    else:
        logger.info("Fetching new traffic data...")
        return fetch_new_traffic_data()

def fetch_new_traffic_data():
    """Fetch new traffic data from TomTom."""
    fetcher = TomTomTrafficFetcher()
    traffic_data = fetcher.get_traffic_flow_data()
    
    # Save traffic data
    os.makedirs(os.path.dirname("data/raw/current_traffic.csv"), exist_ok=True)
    traffic_data.to_csv("data/raw/current_traffic.csv", index=False)
    
    return traffic_data.to_dict('records')

def load_gnn_components():
    """Load GNN model and dataset."""
    model_path = "data/models/traffic_gnn_model"
    dataset_path = "data/processed/processed_graph.pkl"
    
    model = None
    dataset = None
    
    if os.path.exists(model_path) and os.path.exists(dataset_path):
        logger.info("Loading GNN model and dataset...")
        try:
            model = TrafficGNNModel(1, 1, 1)  # Dummy initialization
            model.load_model(model_path)
            
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load GNN components: {e}")
    
    return model, dataset

def initialize_routers():
    """Initialize routing algorithms."""
    routers = {}
    
    # Convert traffic data to dictionary format
    traffic_dict = {}
    for record in cached_traffic_data:
        if 'segment_id' in record:
            traffic_dict[record['segment_id']] = record
    
    # Initialize routers
    routers['astar'] = AStarRouter(cached_graph, traffic_dict)
    routers['dijkstra'] = DijkstraRouter(cached_graph, traffic_dict)
    
    if cached_gnn_model and cached_gnn_dataset:
        routers['gnn'] = GNNRouter(cached_graph, None, None, traffic_dict)
        routers['gnn'].model = cached_gnn_model
        routers['gnn'].dataset = cached_gnn_dataset
    
    return routers

@app.route('/')
def index():
    """Main page."""
    return render_template('map.html')

@app.route('/api/route', methods=['POST'])
def get_route():
    """Get route between two points."""
    try:
        data = request.get_json()
        start_lat = float(data['start_lat'])
        start_lon = float(data['start_lon'])
        end_lat = float(data['end_lat'])
        end_lon = float(data['end_lon'])
        algorithm = data.get('algorithm', 'astar')
        cost_function = data.get('cost_function', 'time')
        
        # Find closest nodes to coordinates
        start_node = find_closest_node(start_lat, start_lon)
        end_node = find_closest_node(end_lat, end_lon)
        
        if not start_node or not end_node:
            return jsonify({'error': 'Could not find start or end point'}), 400
        
        # Get route
        if algorithm in cached_routers:
            router = cached_routers[algorithm]
            route = router.find_route(start_node, end_node, cost_function)
            
            # Convert route to GeoJSON format
            route_geojson = convert_route_to_geojson(route)
            
            return jsonify({
                'route': route_geojson,
                'metrics': {
                    'distance': route.total_distance,
                    'time': route.total_time,
                    'cost': route.total_cost,
                    'algorithm': route.algorithm
                }
            })
        else:
            return jsonify({'error': f'Algorithm {algorithm} not available'}), 400
            
    except Exception as e:
        logger.error(f"Error getting route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_routes', methods=['POST'])
def compare_routes():
    """Compare routes using different algorithms."""
    try:
        data = request.get_json()
        start_lat = float(data['start_lat'])
        start_lon = float(data['start_lon'])
        end_lat = float(data['end_lat'])
        end_lon = float(data['end_lon'])
        cost_function = data.get('cost_function', 'time')
        
        # Find closest nodes
        start_node = find_closest_node(start_lat, start_lon)
        end_node = find_closest_node(end_lat, end_lon)
        
        if not start_node or not end_node:
            return jsonify({'error': 'Could not find start or end point'}), 400
        
        # Initialize comparator
        comparator = RouteComparator(cached_graph, cached_traffic_data)
        
        # Compare routes
        comparison = comparator.compare_routes(
            start_node, end_node, cost_function, include_gnn='gnn' in cached_routers
        )
        
        # Convert results to JSON format
        results = []
        for result in comparison.results:
            if result.success:
                route_geojson = convert_route_to_geojson(result.route)
                results.append({
                    'algorithm': result.algorithm,
                    'route': route_geojson,
                    'metrics': {
                        'distance': result.route.total_distance,
                        'time': result.route.total_time,
                        'cost': result.route.total_cost,
                        'computation_time': result.computation_time
                    }
                })
        
        return jsonify({
            'results': results,
            'best_route': comparison.best_route.algorithm if comparison.best_route else None,
            'performance_metrics': comparison.performance_metrics
        })
        
    except Exception as e:
        logger.error(f"Error comparing routes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/traffic_data')
def get_traffic_data():
    """Get current traffic data."""
    try:
        # Convert traffic data to GeoJSON format
        traffic_geojson = {
            'type': 'FeatureCollection',
            'features': []
        }
        
        for record in cached_traffic_data:
            if 'latitude' in record and 'longitude' in record:
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [record['longitude'], record['latitude']]
                    },
                    'properties': {
                        'current_speed': record.get('current_speed', 50),
                        'jam_factor': record.get('jam_factor', 0),
                        'traffic_level': record.get('traffic_level', 'UNKNOWN'),
                        'confidence': record.get('confidence', 0.8)
                    }
                }
                traffic_geojson['features'].append(feature)
        
        return jsonify(traffic_geojson)
        
    except Exception as e:
        logger.error(f"Error getting traffic data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_traffic', methods=['POST'])
def update_traffic():
    """Update traffic data."""
    try:
        global cached_traffic_data
        
        # Fetch new traffic data
        fetcher = TomTomTrafficFetcher()
        new_traffic = fetcher.get_traffic_flow_data()
        
        # Update cached data
        cached_traffic_data = new_traffic.to_dict('records')
        
        # Save to file
        new_traffic.to_csv("data/raw/current_traffic.csv", index=False)
        
        # Update routers
        traffic_dict = {}
        for record in cached_traffic_data:
            if 'segment_id' in record:
                traffic_dict[record['segment_id']] = record
        
        for router in cached_routers.values():
            if hasattr(router, 'traffic_data'):
                router.traffic_data.update(traffic_dict)
        
        return jsonify({'message': 'Traffic data updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating traffic data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_traffic', methods=['POST'])
def predict_traffic():
    """Get traffic predictions from GNN model."""
    try:
        if 'gnn' not in cached_routers:
            return jsonify({'error': 'GNN model not available'}), 400
        
        gnn_router = cached_routers['gnn']
        predictions = gnn_router.predict_traffic_conditions()
        
        # Convert predictions to GeoJSON format
        prediction_geojson = {
            'type': 'FeatureCollection',
            'features': []
        }
        
        if 'node_predictions' in predictions and 'node_mapping' in predictions:
            node_predictions = predictions['node_predictions']
            node_mapping = predictions['node_mapping']
            
            for node_id, prediction in zip(node_mapping.keys(), node_predictions):
                if node_id in cached_graph.nodes:
                    node_data = cached_graph.nodes[node_id]
                    feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [node_data['lon'], node_data['lat']]
                        },
                        'properties': {
                            'predicted_speed': float(prediction * 50),  # Scale prediction
                            'prediction_confidence': 0.8,
                            'node_id': node_id
                        }
                    }
                    prediction_geojson['features'].append(feature)
        
        return jsonify(prediction_geojson)
        
    except Exception as e:
        logger.error(f"Error predicting traffic: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/route_analysis', methods=['POST'])
def analyze_route():
    """Analyze route and provide insights."""
    try:
        data = request.get_json()
        start_lat = float(data['start_lat'])
        start_lon = float(data['start_lon'])
        end_lat = float(data['end_lat'])
        end_lon = float(data['end_lon'])
        
        # Find closest nodes
        start_node = find_closest_node(start_lat, start_lon)
        end_node = find_closest_node(end_lat, end_lon)
        
        if not start_node or not end_node:
            return jsonify({'error': 'Could not find start or end point'}), 400
        
        # Get route recommendations
        if 'gnn' in cached_routers:
            gnn_router = cached_routers['gnn']
            recommendations = gnn_router.get_route_recommendations(start_node, end_node)
            
            return jsonify({
                'recommendations': recommendations,
                'traffic_insights': gnn_router.get_traffic_insights()
            })
        else:
            return jsonify({'error': 'GNN model not available for analysis'}), 400
            
    except Exception as e:
        logger.error(f"Error analyzing route: {e}")
        return jsonify({'error': str(e)}), 500

def find_closest_node(lat: float, lon: float) -> Optional[str]:
    """Find the closest node to given coordinates."""
    if not cached_graph:
        return None
    
    min_distance = float('inf')
    closest_node = None
    
    for node in cached_graph.nodes():
        node_lat = cached_graph.nodes[node]['lat']
        node_lon = cached_graph.nodes[node]['lon']
        
        distance = ((lat - node_lat) ** 2 + (lon - node_lon) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    
    return closest_node

def convert_route_to_geojson(route) -> Dict:
    """Convert route to GeoJSON format."""
    coordinates = []
    
    for node in route.path:
        if node in cached_graph.nodes:
            node_data = cached_graph.nodes[node]
            coordinates.append([node_data['lon'], node_data['lat']])
    
    return {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': coordinates
        },
        'properties': {
            'distance': route.total_distance,
            'time': route.total_time,
            'cost': route.total_cost,
            'algorithm': route.algorithm
        }
    }

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize system on startup
    initialize_system()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
