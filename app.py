"""
Flask web application for the Mumbai Navigation System.
This module provides the main web interface for the navigation system.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from supabase import create_client, Client

# --- CORRECTED FILE PATH LOGIC ---
# Get the absolute path of the current file's directory
base_dir = os.path.abspath(os.path.dirname(__file__))
# Correctly append the 'src' directory to the system path
sys.path.append(os.path.join(base_dir, 'src'))

# Now, attempt the imports
try:
    from ingest.tomtom_fetch import TomTomTrafficFetcher
    from ingest.osm_overpass import OSMDataFetcher
    from preprocess.build_graph import RoadNetworkProcessor
    from models.model import TrafficGNNModel, TrafficPredictor
    from routing.a_star import AStarRouter, DijkstraRouter
    from routing.gnn_router import GNNRouter
    from routing.route_compare import RouteComparator
    MODULES_IMPORTED = True
except ImportError as e:
    print(f"Warning: Could not import project modules. Error: {e}. Using placeholder classes.")
    MODULES_IMPORTED = False

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initializing Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_for_local_testing')
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Checking if credentials are provided before creating the client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Successfully connected to Supabase.")
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")

# Global variables for caching
cached_graph = None
cached_traffic_data = None
cached_gnn_model = None
cached_gnn_dataset = None
cached_routers = {}

def initialize_system():
    """Initialize the navigation system components."""
    global cached_graph, cached_traffic_data, cached_gnn_model, cached_gnn_dataset, cached_routers
    
    if not MODULES_IMPORTED:
        logger.warning("Skipping initialization due to missing modules.")
        G = nx.Graph()
        G.add_node(1, lat=19.0760, lon=72.8777)
        G.add_node(2, lat=18.9220, lon=72.8347)
        G.add_edge(1, 2, weight=25)
        cached_graph = G
        cached_routers['astar'] = nx.astar_path
        cached_routers['dijkstra'] = nx.dijkstra_path
        return

    logger.info("Initializing navigation system...")
    try:
        if cached_graph is None: cached_graph = load_or_create_graph()
        if cached_traffic_data is None: cached_traffic_data = load_traffic_data()
        if cached_gnn_model is None or cached_gnn_dataset is None:
            cached_gnn_model, cached_gnn_dataset = load_gnn_components()
        if not cached_routers: cached_routers = initialize_routers()
        logger.info("Navigation system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing system: {e}")

def load_or_create_graph():
    graph_file = os.path.join(base_dir, "data", "processed", "mumbai_graph.pkl")
    if os.path.exists(graph_file):
        with open(graph_file, 'rb') as f: return pickle.load(f)
    else:
        return create_new_graph()

def create_new_graph():
    osm_fetcher = OSMDataFetcher()
    roads = osm_fetcher.get_road_network()
    processor = RoadNetworkProcessor()
    graph = processor.build_networkx_graph(roads)
    graph_path = os.path.join(base_dir, "data", "processed", "mumbai_graph.pkl")
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    with open(graph_path, 'wb') as f: pickle.dump(graph, f)
    return graph

def load_traffic_data():
    """Load traffic data."""
    traffic_file = os.path.join(base_dir, "data", "raw", "current_traffic.csv")
    if os.path.exists(traffic_file):
        return pd.read_csv(traffic_file).to_dict('records')
    else: return fetch_new_traffic_data()

def fetch_new_traffic_data():
    """Fetch new traffic data from TomTom."""
    fetcher = TomTomTrafficFetcher()
    traffic_data = fetcher.get_traffic_flow_data()
    traffic_path = os.path.join(base_dir, "data", "raw", "current_traffic.csv")
    os.makedirs(os.path.dirname(traffic_path), exist_ok=True)
    traffic_data.to_csv(traffic_path, index=False)
    return traffic_data.to_dict('records')

def load_gnn_components():
    """Load GNN model and dataset."""
    model_path = os.path.join(base_dir, "data", "models", "traffic_gnn_model")
    dataset_path = os.path.join(base_dir, "data", "processed", "processed_graph.pkl")
    model, dataset = None, None
    if os.path.exists(model_path) and os.path.exists(dataset_path):
        try:
            model = TrafficGNNModel(1, 1, 1); model.load_model(model_path)
            with open(dataset_path, 'rb') as f: dataset = pickle.load(f)
        except Exception as e: logger.warning(f"Could not load GNN components: {e}")
    return model, dataset

def initialize_routers():
    """Initialize routing algorithms."""
    routers = {}
    traffic_dict = {r['segment_id']: r for r in cached_traffic_data if 'segment_id' in r}
    routers['astar'] = AStarRouter(cached_graph, traffic_dict)
    routers['dijkstra'] = DijkstraRouter(cached_graph, traffic_dict)
    if cached_gnn_model and cached_gnn_dataset:
        routers['gnn'] = GNNRouter(cached_graph, None, None, traffic_dict)
        routers['gnn'].model, routers['gnn'].dataset = cached_gnn_model, cached_gnn_dataset
    return routers

@app.route('/')
def index():
    return render_template('map.html')

@app.route('/api/route', methods=['POST'])
def get_route():
    try:
        data = request.get_json()
        start_lat, start_lon = float(data['start_lat']), float(data['start_lon'])
        end_lat, end_lon = float(data['end_lat']), float(data['end_lon'])
        algorithm = data.get('algorithm', 'astar')

        # --- TYPO FIX: Changed find_closest__node to find_closest_node ---
        start_node = find_closest_node(start_lat, start_lon)
        end_node = find_closest_node(end_lat, end_lon)

        if not start_node or not end_node:
            return jsonify({'error': 'Could not find start or end point on the map'}), 400

        if not MODULES_IMPORTED:
            if algorithm in cached_routers:
                try:
                    path = cached_routers[algorithm](cached_graph, start_node, end_node)
                    route_geojson = convert_dummy_route_to_geojson(path)
                    metrics = {'distance': 25, 'time': 45, 'cost': 0, 'algorithm': algorithm}
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return jsonify({'error': f'No path found between points with {algorithm}.'}), 400
            else:
                return jsonify({'error': f'Algorithm {algorithm} not available in dummy mode.'}), 400
        else:
            router = cached_routers.get(algorithm)
            if not router:
                return jsonify({'error': f'Algorithm {algorithm} not available.'}), 400
            
            cost_function = data.get('cost_function', 'time')
            route = router.find_route(start_node, end_node, cost_function)
            if not route or not route.path:
                return jsonify({'error': f'No path found between points with {algorithm}.'}), 400
            
            route_geojson = convert_route_to_geojson(route)
            metrics = {
                'distance': route.total_distance, 'time': route.total_time,
                'cost': route.total_cost, 'algorithm': route.algorithm
            }

        if supabase:
            history_entry = {
                'start_lat': start_lat, 'start_lon': start_lon,
                'end_lat': end_lat, 'end_lon': end_lon,
                'algorithm': algorithm
            }
            try:
                supabase.table('routes').insert(history_entry).execute()
            except Exception as e:
                logger.error(f"Could not write to Supabase: {e}")

        return jsonify({'route': route_geojson, 'metrics': metrics})

    except Exception as e:
        logger.error(f"Error getting route: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/route_history')
def get_route_history():
    if not supabase:
        return jsonify([])
    try:
        response = supabase.table('routes').select("*").order('created_at', desc=True).limit(5).execute()
        # Ensure we always return a list, even if response.data is None
        history_data = response.data or []
        return jsonify(history_data)
    except Exception as e:
        logger.error(f"Could not fetch from Supabase: {e}")
        return jsonify({'error': 'Could not fetch route history'}), 500

@app.route('/api/compare_routes', methods=['POST'])
def compare_routes():
    if not MODULES_IMPORTED:
        return jsonify({'error': 'Route comparison is not available in placeholder mode.'}), 503
    try:
        data = request.get_json()
        start_lat, start_lon = float(data['start_lat']), float(data['start_lon'])
        end_lat, end_lon = float(data['end_lat']), float(data['end_lon'])
        cost_function = data.get('cost_function', 'time')
        start_node = find_closest_node(start_lat, start_lon)
        end_node = find_closest_node(end_lat, end_lon)
        if not start_node or not end_node: return jsonify({'error': 'Could not find start or end point'}), 400
        comparator = RouteComparator(cached_graph, cached_traffic_data)
        comparison = comparator.compare_routes(start_node, end_node, cost_function, include_gnn='gnn' in cached_routers)
        results = []
        for result in comparison.results:
            if result.success:
                results.append({'algorithm': result.algorithm, 'route': convert_route_to_geojson(result.route),
                                'metrics': {'distance': result.route.total_distance, 'time': result.route.total_time,
                                            'cost': result.route.total_cost, 'computation_time': result.computation_time}})
        return jsonify({'results': results, 'best_route': comparison.best_route.algorithm if comparison.best_route else None,
                        'performance_metrics': comparison.performance_metrics})
    except Exception as e:
        logger.error(f"Error comparing routes: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/route_analysis', methods=['POST'])
def analyze_route():
    if not MODULES_IMPORTED or 'gnn' not in cached_routers:
        return jsonify({'error': 'GNN model not available for analysis.'}), 503
    try:
        data = request.get_json()
        start_lat, start_lon = float(data['start_lat']), float(data['start_lon'])
        end_lat, end_lon = float(data['end_lat']), float(data['end_lon'])
        start_node = find_closest_node(start_lat, start_lon)
        end_node = find_closest_node(end_lat, end_lon)
        if not start_node or not end_node: return jsonify({'error': 'Could not find start or end point'}), 400
        gnn_router = cached_routers['gnn']
        recommendations = gnn_router.get_route_recommendations(start_node, end_node)
        return jsonify({'recommendations': recommendations, 'traffic_insights': gnn_router.get_traffic_insights()})
    except Exception as e:
        logger.error(f"Error analyzing route: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def find_closest_node(lat: float, lon: float) -> Optional[str]:
    if not cached_graph: return None
    min_dist, closest_node = float('inf'), None
    for node, data in cached_graph.nodes(data=True):
        dist = ((lat - data['lat']) ** 2 + (lon - data['lon']) ** 2) ** 0.5
        if dist < min_dist:
            min_dist, closest_node = dist, node
    return closest_node

def convert_route_to_geojson(route) -> Dict:
    coords = [[cached_graph.nodes[n]['lon'], cached_graph.nodes[n]['lat']] for n in route.path if n in cached_graph.nodes]
    return {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': coords},
            'properties': {'distance': route.total_distance, 'time': route.total_time,
                           'cost': route.total_cost, 'algorithm': route.algorithm}}

def convert_dummy_route_to_geojson(path) -> Dict:
    coords = [[cached_graph.nodes[n]['lon'], cached_graph.nodes[n]['lat']] for n in path if n in cached_graph.nodes]
    return {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': coords}, 'properties': {}}

@app.errorhandler(404)
def not_found(error):
    return render_template('map.html')

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    initialize_system()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

