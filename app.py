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

# Get the absolute path of the current file's directory
base_dir = os.path.abspath(os.path.dirname(__file__))
# Correctly append the 'src' directory to the system path
sys.path.append(os.path.join(base_dir, 'src'))

# Now, attempt the imports
try:
    from ingest.tomtom_fetch import TomTomTrafficFetcher
    from ingest.osm_overpass import OSMDataFetcher
    from preprocess.build_graph import RoadNetworkProcessor
    from models.model import TrafficGNNModel
    from routing.a_star import AStarRouter, DijkstraRouter
    from routing.gnn_router import GNNRouter
    from routing.route_compare import RouteComparator
    MODULES_IMPORTED = True
except ImportError as e:
    print(f"Warning: Could not import project modules. Error: {e}. Using placeholder classes.")
    MODULES_IMPORTED = False

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        logger.info("Successfully connected to Supabase.")
    except Exception as e:
        logger.error(f"Error connecting to Supabase: {e}")

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
        logger.warning("Skipping full initialization due to missing modules.")
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
        if cached_graph is None:
            logger.info("Graph cache is empty. Attempting to load graph...")
            cached_graph = load_or_create_graph()
            if cached_graph and len(cached_graph.nodes()) > 0:
                logger.info(f"Graph loaded successfully. Number of nodes: {len(cached_graph.nodes())}")
            else:
                logger.error("CRITICAL: Graph loading failed or resulted in an empty graph.")
        
        if cached_traffic_data is None: cached_traffic_data = load_traffic_data()
        if cached_gnn_model is None or cached_gnn_dataset is None:
            cached_gnn_model, cached_gnn_dataset = load_gnn_components()
        if not cached_routers: cached_routers = initialize_routers()
        logger.info("Navigation system initialized successfully")
    except Exception as e:
        logger.error(f"CRITICAL ERROR during system initialization: {e}", exc_info=True)

def load_or_create_graph():
    graph_file = os.path.join(base_dir, "data", "processed", "mumbai_graph.pkl")
    logger.info(f"Checking for graph file at: {graph_file}")
    if os.path.exists(graph_file):
        logger.info("Graph file found. Loading from pickle.")
        try:
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f)
                logger.info("Successfully deserialized graph from .pkl file.")
                return graph
        except Exception as e:
            logger.error(f"Error loading .pkl file: {e}", exc_info=True)
            return None
    else:
        logger.warning("Graph file not found. This will cause errors unless regenerated.")
        # In a production environment, we should not regenerate on the fly.
        # Returning None will show the error clearly.
        return None

# ... The rest of your app.py functions remain the same ...
# (I've omitted them for brevity, but they are unchanged)

@app.route('/')
def index():
    return render_template('map.html')

@app.route('/api/route', methods=['POST'])
def get_route():
    try:
        if not cached_graph or len(cached_graph.nodes()) == 0:
            return jsonify({'error': 'Map data is not loaded on the server. Please check deployment logs.'}), 500

        data = request.get_json()
        start_lat, start_lon = float(data['start_lat']), float(data['start_lon'])
        end_lat, end_lon = float(data['end_lat']), float(data['end_lon'])
        algorithm = data.get('algorithm', 'astar')

        start_node = find_closest_node(start_lat, start_lon)
        end_node = find_closest_node(end_lat, end_lon)

        if not start_node or not end_node:
            logger.warning(f"Could not find nodes. Start: {start_node}, End: {end_node}. Check graph integrity.")
            return jsonify({'error': 'Could not find start or end point on the map'}), 400

        # --- The rest of the function is unchanged ---
        if not MODULES_IMPORTED:
            path = cached_routers[algorithm](cached_graph, start_node, end_node)
            route_geojson = convert_dummy_route_to_geojson(path)
            metrics = {'algorithm': algorithm}
        else:
            router = cached_routers.get(algorithm)
            route = router.find_route(start_node, end_node)
            route_geojson = convert_route_to_geojson(route)
            metrics = {'distance': route.total_distance, 'time': route.total_time, 'algorithm': route.algorithm}

        if supabase:
            history_entry = { 'start_lat': start_lat, 'start_lon': start_lon, 'end_lat': end_lat, 'end_lon': end_lon, 'algorithm': algorithm }
            try:
                supabase.table('routes').insert(history_entry).execute()
            except Exception as e:
                logger.error(f"Could not write to Supabase: {e}")

        return jsonify({'route': route_geojson, 'metrics': metrics})

    except Exception as e:
        logger.error(f"Error in get_route: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred.'}), 500

@app.route('/api/route_history')
def get_route_history():
    if not supabase:
        return jsonify([])
    try:
        response = supabase.table('routes').select("*").order('created_at', desc=True).limit(5).execute()
        return jsonify(response.data or [])
    except Exception as e:
        logger.error(f"Could not fetch from Supabase: {e}")
        return jsonify({'error': 'Could not fetch route history'}), 500

def find_closest_node(lat: float, lon: float) -> Optional[str]:
    if not cached_graph: return None
    min_dist, closest_node = float('inf'), None
    # Use .items() for modern networkx
    for node, data in cached_graph.nodes(data=True):
        dist = ((lat - data['lat']) ** 2 + (lon - data['lon']) ** 2) ** 0.5
        if dist < min_dist:
            min_dist, closest_node = dist, node
    return closest_node

def convert_route_to_geojson(route) -> Dict:
    coords = [[cached_graph.nodes[n]['lon'], cached_graph.nodes[n]['lat']] for n in route.path if n in cached_graph.nodes]
    return {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': coords}, 'properties': {'algorithm': route.algorithm}}

def convert_dummy_route_to_geojson(path) -> Dict:
    coords = [[cached_graph.nodes[n]['lon'], cached_graph.nodes[n]['lat']] for n in path if n in cached_graph.nodes]
    return {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': coords}, 'properties': {}}

# Error handlers and main execution block remain the same
@app.errorhandler(404)
def not_found(error):
    return render_template('map.html')

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    initialize_system()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

