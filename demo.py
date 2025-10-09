"""
Demo script for Mumbai Navigation System.
This script demonstrates the key features of the system.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ingest.tomtom_fetch import TomTomTrafficFetcher
from ingest.osm_overpass import OSMDataFetcher
from preprocess.build_graph import RoadNetworkProcessor
from routing.a_star import AStarRouter, DijkstraRouter
from routing.gnn_router import GNNRouter
from routing.route_compare import RouteComparator
from utils.geo_helpers import GeoHelper, TrafficVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_data_collection():
    """Demonstrate data collection capabilities."""
    print("🔍 Demo: Data Collection")
    print("-" * 30)
    
    # Initialize fetchers
    tomtom_fetcher = TomTomTrafficFetcher()
    osm_fetcher = OSMDataFetcher()
    
    # Fetch traffic data
    print("📊 Fetching traffic data...")
    traffic_data = tomtom_fetcher.get_traffic_flow_data()
    print(f"   Collected {len(traffic_data)} traffic segments")
    
    # Fetch incident data
    print("🚨 Fetching incident data...")
    incident_data = tomtom_fetcher.get_incident_data()
    print(f"   Found {len(incident_data)} traffic incidents")
    
    # Fetch road network
    print("🛣️  Fetching road network...")
    roads = osm_fetcher.get_road_network()
    print(f"   Collected {len(roads)} road segments")
    
    return traffic_data, incident_data, roads

def demo_graph_processing(traffic_data, roads):
    """Demonstrate graph processing capabilities."""
    print("\n🔧 Demo: Graph Processing")
    print("-" * 30)
    
    # Initialize processor
    processor = RoadNetworkProcessor()
    
    # Build graph
    print("🏗️  Building road network graph...")
    graph = processor.build_networkx_graph(roads)
    print(f"   Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Extract features
    print("📈 Extracting node features...")
    node_features = processor.extract_node_features(graph, traffic_data)
    print(f"   Extracted features for {len(node_features)} nodes")
    
    print("📈 Extracting edge features...")
    edge_features = processor.extract_edge_features(graph, traffic_data)
    print(f"   Extracted features for {len(edge_features)} edges")
    
    # Create dataset
    print("📊 Creating GNN dataset...")
    dataset = processor.create_gnn_dataset(graph, node_features, edge_features)
    print(f"   Dataset created with {dataset['num_nodes']} nodes and {dataset['num_edges']} edges")
    
    return graph, dataset

def demo_routing_algorithms(graph, traffic_data):
    """Demonstrate routing algorithms."""
    print("\n🗺️  Demo: Routing Algorithms")
    print("-" * 30)
    
    # Convert traffic data to dictionary format
    traffic_dict = {}
    for _, record in traffic_data.iterrows():
        if 'segment_id' in record:
            traffic_dict[record['segment_id']] = record.to_dict()
    
    # Initialize routers
    astar_router = AStarRouter(graph, traffic_dict)
    dijkstra_router = DijkstraRouter(graph, traffic_dict)
    
    # Get some nodes for testing
    nodes = list(graph.nodes())
    if len(nodes) >= 2:
        start_node = nodes[0]
        end_node = nodes[-1]
        
        print(f"📍 Testing route from {start_node} to {end_node}")
        
        # Test A* algorithm
        try:
            print("🔍 Testing A* algorithm...")
            astar_route = astar_router.find_route(start_node, end_node, cost_function='time')
            print(f"   A* Route: {len(astar_route.path)} nodes")
            print(f"   Distance: {astar_route.total_distance:.2f}m")
            print(f"   Time: {astar_route.total_time:.2f}s")
        except Exception as e:
            print(f"   A* failed: {e}")
        
        # Test Dijkstra algorithm
        try:
            print("🔍 Testing Dijkstra algorithm...")
            dijkstra_route = dijkstra_router.find_route(start_node, end_node, cost_function='time')
            print(f"   Dijkstra Route: {len(dijkstra_route.path)} nodes")
            print(f"   Distance: {dijkstra_route.total_distance:.2f}m")
            print(f"   Time: {dijkstra_route.total_time:.2f}s")
        except Exception as e:
            print(f"   Dijkstra failed: {e}")
    else:
        print("   Not enough nodes for routing test")

def demo_route_comparison(graph, traffic_data):
    """Demonstrate route comparison."""
    print("\n⚖️  Demo: Route Comparison")
    print("-" * 30)
    
    # Convert traffic data
    traffic_dict = {}
    for _, record in traffic_data.iterrows():
        if 'segment_id' in record:
            traffic_dict[record['segment_id']] = record.to_dict()
    
    # Initialize comparator
    comparator = RouteComparator(graph, traffic_dict)
    
    # Get test nodes
    nodes = list(graph.nodes())
    if len(nodes) >= 2:
        start_node = nodes[0]
        end_node = nodes[-1]
        
        print(f"📍 Comparing routes from {start_node} to {end_node}")
        
        try:
            # Compare routes
            comparison = comparator.compare_routes(start_node, end_node, include_gnn=False)
            
            print(f"   Compared {len(comparison.results)} algorithms")
            print(f"   Best route: {comparison.best_route.algorithm if comparison.best_route else 'None'}")
            
            # Show performance metrics
            if comparison.performance_metrics:
                metrics = comparison.performance_metrics
                print(f"   Success rate: {metrics.get('success_rate', 0):.2%}")
                
        except Exception as e:
            print(f"   Route comparison failed: {e}")
    else:
        print("   Not enough nodes for comparison test")

def demo_geographic_helpers():
    """Demonstrate geographic helper functions."""
    print("\n🌍 Demo: Geographic Helpers")
    print("-" * 30)
    
    # Initialize geo helper
    geo_helper = GeoHelper()
    
    # Test distance calculation
    print("📏 Testing distance calculation...")
    distance = geo_helper.haversine_distance(19.0760, 72.8777, 19.0176, 72.8562)
    print(f"   Distance between Mumbai points: {distance:.2f} meters")
    
    # Test bearing calculation
    print("🧭 Testing bearing calculation...")
    bearing = geo_helper.bearing(19.0760, 72.8777, 19.0176, 72.8562)
    print(f"   Bearing: {bearing:.2f} degrees")
    
    # Test point in polygon
    print("🔍 Testing point in polygon...")
    mumbai_bounds = [
        (18.9, 72.8), (19.3, 72.8), (19.3, 73.2), (18.9, 73.2), (18.9, 72.8)
    ]
    test_point = (19.0760, 72.8777)
    is_inside = geo_helper.point_in_polygon(test_point, mumbai_bounds)
    print(f"   Point {test_point} is {'inside' if is_inside else 'outside'} Mumbai bounds")

def demo_visualization(traffic_data):
    """Demonstrate visualization capabilities."""
    print("\n📊 Demo: Visualization")
    print("-" * 30)
    
    # Initialize visualizer
    visualizer = TrafficVisualizer()
    
    # Create sample routes for visualization
    sample_routes = [
        {
            'path': [
                {'lat': 19.0760, 'lon': 72.8777},
                {'lat': 19.0760, 'lon': 72.8778},
                {'lat': 19.0761, 'lon': 72.8778}
            ],
            'algorithm': 'A*'
        },
        {
            'path': [
                {'lat': 19.0760, 'lon': 72.8777},
                {'lat': 19.0761, 'lon': 72.8777},
                {'lat': 19.0761, 'lon': 72.8778}
            ],
            'algorithm': 'Dijkstra'
        }
    ]
    
    print("🗺️  Creating traffic map...")
    try:
        traffic_map = visualizer.create_traffic_map(traffic_data, sample_routes)
        traffic_map.save('demo_traffic_map.html')
        print("   Traffic map saved to demo_traffic_map.html")
    except Exception as e:
        print(f"   Map creation failed: {e}")
    
    print("📈 Creating traffic heatmap...")
    try:
        heatmap_fig = visualizer.create_traffic_heatmap(traffic_data, 'demo_heatmap.png')
        print("   Traffic heatmap saved to demo_heatmap.png")
    except Exception as e:
        print(f"   Heatmap creation failed: {e}")

def demo_system_integration():
    """Demonstrate complete system integration."""
    print("\n🔗 Demo: System Integration")
    print("-" * 30)
    
    print("🚀 Running complete system demo...")
    
    try:
        # Step 1: Data Collection
        traffic_data, incident_data, roads = demo_data_collection()
        
        # Step 2: Graph Processing
        graph, dataset = demo_graph_processing(traffic_data, roads)
        
        # Step 3: Routing
        demo_routing_algorithms(graph, traffic_data)
        
        # Step 4: Route Comparison
        demo_route_comparison(graph, traffic_data)
        
        # Step 5: Geographic Helpers
        demo_geographic_helpers()
        
        # Step 6: Visualization
        demo_visualization(traffic_data)
        
        print("\n✅ System integration demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ System integration demo failed: {e}")
        logger.exception("Demo failed")

def main():
    """Main demo function."""
    print("🎯 Mumbai Navigation System Demo")
    print("=" * 50)
    print("This demo showcases the key features of the navigation system.")
    print("=" * 50)
    
    # Check if required directories exist
    if not os.path.exists('data'):
        print("❌ Data directory not found. Please run setup.py first.")
        return
    
    # Run system integration demo
    demo_system_integration()
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n📋 Generated files:")
    print("   - demo_traffic_map.html (Interactive traffic map)")
    print("   - demo_heatmap.png (Traffic analysis chart)")
    print("\n🚀 To run the full web application:")
    print("   python app/app.py")
    print("   Then open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()
