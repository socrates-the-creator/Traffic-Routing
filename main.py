"""
Main entry point for Mumbai Navigation System.
This file runs the complete system and showcases all features.
"""

import os
import sys
import time
import logging
from datetime import datetime
import webbrowser
import threading
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ingest.tomtom_fetch import TomTomTrafficFetcher
from ingest.osm_overpass import OSMDataFetcher
from preprocess.build_graph import RoadNetworkProcessor
from routing.a_star import AStarRouter, DijkstraRouter
from routing.gnn_router import GNNRouter
from routing.route_compare import RouteComparator
from utils.geo_helpers import GeoHelper, TrafficVisualizer
from utils.data_pipeline import DataPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MumbaiNavigationSystem:
    """
    Main class for the Mumbai Navigation System.
    """
    
    def __init__(self):
        """Initialize the navigation system."""
        self.system_ready = False
        self.graph = None
        self.traffic_data = None
        self.routers = {}
        self.visualizer = None
        self.geo_helper = None
        
        # Create necessary directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories."""
        directories = ['data/raw', 'data/processed', 'data/models', 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_system(self):
        """Initialize all system components."""
        print("🚀 Initializing Mumbai Navigation System...")
        print("=" * 60)
        
        try:
            # Step 1: Data Collection
            print("📊 Step 1: Collecting data...")
            self._collect_data()
            
            # Step 2: Graph Processing
            print("\n🏗️  Step 2: Building road network graph...")
            self._build_graph()
            
            # Step 3: Initialize Routers
            print("\n🗺️  Step 3: Initializing routing algorithms...")
            self._initialize_routers()
            
            # Step 4: Initialize Visualization
            print("\n📊 Step 4: Setting up visualization...")
            self._initialize_visualization()
            
            # Step 5: System Ready
            print("\n✅ Step 5: System initialization complete!")
            self.system_ready = True
            
            print("\n" + "=" * 60)
            print("🎉 Mumbai Navigation System is ready!")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            print(f"❌ System initialization failed: {e}")
            return False
        
        return True
    
    def _collect_data(self):
        """Collect initial data for the system."""
        # Check if we have existing data
        if self._check_existing_data():
            print("   ✅ Using existing data files")
            return
        
        print("   📡 Fetching fresh data...")
        
        # Fetch OSM data
        print("   🛣️  Fetching road network from OpenStreetMap...")
        osm_fetcher = OSMDataFetcher()
        roads = osm_fetcher.get_road_network()
        roads.to_file("data/raw/mumbai_roads.geojson", driver='GeoJSON')
        print(f"   ✅ Collected {len(roads)} road segments")
        
        # Fetch traffic data
        print("   🚦 Fetching traffic data from TomTom...")
        tomtom_fetcher = TomTomTrafficFetcher()
        traffic_data = tomtom_fetcher.get_traffic_flow_data()
        traffic_data.to_csv("data/raw/current_traffic.csv", index=False)
        print(f"   ✅ Collected {len(traffic_data)} traffic segments")
        
        # Fetch incident data
        print("   🚨 Fetching incident data...")
        incident_data = tomtom_fetcher.get_incident_data()
        incident_data.to_csv("data/raw/incidents.csv", index=False)
        print(f"   ✅ Found {len(incident_data)} incidents")
    
    def _check_existing_data(self):
        """Check if we have existing data files."""
        required_files = [
            "data/raw/mumbai_roads.geojson",
            "data/raw/current_traffic.csv"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                return False
        
        return True
    
    def _build_graph(self):
        """Build the road network graph."""
        # Load data
        processor = RoadNetworkProcessor()
        roads = processor.load_osm_data("data/raw/mumbai_roads.geojson")
        traffic_data = processor.load_traffic_data("data/raw/current_traffic.csv")
        
        # Build graph
        self.graph = processor.build_networkx_graph(roads)
        print(f"   ✅ Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Extract features
        node_features = processor.extract_node_features(self.graph, traffic_data)
        edge_features = processor.extract_edge_features(self.graph, traffic_data)
        
        # Create dataset
        dataset = processor.create_gnn_dataset(self.graph, node_features, edge_features)
        processor.save_processed_data(dataset, "data/processed/processed_graph.pkl")
        
        # Store traffic data
        self.traffic_data = traffic_data.to_dict('records')
        
        print(f"   ✅ Extracted features for {len(node_features)} nodes and {len(edge_features)} edges")
    
    def _initialize_routers(self):
        """Initialize routing algorithms."""
        # Convert traffic data to dictionary format
        traffic_dict = {}
        for record in self.traffic_data:
            if 'segment_id' in record:
                traffic_dict[record['segment_id']] = record
        
        # Initialize routers
        self.routers['astar'] = AStarRouter(self.graph, traffic_dict)
        self.routers['dijkstra'] = DijkstraRouter(self.graph, traffic_dict)
        
        # Try to load GNN router if model exists
        if os.path.exists("data/models/traffic_gnn_model"):
            try:
                self.routers['gnn'] = GNNRouter(self.graph, "data/models/traffic_gnn_model", 
                                              "data/processed/processed_graph.pkl", traffic_dict)
                print("   ✅ GNN router loaded successfully")
            except Exception as e:
                print(f"   ⚠️  GNN router not available: {e}")
        else:
            print("   ⚠️  GNN model not found. Run training.py to train the model.")
        
        print(f"   ✅ Initialized {len(self.routers)} routing algorithms")
    
    def _initialize_visualization(self):
        """Initialize visualization components."""
        self.visualizer = TrafficVisualizer()
        self.geo_helper = GeoHelper()
        print("   ✅ Visualization components ready")
    
    def run_demo(self):
        """Run a comprehensive demo of the system."""
        if not self.system_ready:
            print("❌ System not initialized. Please run initialize_system() first.")
            return
        
        print("\n🎯 Running System Demo")
        print("=" * 60)
        
        # Demo 1: Geographic Calculations
        self._demo_geographic_calculations()
        
        # Demo 2: Routing Algorithms
        self._demo_routing_algorithms()
        
        # Demo 3: Route Comparison
        self._demo_route_comparison()
        
        # Demo 4: Traffic Analysis
        self._demo_traffic_analysis()
        
        # Demo 5: Visualization
        self._demo_visualization()
        
        print("\n✅ Demo completed successfully!")
    
    def _demo_geographic_calculations(self):
        """Demo geographic helper functions."""
        print("\n🌍 Demo: Geographic Calculations")
        print("-" * 40)
        
        # Test distance calculation
        distance = self.geo_helper.haversine_distance(19.0760, 72.8777, 19.0176, 72.8562)
        print(f"   📏 Distance between Mumbai landmarks: {distance:.2f} meters")
        
        # Test bearing calculation
        bearing = self.geo_helper.bearing(19.0760, 72.8777, 19.0176, 72.8562)
        print(f"   🧭 Bearing: {bearing:.2f} degrees")
        
        # Test point in polygon
        mumbai_bounds = [(18.9, 72.8), (19.3, 72.8), (19.3, 73.2), (18.9, 73.2)]
        test_point = (19.0760, 72.8777)
        is_inside = self.geo_helper.point_in_polygon(test_point, mumbai_bounds)
        print(f"   🔍 Point is {'inside' if is_inside else 'outside'} Mumbai bounds")
    
    def _demo_routing_algorithms(self):
        """Demo routing algorithms."""
        print("\n🗺️  Demo: Routing Algorithms")
        print("-" * 40)
        
        # Get test nodes
        nodes = list(self.graph.nodes())
        if len(nodes) < 2:
            print("   ⚠️  Not enough nodes for routing demo")
            return
        
        start_node = nodes[0]
        end_node = nodes[-1]
        
        print(f"   📍 Testing route from {start_node} to {end_node}")
        
        # Test A* algorithm
        try:
            route = self.routers['astar'].find_route(start_node, end_node, cost_function='time')
            print(f"   🔍 A* Route: {len(route.path)} nodes, {route.total_distance:.2f}m, {route.total_time:.2f}s")
        except Exception as e:
            print(f"   ❌ A* failed: {e}")
        
        # Test Dijkstra algorithm
        try:
            route = self.routers['dijkstra'].find_route(start_node, end_node, cost_function='time')
            print(f"   🔍 Dijkstra Route: {len(route.path)} nodes, {route.total_distance:.2f}m, {route.total_time:.2f}s")
        except Exception as e:
            print(f"   ❌ Dijkstra failed: {e}")
        
        # Test GNN algorithm if available
        if 'gnn' in self.routers:
            try:
                route = self.routers['gnn'].find_route(start_node, end_node, use_gnn_predictions=False)
                print(f"   🤖 GNN Route: {len(route.path)} nodes, {route.total_distance:.2f}m, {route.total_time:.2f}s")
            except Exception as e:
                print(f"   ❌ GNN failed: {e}")
    
    def _demo_route_comparison(self):
        """Demo route comparison."""
        print("\n⚖️  Demo: Route Comparison")
        print("-" * 40)
        
        # Convert traffic data
        traffic_dict = {}
        for record in self.traffic_data:
            if 'segment_id' in record:
                traffic_dict[record['segment_id']] = record
        
        # Initialize comparator
        comparator = RouteComparator(self.graph, traffic_dict)
        
        # Get test nodes
        nodes = list(self.graph.nodes())
        if len(nodes) < 2:
            print("   ⚠️  Not enough nodes for comparison demo")
            return
        
        start_node = nodes[0]
        end_node = nodes[-1]
        
        try:
            # Compare routes
            comparison = comparator.compare_routes(start_node, end_node, include_gnn='gnn' in self.routers)
            
            print(f"   📊 Compared {len(comparison.results)} algorithms")
            if comparison.best_route:
                print(f"   🏆 Best route: {comparison.best_route.algorithm}")
                print(f"   📏 Best distance: {comparison.best_route.total_distance:.2f}m")
                print(f"   ⏱️  Best time: {comparison.best_route.total_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Route comparison failed: {e}")
    
    def _demo_traffic_analysis(self):
        """Demo traffic analysis."""
        print("\n🚦 Demo: Traffic Analysis")
        print("-" * 40)
        
        if not self.traffic_data:
            print("   ⚠️  No traffic data available")
            return
        
        # Convert to DataFrame for analysis
        import pandas as pd
        traffic_df = pd.DataFrame(self.traffic_data)
        
        # Basic statistics
        avg_speed = traffic_df['current_speed'].mean()
        avg_jam_factor = traffic_df['jam_factor'].mean()
        
        print(f"   📊 Average speed: {avg_speed:.2f} km/h")
        print(f"   📊 Average jam factor: {avg_jam_factor:.2%}")
        
        # Traffic level distribution
        traffic_levels = traffic_df['traffic_level'].value_counts()
        print("   📊 Traffic level distribution:")
        for level, count in traffic_levels.items():
            print(f"      {level}: {count} segments")
        
        # GNN predictions if available
        if 'gnn' in self.routers:
            try:
                predictions = self.routers['gnn'].predict_traffic_conditions()
                if predictions:
                    print("   🤖 GNN traffic predictions generated")
            except Exception as e:
                print(f"   ⚠️  GNN predictions not available: {e}")
    
    def _demo_visualization(self):
        """Demo visualization capabilities."""
        print("\n📊 Demo: Visualization")
        print("-" * 40)
        
        try:
            # Create sample routes
            nodes = list(self.graph.nodes())
            if len(nodes) >= 3:
                sample_routes = [
                    {
                        'path': [
                            {'lat': self.graph.nodes[nodes[0]]['lat'], 'lon': self.graph.nodes[nodes[0]]['lon']},
                            {'lat': self.graph.nodes[nodes[1]]['lat'], 'lon': self.graph.nodes[nodes[1]]['lon']},
                            {'lat': self.graph.nodes[nodes[2]]['lat'], 'lon': self.graph.nodes[nodes[2]]['lon']}
                        ],
                        'algorithm': 'A*'
                    }
                ]
                
                # Create traffic map
                traffic_df = pd.DataFrame(self.traffic_data)
                traffic_map = self.visualizer.create_traffic_map(traffic_df, sample_routes)
                traffic_map.save('demo_traffic_map.html')
                print("   🗺️  Traffic map saved to demo_traffic_map.html")
                
                # Create heatmap
                heatmap_fig = self.visualizer.create_traffic_heatmap(traffic_df, 'demo_heatmap.png')
                print("   📈 Traffic heatmap saved to demo_heatmap.png")
                
            else:
                print("   ⚠️  Not enough nodes for visualization demo")
                
        except Exception as e:
            print(f"   ❌ Visualization demo failed: {e}")
    
    def start_web_application(self):
        """Start the web application."""
        if not self.system_ready:
            print("❌ System not initialized. Please run initialize_system() first.")
            return
        
        print("\n🌐 Starting Web Application")
        print("=" * 60)
        
        # Start Flask app in a separate thread
        def run_flask_app():
            from app.app import app
            app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
        
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        print("✅ Web application started successfully!")
        print("🌐 Open your browser and go to: http://localhost:5000")
        print("📱 The interactive map will load with Mumbai's road network")
        print("\n🎯 Features available in the web interface:")
        print("   • Click on map to set start/end points")
        print("   • Compare different routing algorithms")
        print("   • View real-time traffic conditions")
        print("   • Get GNN-based traffic predictions")
        print("   • Analyze route recommendations")
        
        # Try to open browser automatically
        try:
            webbrowser.open('http://localhost:5000')
            print("🚀 Browser opened automatically")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
        
        print("\n⏹️  Press Ctrl+C to stop the web application")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Web application stopped")
    
    def run_data_pipeline(self):
        """Run the automated data pipeline."""
        print("\n🔄 Running Data Pipeline")
        print("=" * 60)
        
        try:
            pipeline = DataPipeline()
            results = pipeline.run_daily_pipeline()
            
            print("📊 Pipeline Results:")
            for step, success in results.items():
                status = "✅" if success else "❌"
                print(f"   {status} {step}")
            
            successful_steps = sum(1 for success in results.values() if success)
            total_steps = len(results)
            print(f"\n📈 Pipeline completed: {successful_steps}/{total_steps} steps successful")
            
        except Exception as e:
            print(f"❌ Data pipeline failed: {e}")
    
    def show_system_status(self):
        """Show current system status."""
        print("\n📊 System Status")
        print("=" * 60)
        
        print(f"🔄 System Ready: {'✅ Yes' if self.system_ready else '❌ No'}")
        
        if self.graph:
            print(f"🗺️  Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        if self.traffic_data:
            print(f"🚦 Traffic Data: {len(self.traffic_data)} segments")
        
        print(f"🔧 Routers: {list(self.routers.keys())}")
        
        # Check for model files
        model_files = []
        if os.path.exists("data/models/traffic_gnn_model"):
            model_files.append("GNN Model")
        if os.path.exists("data/processed/processed_graph.pkl"):
            model_files.append("Processed Graph")
        
        print(f"🤖 Models: {model_files if model_files else 'None'}")
        
        # Check data files
        data_files = []
        if os.path.exists("data/raw/mumbai_roads.geojson"):
            data_files.append("Road Network")
        if os.path.exists("data/raw/current_traffic.csv"):
            data_files.append("Traffic Data")
        if os.path.exists("data/raw/incidents.csv"):
            data_files.append("Incident Data")
        
        print(f"📁 Data Files: {data_files}")

def main():
    """Main function."""
    print("🎯 Mumbai Navigation System - Main Entry Point")
    print("=" * 60)
    
    # Initialize system
    nav_system = MumbaiNavigationSystem()
    
    # Show menu
    while True:
        print("\n📋 Choose an option:")
        print("1. 🚀 Initialize System")
        print("2. 🎯 Run Demo")
        print("3. 🌐 Start Web Application")
        print("4. 🔄 Run Data Pipeline")
        print("5. 📊 Show System Status")
        print("6. 🚪 Exit")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                nav_system.initialize_system()
            elif choice == '2':
                nav_system.run_demo()
            elif choice == '3':
                nav_system.start_web_application()
            elif choice == '4':
                nav_system.run_data_pipeline()
            elif choice == '5':
                nav_system.show_system_status()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
