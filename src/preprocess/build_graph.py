"""
Graph preprocessing module for building road network graphs from OSM data.
This module processes raw OSM data into graph structures suitable for GNN training.
"""

import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple, Optional
import logging
import os
import pickle
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoadNetworkProcessor:
    """
    Processes road network data into graph structures for GNN training.
    """
    
    def __init__(self):
        """Initialize the road network processor."""
        self.graph = None
        self.node_features = None
        self.edge_features = None
        self.traffic_data = None
        
    def load_osm_data(self, filepath: str) -> gpd.GeoDataFrame:
        """
        Load OSM road network data from file.
        
        Args:
            filepath: Path to OSM data file
            
        Returns:
            GeoDataFrame with road network data
        """
        try:
            roads = gpd.read_file(filepath)
            logger.info(f"Loaded {len(roads)} road segments from {filepath}")
            return roads
        except Exception as e:
            logger.error(f"Error loading OSM data: {e}")
            return gpd.GeoDataFrame()
    
    def load_traffic_data(self, filepath: str) -> pd.DataFrame:
        """
        Load traffic data from file.
        
        Args:
            filepath: Path to traffic data file
            
        Returns:
            DataFrame with traffic data
        """
        try:
            traffic = pd.read_csv(filepath)
            logger.info(f"Loaded {len(traffic)} traffic records from {filepath}")
            return traffic
        except Exception as e:
            logger.error(f"Error loading traffic data: {e}")
            return pd.DataFrame()
    
    def build_networkx_graph(self, roads: gpd.GeoDataFrame) -> nx.Graph:
        """
        Build NetworkX graph from road network data.
        
        Args:
            roads: Road network GeoDataFrame
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes and edges
        for idx, road in roads.iterrows():
            coords = list(road.geometry.coords)
            
            # Add nodes for each coordinate
            for i, coord in enumerate(coords):
                node_id = f"{road['id']}_{i}"
                G.add_node(node_id, 
                          lat=coord[1], 
                          lon=coord[0],
                          road_id=road['id'],
                          position_in_road=i)
            
            # Add edges between consecutive coordinates
            for i in range(len(coords) - 1):
                start_node = f"{road['id']}_{i}"
                end_node = f"{road['id']}_{i+1}"
                
                # Calculate edge length
                start_point = Point(coords[i])
                end_point = Point(coords[i+1])
                length = start_point.distance(end_point) * 111000  # Convert to meters
                
                G.add_edge(start_node, end_node,
                          id=road['id'],
                          highway=road['highway'],
                          name=road['name'],
                          length=length,
                          maxspeed=road['maxspeed'],
                          lanes=road['lanes'],
                          oneway=road['oneway'],
                          surface=road['surface'])
        
        # Connect roads at intersections
        G = self._connect_intersections(G)
        
        self.graph = G
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def _connect_intersections(self, G: nx.Graph, tolerance: float = 0.0001) -> nx.Graph:
        """
        Connect roads at intersections by merging nearby nodes.
        
        Args:
            G: NetworkX graph
            tolerance: Distance tolerance for merging nodes
            
        Returns:
            Updated NetworkX graph
        """
        # Group nodes by proximity
        node_groups = {}
        processed_nodes = set()
        
        for node in G.nodes():
            if node in processed_nodes:
                continue
                
            lat, lon = G.nodes[node]['lat'], G.nodes[node]['lon']
            group_key = (round(lat / tolerance), round(lon / tolerance))
            
            if group_key not in node_groups:
                node_groups[group_key] = []
            node_groups[group_key].append(node)
            processed_nodes.add(node)
        
        # Merge nodes in each group
        for group in node_groups.values():
            if len(group) > 1:
                # Keep the first node and merge others
                main_node = group[0]
                
                for other_node in group[1:]:
                    # Move edges from other_node to main_node
                    for neighbor in list(G.neighbors(other_node)):
                        if neighbor != main_node:
                            # Copy edge attributes
                            edge_attrs = G[other_node][neighbor].copy()
                            G.add_edge(main_node, neighbor, **edge_attrs)
                    
                    # Remove the other node
                    G.remove_node(other_node)
        
        return G
    
    def extract_node_features(self, G: nx.Graph, traffic_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract node features for GNN training.
        
        Args:
            G: NetworkX graph
            traffic_data: Traffic data DataFrame
            
        Returns:
            DataFrame with node features
        """
        node_features = []
        
        for node in G.nodes():
            features = {
                'node_id': node,
                'lat': G.nodes[node]['lat'],
                'lon': G.nodes[node]['lon'],
                'degree': G.degree(node),
                'betweenness_centrality': 0,  # Will be calculated separately
                'closeness_centrality': 0,    # Will be calculated separately
                'road_type': 'unknown',
                'max_speed': 50,
                'lanes': 2,
                'is_intersection': G.degree(node) > 2,
                'traffic_volume': 0,
                'current_speed': 50,
                'jam_factor': 0
            }
            
            # Get road information from connected edges
            for neighbor in G.neighbors(node):
                edge_data = G[node][neighbor]
                if 'highway' in edge_data:
                    features['road_type'] = edge_data['highway']
                if 'maxspeed' in edge_data:
                    try:
                        features['max_speed'] = int(edge_data['maxspeed'])
                    except:
                        features['max_speed'] = 50
                if 'lanes' in edge_data:
                    try:
                        features['lanes'] = int(edge_data['lanes'])
                    except:
                        features['lanes'] = 2
                break  # Use first edge's data
            
            # Add traffic data if available
            if traffic_data is not None:
                # Find closest traffic segment
                closest_traffic = self._find_closest_traffic_segment(
                    features['lat'], features['lon'], traffic_data
                )
                if closest_traffic is not None:
                    features['traffic_volume'] = closest_traffic.get('volume', 0)
                    features['current_speed'] = closest_traffic.get('current_speed', 50)
                    features['jam_factor'] = closest_traffic.get('jam_factor', 0)
            
            node_features.append(features)
        
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        
        for features in node_features:
            node_id = features['node_id']
            features['betweenness_centrality'] = betweenness.get(node_id, 0)
            features['closeness_centrality'] = closeness.get(node_id, 0)
        
        self.node_features = pd.DataFrame(node_features)
        logger.info(f"Extracted features for {len(node_features)} nodes")
        
        return self.node_features
    
    def _find_closest_traffic_segment(self, lat: float, lon: float, traffic_data: pd.DataFrame) -> Optional[Dict]:
        """
        Find the closest traffic segment to a given coordinate.
        
        Args:
            lat: Latitude
            lon: Longitude
            traffic_data: Traffic data DataFrame
            
        Returns:
            Closest traffic segment data or None
        """
        if traffic_data.empty:
            return None
        
        # Calculate distances
        distances = np.sqrt(
            (traffic_data['latitude'] - lat)**2 + 
            (traffic_data['longitude'] - lon)**2
        )
        
        closest_idx = distances.idxmin()
        if distances[closest_idx] < 0.01:  # Within ~1km
            return traffic_data.iloc[closest_idx].to_dict()
        
        return None
    
    def extract_edge_features(self, G: nx.Graph, traffic_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract edge features for GNN training.
        
        Args:
            G: NetworkX graph
            traffic_data: Traffic data DataFrame
            
        Returns:
            DataFrame with edge features
        """
        edge_features = []
        
        for edge in G.edges(data=True):
            source, target, data = edge
            
            features = {
                'source': source,
                'target': target,
                'length': data.get('length', 100),
                'highway': data.get('highway', 'unknown'),
                'maxspeed': data.get('maxspeed', '50'),
                'lanes': data.get('lanes', '2'),
                'oneway': data.get('oneway', 'no'),
                'surface': data.get('surface', 'unknown'),
                'traffic_volume': 0,
                'current_speed': 50,
                'jam_factor': 0,
                'travel_time': 0
            }
            
            # Calculate travel time
            try:
                max_speed = int(features['maxspeed'])
                features['travel_time'] = features['length'] / (max_speed / 3.6)  # Convert to seconds
            except:
                features['travel_time'] = features['length'] / (50 / 3.6)
            
            # Add traffic data if available
            if traffic_data is not None:
                # Find traffic data for this edge
                edge_traffic = self._find_edge_traffic_data(edge, traffic_data)
                if edge_traffic is not None:
                    features['traffic_volume'] = edge_traffic.get('volume', 0)
                    features['current_speed'] = edge_traffic.get('current_speed', 50)
                    features['jam_factor'] = edge_traffic.get('jam_factor', 0)
                    
                    # Update travel time based on current speed
                    if features['current_speed'] > 0:
                        features['travel_time'] = features['length'] / (features['current_speed'] / 3.6)
            
            edge_features.append(features)
        
        self.edge_features = pd.DataFrame(edge_features)
        logger.info(f"Extracted features for {len(edge_features)} edges")
        
        return self.edge_features
    
    def _find_edge_traffic_data(self, edge: Tuple, traffic_data: pd.DataFrame) -> Optional[Dict]:
        """
        Find traffic data for a specific edge.
        
        Args:
            edge: Edge tuple (source, target, data)
            traffic_data: Traffic data DataFrame
            
        Returns:
            Traffic data for the edge or None
        """
        if traffic_data.empty:
            return None
        
        source, target, data = edge
        source_lat = self.graph.nodes[source]['lat']
        source_lon = self.graph.nodes[source]['lon']
        target_lat = self.graph.nodes[target]['lat']
        target_lon = self.graph.nodes[target]['lon']
        
        # Find traffic data near the edge midpoint
        mid_lat = (source_lat + target_lat) / 2
        mid_lon = (source_lon + target_lon) / 2
        
        distances = np.sqrt(
            (traffic_data['latitude'] - mid_lat)**2 + 
            (traffic_data['longitude'] - mid_lon)**2
        )
        
        closest_idx = distances.idxmin()
        if distances[closest_idx] < 0.005:  # Within ~500m
            return traffic_data.iloc[closest_idx].to_dict()
        
        return None
    
    def create_gnn_dataset(self, G: nx.Graph, node_features: pd.DataFrame, edge_features: pd.DataFrame) -> Dict:
        """
        Create dataset suitable for GNN training.
        
        Args:
            G: NetworkX graph
            node_features: Node features DataFrame
            edge_features: Edge features DataFrame
            
        Returns:
            Dictionary with GNN dataset
        """
        # Create node mapping
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        
        # Create edge list
        edge_list = []
        for source, target in G.edges():
            edge_list.append([node_mapping[source], node_mapping[target]])
        
        # Normalize features
        node_features_norm = self._normalize_features(node_features)
        edge_features_norm = self._normalize_features(edge_features)
        
        dataset = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'edge_list': np.array(edge_list),
            'node_features': node_features_norm.values,
            'edge_features': edge_features_norm.values,
            'node_mapping': node_mapping,
            'graph': G
        }
        
        logger.info(f"Created GNN dataset with {dataset['num_nodes']} nodes and {dataset['num_edges']} edges")
        
        return dataset
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features for GNN training.
        
        Args:
            features: Features DataFrame
            
        Returns:
            Normalized features DataFrame
        """
        normalized = features.copy()
        
        # Normalize numerical features
        numerical_cols = ['lat', 'lon', 'degree', 'betweenness_centrality', 'closeness_centrality',
                         'max_speed', 'lanes', 'traffic_volume', 'current_speed', 'jam_factor',
                         'length', 'travel_time']
        
        for col in numerical_cols:
            if col in normalized.columns:
                # Min-max normalization
                min_val = normalized[col].min()
                max_val = normalized[col].max()
                if max_val > min_val:
                    normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        
        return normalized
    
    def save_processed_data(self, dataset: Dict, filepath: str = None) -> str:
        """
        Save processed dataset to file.
        
        Args:
            dataset: Processed dataset dictionary
            filepath: Output filepath (optional)
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"processed_graph_{timestamp}.pkl"
        
        filepath = os.path.join("data", "processed", filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save without the NetworkX graph (too large)
        save_data = dataset.copy()
        save_data.pop('graph', None)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved processed dataset to {filepath}")
        
        return filepath

def main():
    """
    Example usage of RoadNetworkProcessor.
    """
    # Initialize processor
    processor = RoadNetworkProcessor()
    
    # Load data (using simulated data for demonstration)
    print("Loading OSM data...")
    # In real usage, you would load from actual files
    # roads = processor.load_osm_data("data/raw/mumbai_roads.geojson")
    # traffic = processor.load_traffic_data("data/raw/traffic_data.csv")
    
    # For demonstration, create sample data
    from src.ingest.osm_overpass import OSMDataFetcher
    from src.ingest.tomtom_fetch import TomTomTrafficFetcher
    
    osm_fetcher = OSMDataFetcher()
    roads = osm_fetcher.get_road_network()
    
    tomtom_fetcher = TomTomTrafficFetcher()
    traffic = tomtom_fetcher.get_traffic_flow_data()
    
    # Build graph
    print("Building NetworkX graph...")
    graph = processor.build_networkx_graph(roads)
    
    # Extract features
    print("Extracting node features...")
    node_features = processor.extract_node_features(graph, traffic)
    
    print("Extracting edge features...")
    edge_features = processor.extract_edge_features(graph, traffic)
    
    # Create GNN dataset
    print("Creating GNN dataset...")
    dataset = processor.create_gnn_dataset(graph, node_features, edge_features)
    
    # Save processed data
    print("Saving processed data...")
    saved_file = processor.save_processed_data(dataset)
    print(f"Processed data saved to {saved_file}")

if __name__ == "__main__":
    main()
