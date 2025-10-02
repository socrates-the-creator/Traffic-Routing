"""
OpenStreetMap data ingestion using Overpass API for Mumbai road network.
This module fetches road network data and processes it for graph construction.
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import networkx as nx
from shapely.geometry import Point, LineString
import geopandas as gpd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMDataFetcher:
    """
    Fetches OpenStreetMap data for Mumbai using Overpass API.
    """
    
    def __init__(self):
        """Initialize OSM data fetcher."""
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.mumbai_bounds = {
            'north': 19.3200,
            'south': 18.9000,
            'east': 73.2000,
            'west': 72.8000
        }
        
    def get_road_network(self, bounds: Dict = None) -> gpd.GeoDataFrame:
        """
        Fetch road network data from OpenStreetMap.
        
        Args:
            bounds: Bounding box for the area of interest
            
        Returns:
            GeoDataFrame with road network data
        """
        bounds = bounds or self.mumbai_bounds
        
        # Overpass query for road network
        query = self._build_road_query(bounds)
        
        try:
            # Make request to Overpass API
            response = requests.post(self.overpass_url, data=query, timeout=300)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            roads = self._parse_road_data(data)
            
            logger.info(f"Fetched {len(roads)} road segments from OSM")
            return roads
            
        except requests.RequestException as e:
            logger.error(f"Error fetching OSM data: {e}")
            # Return simulated data as fallback
            return self._simulate_road_network(bounds)
    
    def _build_road_query(self, bounds: Dict) -> str:
        """
        Build Overpass query for road network.
        
        Args:
            bounds: Bounding box coordinates
            
        Returns:
            Overpass query string
        """
        query = f"""
        [out:json][timeout:300];
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service)$"]
          ({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
          relation["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service)$"]
          ({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
        );
        out geom;
        """
        return query
    
    def _parse_road_data(self, data: Dict) -> gpd.GeoDataFrame:
        """
        Parse Overpass API response into GeoDataFrame.
        
        Args:
            data: JSON response from Overpass API
            
        Returns:
            GeoDataFrame with road network
        """
        roads = []
        
        for element in data.get('elements', []):
            if element['type'] == 'way':
                # Extract coordinates
                if 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    if len(coords) >= 2:
                        geometry = LineString(coords)
                        
                        # Extract road properties
                        tags = element.get('tags', {})
                        road_data = {
                            'id': element['id'],
                            'highway': tags.get('highway', 'unknown'),
                            'name': tags.get('name', 'Unnamed Road'),
                            'oneway': tags.get('oneway', 'no'),
                            'lanes': tags.get('lanes', '1'),
                            'maxspeed': tags.get('maxspeed', '50'),
                            'surface': tags.get('surface', 'unknown'),
                            'geometry': geometry,
                            'length': geometry.length * 111000,  # Approximate length in meters
                        }
                        roads.append(road_data)
        
        if roads:
            gdf = gpd.GeoDataFrame(roads, crs='EPSG:4326')
            return gdf
        else:
            # Return empty GeoDataFrame with proper structure
            return gpd.GeoDataFrame(columns=[
                'id', 'highway', 'name', 'oneway', 'lanes', 'maxspeed', 
                'surface', 'geometry', 'length'
            ], crs='EPSG:4326')
    
    def _simulate_road_network(self, bounds: Dict) -> gpd.GeoDataFrame:
        """
        Simulate road network data for demonstration purposes.
        """
        logger.info("Using simulated road network data")
        
        np.random.seed(42)
        roads = []
        
        # Generate main arterial roads
        n_arterials = 20
        for i in range(n_arterials):
            # Generate horizontal arterial
            y = np.random.uniform(bounds['south'], bounds['north'])
            x1, x2 = bounds['west'], bounds['east']
            geometry = LineString([(x1, y), (x2, y)])
            
            roads.append({
                'id': f"arterial_h_{i}",
                'highway': np.random.choice(['primary', 'secondary']),
                'name': f"Arterial Road {i}",
                'oneway': 'no',
                'lanes': str(np.random.choice([2, 4, 6])),
                'maxspeed': str(np.random.choice([50, 60, 80])),
                'surface': 'asphalt',
                'geometry': geometry,
                'length': geometry.length * 111000,
            })
            
            # Generate vertical arterial
            x = np.random.uniform(bounds['west'], bounds['east'])
            y1, y2 = bounds['south'], bounds['north']
            geometry = LineString([(x, y1), (x, y2)])
            
            roads.append({
                'id': f"arterial_v_{i}",
                'highway': np.random.choice(['primary', 'secondary']),
                'name': f"Arterial Road {i}",
                'oneway': 'no',
                'lanes': str(np.random.choice([2, 4, 6])),
                'maxspeed': str(np.random.choice([50, 60, 80])),
                'surface': 'asphalt',
                'geometry': geometry,
                'length': geometry.length * 111000,
            })
        
        # Generate local roads
        n_local = 100
        for i in range(n_local):
            # Generate random local road
            x1 = np.random.uniform(bounds['west'], bounds['east'])
            y1 = np.random.uniform(bounds['south'], bounds['north'])
            x2 = x1 + np.random.uniform(-0.01, 0.01)
            y2 = y1 + np.random.uniform(-0.01, 0.01)
            
            geometry = LineString([(x1, y1), (x2, y2)])
            
            roads.append({
                'id': f"local_{i}",
                'highway': np.random.choice(['tertiary', 'residential']),
                'name': f"Local Road {i}",
                'oneway': 'no',
                'lanes': '2',
                'maxspeed': '30',
                'surface': 'asphalt',
                'geometry': geometry,
                'length': geometry.length * 111000,
            })
        
        gdf = gpd.GeoDataFrame(roads, crs='EPSG:4326')
        return gdf
    
    def get_poi_data(self, bounds: Dict = None) -> gpd.GeoDataFrame:
        """
        Fetch Points of Interest (POI) data from OpenStreetMap.
        
        Args:
            bounds: Bounding box for the area of interest
            
        Returns:
            GeoDataFrame with POI data
        """
        bounds = bounds or self.mumbai_bounds
        
        # Simulate POI data
        pois = self._simulate_poi_data(bounds)
        
        logger.info(f"Fetched {len(pois)} points of interest")
        return pois
    
    def _simulate_poi_data(self, bounds: Dict) -> gpd.GeoDataFrame:
        """
        Simulate POI data for demonstration.
        """
        np.random.seed(42)
        pois = []
        
        poi_types = ['restaurant', 'hospital', 'school', 'bank', 'fuel', 'parking']
        n_pois = 200
        
        for i in range(n_pois):
            lat = np.random.uniform(bounds['south'], bounds['north'])
            lon = np.random.uniform(bounds['west'], bounds['east'])
            
            geometry = Point(lon, lat)
            
            pois.append({
                'id': f"poi_{i}",
                'name': f"POI {i}",
                'type': np.random.choice(poi_types),
                'geometry': geometry,
            })
        
        gdf = gpd.GeoDataFrame(pois, crs='EPSG:4326')
        return gdf
    
    def save_road_network(self, roads: gpd.GeoDataFrame, filename: str = None) -> str:
        """
        Save road network to file.
        
        Args:
            roads: Road network GeoDataFrame
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mumbai_roads_{timestamp}.geojson"
        
        filepath = os.path.join("data", "raw", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        roads.to_file(filepath, driver='GeoJSON')
        logger.info(f"Saved road network to {filepath}")
        
        return filepath
    
    def create_networkx_graph(self, roads: gpd.GeoDataFrame) -> nx.Graph:
        """
        Convert road network to NetworkX graph.
        
        Args:
            roads: Road network GeoDataFrame
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        for idx, road in roads.iterrows():
            # Add nodes (start and end points)
            coords = list(road.geometry.coords)
            start_node = coords[0]
            end_node = coords[-1]
            
            # Add nodes if they don't exist
            if start_node not in G:
                G.add_node(start_node, lat=start_node[1], lon=start_node[0])
            if end_node not in G:
                G.add_node(end_node, lat=end_node[1], lon=end_node[0])
            
            # Add edge
            G.add_edge(
                start_node, 
                end_node,
                id=road['id'],
                highway=road['highway'],
                name=road['name'],
                length=road['length'],
                maxspeed=road['maxspeed'],
                lanes=road['lanes'],
                oneway=road['oneway']
            )
        
        logger.info(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

def main():
    """
    Example usage of OSMDataFetcher.
    """
    # Initialize fetcher
    fetcher = OSMDataFetcher()
    
    # Fetch road network
    print("Fetching road network from OSM...")
    roads = fetcher.get_road_network()
    print(f"Fetched {len(roads)} road segments")
    
    # Save road network
    road_file = fetcher.save_road_network(roads)
    print(f"Road network saved to {road_file}")
    
    # Create NetworkX graph
    print("Creating NetworkX graph...")
    graph = fetcher.create_networkx_graph(roads)
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Fetch POI data
    print("Fetching POI data...")
    pois = fetcher.get_poi_data()
    print(f"Fetched {len(pois)} points of interest")
    
    # Save POI data
    poi_file = fetcher.save_road_network(pois, "mumbai_pois.geojson")
    print(f"POI data saved to {poi_file}")

if __name__ == "__main__":
    main()
