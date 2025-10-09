"""
Geographic helper functions for the Mumbai Navigation System.
This module provides utilities for geographic calculations and visualizations.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from typing import List, Tuple, Dict, Optional
import logging
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoHelper:
    """
    Geographic helper class for calculations and visualizations.
    """
    
    def __init__(self):
        """Initialize GeoHelper."""
        self.earth_radius = 6371000  # Earth's radius in meters
        
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Haversine distance between two points.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return self.earth_radius * c
    
    def bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate bearing between two points.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Bearing in degrees
        """
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lon = np.radians(lon2 - lon1)
        
        y = np.sin(delta_lon) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) - 
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon))
        
        bearing_rad = np.arctan2(y, x)
        bearing_deg = np.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside a polygon.
        
        Args:
            point: (lat, lon) tuple
            polygon: List of (lat, lon) tuples forming the polygon
            
        Returns:
            True if point is inside polygon
        """
        lat, lon = point
        n = len(polygon)
        inside = False
        
        p1_lat, p1_lon = polygon[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = polygon[i % n]
            if lat > min(p1_lat, p2_lat):
                if lat <= max(p1_lat, p2_lat):
                    if lon <= max(p1_lon, p2_lon):
                        if p1_lat != p2_lat:
                            xinters = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                        if p1_lon == p2_lon or lon <= xinters:
                            inside = not inside
            p1_lat, p1_lon = p2_lat, p2_lon
        
        return inside
    
    def create_bounding_box(self, center_lat: float, center_lon: float, 
                          width_km: float, height_km: float) -> List[Tuple[float, float]]:
        """
        Create a bounding box around a center point.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            width_km: Width in kilometers
            height_km: Height in kilometers
            
        Returns:
            List of (lat, lon) tuples forming the bounding box
        """
        # Convert km to degrees (approximate)
        lat_offset = height_km / 111.0  # 1 degree latitude ≈ 111 km
        lon_offset = width_km / (111.0 * np.cos(np.radians(center_lat)))
        
        return [
            (center_lat - lat_offset, center_lon - lon_offset),  # SW
            (center_lat + lat_offset, center_lon - lon_offset),  # NW
            (center_lat + lat_offset, center_lon + lon_offset),  # NE
            (center_lat - lat_offset, center_lon + lon_offset),  # SE
            (center_lat - lat_offset, center_lon - lon_offset)   # Close polygon
        ]
    
    def interpolate_points(self, start: Tuple[float, float], end: Tuple[float, float], 
                         num_points: int) -> List[Tuple[float, float]]:
        """
        Interpolate points between two coordinates.
        
        Args:
            start: Start point (lat, lon)
            end: End point (lat, lon)
            num_points: Number of points to interpolate
            
        Returns:
            List of interpolated points
        """
        start_lat, start_lon = start
        end_lat, end_lon = end
        
        lats = np.linspace(start_lat, end_lat, num_points)
        lons = np.linspace(start_lon, end_lon, num_points)
        
        return list(zip(lats, lons))

class TrafficVisualizer:
    """
    Traffic visualization class using Folium and Matplotlib.
    """
    
    def __init__(self, center_lat: float = 19.0760, center_lon: float = 72.8777):
        """
        Initialize traffic visualizer.
        
        Args:
            center_lat: Center latitude for maps
            center_lon: Center longitude for maps
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.geo_helper = GeoHelper()
        
    def create_traffic_map(self, traffic_data: pd.DataFrame, 
                          routes: List[Dict] = None,
                          predictions: pd.DataFrame = None) -> folium.Map:
        """
        Create a Folium map with traffic visualization.
        
        Args:
            traffic_data: Traffic data DataFrame
            routes: List of route dictionaries
            predictions: Traffic predictions DataFrame
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add traffic data
        if not traffic_data.empty:
            self._add_traffic_layer(m, traffic_data)
        
        # Add routes
        if routes:
            self._add_routes_layer(m, routes)
        
        # Add predictions
        if predictions is not None and not predictions.empty:
            self._add_predictions_layer(m, predictions)
        
        # Add legend
        self._add_traffic_legend(m)
        
        return m
    
    def _add_traffic_layer(self, map_obj: folium.Map, traffic_data: pd.DataFrame):
        """Add traffic data layer to map."""
        for _, row in traffic_data.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            
            # Determine color based on traffic level
            color = self._get_traffic_color(row.get('traffic_level', 'UNKNOWN'))
            
            # Create popup text
            popup_text = f"""
            <b>Traffic Information</b><br>
            Speed: {row.get('current_speed', 'N/A')} km/h<br>
            Level: {row.get('traffic_level', 'N/A')}<br>
            Jam Factor: {row.get('jam_factor', 0):.1%}<br>
            Confidence: {row.get('confidence', 0):.1%}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=folium.Popup(popup_text, max_width=200),
                color='white',
                weight=2,
                fillColor=color,
                fillOpacity=0.8
            ).add_to(map_obj)
    
    def _add_routes_layer(self, map_obj: folium.Map, routes: List[Dict]):
        """Add routes layer to map."""
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
        
        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            
            # Convert route coordinates
            coordinates = []
            for node in route.get('path', []):
                if isinstance(node, dict) and 'lat' in node and 'lon' in node:
                    coordinates.append([node['lat'], node['lon']])
                elif isinstance(node, tuple) and len(node) == 2:
                    coordinates.append([node[0], node[1]])
            
            if coordinates:
                # Add route line
                folium.PolyLine(
                    locations=coordinates,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    popup=f"Route {i+1}: {route.get('algorithm', 'Unknown')}"
                ).add_to(map_obj)
                
                # Add start and end markers
                if len(coordinates) >= 2:
                    folium.Marker(
                        coordinates[0],
                        popup=f"Start - Route {i+1}",
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(map_obj)
                    
                    folium.Marker(
                        coordinates[-1],
                        popup=f"End - Route {i+1}",
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(map_obj)
    
    def _add_predictions_layer(self, map_obj: folium.Map, predictions: pd.DataFrame):
        """Add traffic predictions layer to map."""
        for _, row in predictions.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            
            # Determine color based on prediction
            predicted_speed = row.get('predicted_speed', 50)
            if predicted_speed > 40:
                color = '#28a745'  # Green
            elif predicted_speed > 20:
                color = '#ffc107'  # Yellow
            else:
                color = '#dc3545'  # Red
            
            # Create popup text
            popup_text = f"""
            <b>GNN Prediction</b><br>
            Predicted Speed: {predicted_speed:.1f} km/h<br>
            Confidence: {row.get('confidence', 0):.1%}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                popup=folium.Popup(popup_text, max_width=200),
                color='white',
                weight=2,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(map_obj)
    
    def _get_traffic_color(self, traffic_level: str) -> str:
        """Get color for traffic level."""
        color_map = {
            'FLOWING': '#28a745',    # Green
            'SLOW': '#ffc107',       # Yellow
            'CONGESTED': '#dc3545',  # Red
            'UNKNOWN': '#6c757d'     # Gray
        }
        return color_map.get(traffic_level, '#6c757d')
    
    def _add_traffic_legend(self, map_obj: folium.Map):
        """Add traffic legend to map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Traffic Conditions</b></p>
        <p><i class="fa fa-circle" style="color:#28a745"></i> Flowing</p>
        <p><i class="fa fa-circle" style="color:#ffc107"></i> Slow</p>
        <p><i class="fa fa-circle" style="color:#dc3545"></i> Congested</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def create_traffic_heatmap(self, traffic_data: pd.DataFrame, 
                              save_path: str = None) -> plt.Figure:
        """
        Create a traffic heatmap visualization.
        
        Args:
            traffic_data: Traffic data DataFrame
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Speed distribution
        axes[0, 0].hist(traffic_data['current_speed'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Speed Distribution')
        axes[0, 0].set_xlabel('Speed (km/h)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Jam factor distribution
        axes[0, 1].hist(traffic_data['jam_factor'], bins=30, alpha=0.7, color='orange')
        axes[0, 1].set_title('Jam Factor Distribution')
        axes[0, 1].set_xlabel('Jam Factor')
        axes[0, 1].set_ylabel('Frequency')
        
        # Traffic level counts
        traffic_counts = traffic_data['traffic_level'].value_counts()
        axes[1, 0].pie(traffic_counts.values, labels=traffic_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Traffic Level Distribution')
        
        # Speed vs Jam Factor scatter
        scatter = axes[1, 1].scatter(traffic_data['current_speed'], traffic_data['jam_factor'], 
                                   c=traffic_data['confidence'], cmap='viridis', alpha=0.6)
        axes[1, 1].set_title('Speed vs Jam Factor')
        axes[1, 1].set_xlabel('Speed (km/h)')
        axes[1, 1].set_ylabel('Jam Factor')
        plt.colorbar(scatter, ax=axes[1, 1], label='Confidence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Traffic heatmap saved to {save_path}")
        
        return fig
    
    def create_route_comparison_chart(self, comparison_results: List[Dict], 
                                    save_path: str = None) -> plt.Figure:
        """
        Create route comparison visualization.
        
        Args:
            comparison_results: List of route comparison results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        algorithms = [r['algorithm'] for r in comparison_results]
        distances = [r['metrics']['distance'] for r in comparison_results]
        times = [r['metrics']['time'] for r in comparison_results]
        costs = [r['metrics']['cost'] for r in comparison_results]
        computation_times = [r['metrics']['computation_time'] for r in comparison_results]
        
        # Distance comparison
        axes[0, 0].bar(algorithms, distances, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Route Distance Comparison')
        axes[0, 0].set_ylabel('Distance (m)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Time comparison
        axes[0, 1].bar(algorithms, times, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Route Time Comparison')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cost comparison
        axes[1, 0].bar(algorithms, costs, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Route Cost Comparison')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Computation time comparison
        axes[1, 1].bar(algorithms, computation_times, color='gold', alpha=0.7)
        axes[1, 1].set_title('Computation Time Comparison')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Route comparison chart saved to {save_path}")
        
        return fig
    
    def create_traffic_timeline(self, historical_data: pd.DataFrame, 
                              save_path: str = None) -> plt.Figure:
        """
        Create traffic timeline visualization.
        
        Args:
            historical_data: Historical traffic data
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Convert timestamp to datetime
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        
        # Group by hour
        hourly_data = historical_data.groupby(historical_data['timestamp'].dt.hour).agg({
            'speed': 'mean',
            'volume': 'mean',
            'jam_factor': 'mean'
        }).reset_index()
        
        # Speed over time
        axes[0].plot(hourly_data['timestamp'], hourly_data['speed'], marker='o', linewidth=2)
        axes[0].set_title('Average Speed by Hour')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Speed (km/h)')
        axes[0].grid(True, alpha=0.3)
        
        # Volume over time
        axes[1].plot(hourly_data['timestamp'], hourly_data['volume'], marker='s', linewidth=2, color='orange')
        axes[1].set_title('Average Volume by Hour')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Traffic timeline saved to {save_path}")
        
        return fig

def main():
    """
    Example usage of GeoHelper and TrafficVisualizer.
    """
    # Initialize helpers
    geo_helper = GeoHelper()
    visualizer = TrafficVisualizer()
    
    # Test geographic calculations
    print("Testing geographic calculations...")
    
    # Distance between two points in Mumbai
    distance = geo_helper.haversine_distance(19.0760, 72.8777, 19.0176, 72.8562)
    print(f"Distance between two Mumbai points: {distance:.2f} meters")
    
    # Bearing calculation
    bearing = geo_helper.bearing(19.0760, 72.8777, 19.0176, 72.8562)
    print(f"Bearing: {bearing:.2f} degrees")
    
    # Create sample traffic data
    print("\nCreating sample traffic data...")
    np.random.seed(42)
    
    traffic_data = pd.DataFrame({
        'latitude': np.random.uniform(18.9, 19.3, 100),
        'longitude': np.random.uniform(72.8, 73.2, 100),
        'current_speed': np.random.uniform(10, 60, 100),
        'jam_factor': np.random.uniform(0, 1, 100),
        'traffic_level': np.random.choice(['FLOWING', 'SLOW', 'CONGESTED'], 100),
        'confidence': np.random.uniform(0.7, 1.0, 100)
    })
    
    # Create traffic map
    print("Creating traffic map...")
    traffic_map = visualizer.create_traffic_map(traffic_data)
    traffic_map.save('traffic_map.html')
    print("Traffic map saved to traffic_map.html")
    
    # Create traffic heatmap
    print("Creating traffic heatmap...")
    heatmap_fig = visualizer.create_traffic_heatmap(traffic_data, 'traffic_heatmap.png')
    plt.show()
    
    print("GeoHelper and TrafficVisualizer examples completed!")

if __name__ == "__main__":
    main()
