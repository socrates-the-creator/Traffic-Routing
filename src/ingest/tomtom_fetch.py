"""
TomTom API integration for fetching real-time traffic data for Mumbai.
This module handles authentication, data fetching, and preprocessing of traffic data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TomTomTrafficFetcher:
    """
    Fakes TomTom API integration for traffic data fetching.
    In a real implementation, you would use actual TomTom API endpoints.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize TomTom API client.
        
        Args:
            api_key: TomTom API key (optional, can be set via environment variable)
        """
        self.api_key = api_key or os.getenv('TOMTOM_API_KEY', 'demo_key')
        self.base_url = "https://api.tomtom.com"
        self.mumbai_bounds = {
            'north': 19.3200,
            'south': 18.9000,
            'east': 73.2000,
            'west': 72.8000
        }
        
    def get_traffic_flow_data(self, bounds: Dict = None) -> pd.DataFrame:
        """
        Fetch real-time traffic flow data for Mumbai.
        
        Args:
            bounds: Bounding box for the area of interest
            
        Returns:
            DataFrame with traffic flow data
        """
        bounds = bounds or self.mumbai_bounds
        
        # Simulate traffic data (in real implementation, this would be API calls)
        traffic_data = self._simulate_traffic_data(bounds)
        
        logger.info(f"Fetched traffic data for {len(traffic_data)} road segments")
        return traffic_data
    
    def _simulate_traffic_data(self, bounds: Dict) -> pd.DataFrame:
        """
        Simulate traffic data for demonstration purposes.
        In production, this would be replaced with actual TomTom API calls.
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate synthetic traffic data
        n_segments = 1000
        data = []
        
        for i in range(n_segments):
            # Generate random coordinates within Mumbai bounds
            lat = np.random.uniform(bounds['south'], bounds['north'])
            lon = np.random.uniform(bounds['west'], bounds['east'])
            
            # Generate traffic metrics
            current_speed = np.random.uniform(5, 60)  # km/h
            free_flow_speed = np.random.uniform(40, 80)  # km/h
            jam_factor = 1 - (current_speed / free_flow_speed)
            
            # Traffic level classification
            if jam_factor < 0.3:
                traffic_level = 'FLOWING'
            elif jam_factor < 0.6:
                traffic_level = 'SLOW'
            else:
                traffic_level = 'CONGESTED'
            
            data.append({
                'segment_id': f"segment_{i}",
                'latitude': lat,
                'longitude': lon,
                'current_speed': current_speed,
                'free_flow_speed': free_flow_speed,
                'jam_factor': jam_factor,
                'traffic_level': traffic_level,
                'confidence': np.random.uniform(0.7, 1.0),
                'timestamp': datetime.now().isoformat(),
                'road_type': np.random.choice(['HIGHWAY', 'ARTERIAL', 'LOCAL']),
                'length': np.random.uniform(100, 2000)  # meters
            })
        
        return pd.DataFrame(data)
    
    def get_incident_data(self, bounds: Dict = None) -> pd.DataFrame:
        """
        Fetch traffic incident data for Mumbai.
        
        Args:
            bounds: Bounding box for the area of interest
            
        Returns:
            DataFrame with incident data
        """
        bounds = bounds or self.mumbai_bounds
        
        # Simulate incident data
        incidents = self._simulate_incident_data(bounds)
        
        logger.info(f"Fetched {len(incidents)} traffic incidents")
        return incidents
    
    def _simulate_incident_data(self, bounds: Dict) -> pd.DataFrame:
        """
        Simulate traffic incident data.
        """
        np.random.seed(42)
        
        n_incidents = np.random.poisson(15)  # Average 15 incidents
        data = []
        
        incident_types = ['ACCIDENT', 'CONSTRUCTION', 'ROAD_CLOSURE', 'HAZARD']
        
        for i in range(n_incidents):
            lat = np.random.uniform(bounds['south'], bounds['north'])
            lon = np.random.uniform(bounds['west'], bounds['east'])
            
            data.append({
                'incident_id': f"incident_{i}",
                'latitude': lat,
                'longitude': lon,
                'type': np.random.choice(incident_types),
                'severity': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'description': f"Traffic incident at {lat:.4f}, {lon:.4f}",
                'start_time': datetime.now().isoformat(),
                'end_time': (datetime.now() + timedelta(hours=np.random.uniform(1, 6))).isoformat(),
                'impact': np.random.choice(['MINOR', 'MODERATE', 'MAJOR'])
            })
        
        return pd.DataFrame(data)
    
    def get_historical_traffic_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical traffic data for model training.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical traffic data
        """
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        
        # Simulate historical data
        historical_data = self._simulate_historical_data(start_date, end_date)
        
        return historical_data
    
    def _simulate_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Simulate historical traffic data for training.
        """
        np.random.seed(42)
        
        # Generate data for each day in the range
        current_date = start_date
        all_data = []
        
        while current_date <= end_date:
            # Generate hourly data for each day
            for hour in range(24):
                # Simulate daily traffic patterns
                base_speed = 50
                rush_hour_factor = 1.0
                
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                    rush_hour_factor = 0.4
                elif 10 <= hour <= 16:  # Daytime
                    rush_hour_factor = 0.7
                elif 20 <= hour <= 22:  # Evening
                    rush_hour_factor = 0.6
                else:  # Night
                    rush_hour_factor = 0.9
                
                # Generate data for multiple road segments
                for segment_id in range(100):  # 100 road segments
                    speed = base_speed * rush_hour_factor * np.random.uniform(0.8, 1.2)
                    
                    all_data.append({
                        'timestamp': current_date.replace(hour=hour, minute=0, second=0),
                        'segment_id': f"segment_{segment_id}",
                        'speed': speed,
                        'volume': np.random.poisson(100 * rush_hour_factor),
                        'day_of_week': current_date.weekday(),
                        'hour': hour,
                        'is_weekend': current_date.weekday() >= 5
                    })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(all_data)
    
    def save_traffic_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Save traffic data to file.
        
        Args:
            data: Traffic data DataFrame
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_data_{timestamp}.csv"
        
        filepath = os.path.join("data", "raw", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data.to_csv(filepath, index=False)
        logger.info(f"Saved traffic data to {filepath}")
        
        return filepath

def main():
    """
    Example usage of TomTomTrafficFetcher.
    """
    # Initialize fetcher
    fetcher = TomTomTrafficFetcher()
    
    # Fetch current traffic data
    print("Fetching current traffic data...")
    traffic_data = fetcher.get_traffic_flow_data()
    print(f"Fetched {len(traffic_data)} traffic segments")
    
    # Fetch incident data
    print("Fetching incident data...")
    incident_data = fetcher.get_incident_data()
    print(f"Fetched {len(incident_data)} incidents")
    
    # Save data
    traffic_file = fetcher.save_traffic_data(traffic_data)
    incident_file = fetcher.save_traffic_data(incident_data, "incidents.csv")
    
    print(f"Data saved to {traffic_file} and {incident_file}")
    
    # Fetch historical data for training
    print("Fetching historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    historical_data = fetcher.get_historical_traffic_data(start_date, end_date)
    print(f"Fetched {len(historical_data)} historical records")
    
    # Save historical data
    hist_file = fetcher.save_traffic_data(historical_data, "historical_traffic.csv")
    print(f"Historical data saved to {hist_file}")

if __name__ == "__main__":
    main()
