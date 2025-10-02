"""
Daily data pipeline for the Mumbai Navigation System.
This module handles automated data collection, processing, and model updates.
"""

import os
import sys
import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pickle
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ingest.tomtom_fetch import TomTomTrafficFetcher
from ingest.osm_overpass import OSMDataFetcher
from preprocess.build_graph import RoadNetworkProcessor
from models.train import ModelTrainer
from models.model import TrafficGNNModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Daily data pipeline for automated data collection and processing.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize data pipeline.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.tomtom_fetcher = TomTomTrafficFetcher()
        self.osm_fetcher = OSMDataFetcher()
        self.processor = RoadNetworkProcessor()
        self.model_trainer = None
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize model trainer
        self._initialize_model_trainer()
        
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration."""
        return {
            'data_collection': {
                'traffic_update_interval': 15,  # minutes
                'osm_update_interval': 24,      # hours
                'historical_data_days': 7,      # days
                'retention_days': 30            # days
            },
            'model_training': {
                'retrain_interval': 24,         # hours
                'min_data_points': 1000,
                'validation_split': 0.2,
                'epochs': 50
            },
            'storage': {
                'data_dir': 'data',
                'models_dir': 'data/models',
                'logs_dir': 'logs',
                'backup_dir': 'data/backup'
            },
            'notifications': {
                'email_alerts': False,
                'log_level': 'INFO'
            }
        }
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.config['storage']['data_dir'],
            self.config['storage']['models_dir'],
            self.config['storage']['logs_dir'],
            self.config['storage']['backup_dir'],
            'data/raw',
            'data/processed',
            'data/models',
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_model_trainer(self):
        """Initialize model trainer."""
        try:
            self.model_trainer = ModelTrainer(self.config.get('model_training', {}))
            logger.info("Model trainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model trainer: {e}")
            self.model_trainer = None
    
    def collect_traffic_data(self) -> bool:
        """
        Collect current traffic data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Collecting traffic data...")
            
            # Fetch current traffic data
            traffic_data = self.tomtom_fetcher.get_traffic_flow_data()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_data_{timestamp}.csv"
            filepath = os.path.join("data", "raw", filename)
            
            traffic_data.to_csv(filepath, index=False)
            
            # Also save as current traffic data
            current_filepath = os.path.join("data", "raw", "current_traffic.csv")
            traffic_data.to_csv(current_filepath, index=False)
            
            logger.info(f"Traffic data collected and saved to {filepath}")
            
            # Update graph with new traffic data
            self._update_graph_traffic_data(traffic_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting traffic data: {e}")
            return False
    
    def collect_incident_data(self) -> bool:
        """
        Collect traffic incident data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Collecting incident data...")
            
            # Fetch incident data
            incident_data = self.tomtom_fetcher.get_incident_data()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"incidents_{timestamp}.csv"
            filepath = os.path.join("data", "raw", filename)
            
            incident_data.to_csv(filepath, index=False)
            
            logger.info(f"Incident data collected and saved to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting incident data: {e}")
            return False
    
    def collect_historical_data(self) -> bool:
        """
        Collect historical traffic data for model training.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Collecting historical traffic data...")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['data_collection']['historical_data_days'])
            
            # Fetch historical data
            historical_data = self.tomtom_fetcher.get_historical_traffic_data(start_date, end_date)
            
            # Save historical data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historical_traffic_{timestamp}.csv"
            filepath = os.path.join("data", "raw", filename)
            
            historical_data.to_csv(filepath, index=False)
            
            logger.info(f"Historical data collected and saved to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            return False
    
    def update_osm_data(self) -> bool:
        """
        Update OpenStreetMap road network data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Updating OSM road network data...")
            
            # Fetch road network
            roads = self.osm_fetcher.get_road_network()
            
            # Save road network
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mumbai_roads_{timestamp}.geojson"
            filepath = os.path.join("data", "raw", filename)
            
            roads.to_file(filepath, driver='GeoJSON')
            
            # Also save as current road network
            current_filepath = os.path.join("data", "raw", "mumbai_roads.geojson")
            roads.to_file(current_filepath, driver='GeoJSON')
            
            logger.info(f"OSM data updated and saved to {filepath}")
            
            # Rebuild graph
            self._rebuild_graph(roads)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating OSM data: {e}")
            return False
    
    def _update_graph_traffic_data(self, traffic_data: pd.DataFrame):
        """Update graph with new traffic data."""
        try:
            # Load existing graph
            graph_file = "data/processed/mumbai_graph.pkl"
            if os.path.exists(graph_file):
                with open(graph_file, 'rb') as f:
                    graph = pickle.load(f)
                
                # Update traffic data in graph
                # This would involve updating edge weights based on traffic conditions
                logger.info("Graph traffic data updated")
            else:
                logger.warning("Graph file not found, skipping traffic update")
                
        except Exception as e:
            logger.error(f"Error updating graph traffic data: {e}")
    
    def _rebuild_graph(self, roads):
        """Rebuild graph from road network data."""
        try:
            logger.info("Rebuilding graph from road network...")
            
            # Build new graph
            graph = self.processor.build_networkx_graph(roads)
            
            # Save graph
            graph_file = "data/processed/mumbai_graph.pkl"
            with open(graph_file, 'wb') as f:
                pickle.dump(graph, f)
            
            logger.info("Graph rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding graph: {e}")
    
    def process_data(self) -> bool:
        """
        Process raw data into training-ready format.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Processing data for model training...")
            
            # Load road network
            roads_file = "data/raw/mumbai_roads.geojson"
            if not os.path.exists(roads_file):
                logger.error("Road network file not found")
                return False
            
            roads = self.processor.load_osm_data(roads_file)
            
            # Load traffic data
            traffic_file = "data/raw/current_traffic.csv"
            if not os.path.exists(traffic_file):
                logger.error("Traffic data file not found")
                return False
            
            traffic = self.processor.load_traffic_data(traffic_file)
            
            # Build graph
            graph = self.processor.build_networkx_graph(roads)
            
            # Extract features
            node_features = self.processor.extract_node_features(graph, traffic)
            edge_features = self.processor.extract_edge_features(graph, traffic)
            
            # Create dataset
            dataset = self.processor.create_gnn_dataset(graph, node_features, edge_features)
            
            # Save processed dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_graph_{timestamp}.pkl"
            filepath = os.path.join("data", "processed", filename)
            
            self.processor.save_processed_data(dataset, filepath)
            
            # Also save as current processed dataset
            current_filepath = "data/processed/processed_graph.pkl"
            self.processor.save_processed_data(dataset, current_filepath)
            
            logger.info(f"Data processed and saved to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return False
    
    def train_model(self) -> bool:
        """
        Train the GNN model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.model_trainer:
                logger.error("Model trainer not initialized")
                return False
            
            logger.info("Training GNN model...")
            
            # Check if we have enough data
            if not self._check_training_data():
                logger.warning("Insufficient data for training")
                return False
            
            # Train model
            history = self.model_trainer.train_model()
            
            # Save training results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"data/models/training_{timestamp}"
            self.model_trainer.save_training_results(output_dir)
            
            logger.info(f"Model training completed and saved to {output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _check_training_data(self) -> bool:
        """Check if we have sufficient data for training."""
        try:
            # Check historical data
            historical_files = [f for f in os.listdir("data/raw") if f.startswith("historical_traffic_")]
            
            if len(historical_files) < 3:  # Need at least 3 days of data
                return False
            
            # Check processed data
            processed_file = "data/processed/processed_graph.pkl"
            if not os.path.exists(processed_file):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking training data: {e}")
            return False
    
    def cleanup_old_data(self) -> bool:
        """
        Clean up old data files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Cleaning up old data files...")
            
            retention_days = self.config['data_collection']['retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean up raw data files
            raw_dir = "data/raw"
            for filename in os.listdir(raw_dir):
                if filename.endswith('.csv') or filename.endswith('.geojson'):
                    filepath = os.path.join(raw_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_date:
                        os.remove(filepath)
                        logger.info(f"Removed old file: {filename}")
            
            # Clean up processed data files
            processed_dir = "data/processed"
            for filename in os.listdir(processed_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(processed_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_date:
                        os.remove(filepath)
                        logger.info(f"Removed old processed file: {filename}")
            
            logger.info("Data cleanup completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False
    
    def backup_data(self) -> bool:
        """
        Create backup of important data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating data backup...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.config['storage']['backup_dir'], f"backup_{timestamp}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup important files
            important_files = [
                "data/raw/current_traffic.csv",
                "data/raw/mumbai_roads.geojson",
                "data/processed/processed_graph.pkl",
                "data/processed/mumbai_graph.pkl"
            ]
            
            for filepath in important_files:
                if os.path.exists(filepath):
                    filename = os.path.basename(filepath)
                    backup_filepath = os.path.join(backup_dir, filename)
                    
                    # Copy file
                    import shutil
                    shutil.copy2(filepath, backup_filepath)
                    logger.info(f"Backed up: {filename}")
            
            # Backup model files
            models_dir = "data/models"
            if os.path.exists(models_dir):
                for filename in os.listdir(models_dir):
                    if filename.endswith('.pkl') or filename.endswith('.h5'):
                        filepath = os.path.join(models_dir, filename)
                        backup_filepath = os.path.join(backup_dir, filename)
                        shutil.copy2(filepath, backup_filepath)
                        logger.info(f"Backed up model: {filename}")
            
            logger.info(f"Data backup completed: {backup_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def run_daily_pipeline(self):
        """Run the complete daily pipeline."""
        logger.info("Starting daily data pipeline...")
        
        pipeline_steps = [
            ("Collect Traffic Data", self.collect_traffic_data),
            ("Collect Incident Data", self.collect_incident_data),
            ("Process Data", self.process_data),
            ("Cleanup Old Data", self.cleanup_old_data),
            ("Create Backup", self.backup_data)
        ]
        
        results = {}
        
        for step_name, step_function in pipeline_steps:
            logger.info(f"Running: {step_name}")
            try:
                results[step_name] = step_function()
                if results[step_name]:
                    logger.info(f"✓ {step_name} completed successfully")
                else:
                    logger.warning(f"✗ {step_name} failed")
            except Exception as e:
                logger.error(f"✗ {step_name} failed with error: {e}")
                results[step_name] = False
        
        # Run model training if we have enough data
        if self._check_training_data():
            logger.info("Running: Train Model")
            try:
                results["Train Model"] = self.train_model()
                if results["Train Model"]:
                    logger.info("✓ Train Model completed successfully")
                else:
                    logger.warning("✗ Train Model failed")
            except Exception as e:
                logger.error(f"✗ Train Model failed with error: {e}")
                results["Train Model"] = False
        
        # Log summary
        successful_steps = sum(1 for success in results.values() if success)
        total_steps = len(results)
        
        logger.info(f"Daily pipeline completed: {successful_steps}/{total_steps} steps successful")
        
        return results
    
    def schedule_pipeline(self):
        """Schedule the data pipeline to run automatically."""
        logger.info("Scheduling data pipeline...")
        
        # Schedule traffic data collection every 15 minutes
        schedule.every(self.config['data_collection']['traffic_update_interval']).minutes.do(
            self.collect_traffic_data
        )
        
        # Schedule incident data collection every hour
        schedule.every().hour.do(self.collect_incident_data)
        
        # Schedule OSM data update daily
        schedule.every(self.config['data_collection']['osm_update_interval']).hours.do(
            self.update_osm_data
        )
        
        # Schedule daily pipeline
        schedule.every().day.at("02:00").do(self.run_daily_pipeline)
        
        # Schedule model retraining
        schedule.every(self.config['model_training']['retrain_interval']).hours.do(
            self.train_model
        )
        
        logger.info("Data pipeline scheduled successfully")
        
        # Run the scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """
    Main function to run the data pipeline.
    """
    # Configuration
    config = {
        'data_collection': {
            'traffic_update_interval': 15,  # minutes
            'osm_update_interval': 24,      # hours
            'historical_data_days': 7,      # days
            'retention_days': 30            # days
        },
        'model_training': {
            'retrain_interval': 24,         # hours
            'min_data_points': 1000,
            'validation_split': 0.2,
            'epochs': 50
        },
        'storage': {
            'data_dir': 'data',
            'models_dir': 'data/models',
            'logs_dir': 'logs',
            'backup_dir': 'data/backup'
        }
    }
    
    # Initialize pipeline
    pipeline = DataPipeline(config)
    
    # Run once for testing
    print("Running data pipeline once...")
    results = pipeline.run_daily_pipeline()
    
    print("\nPipeline Results:")
    for step, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {step}")
    
    # Uncomment to run scheduled pipeline
    # print("\nStarting scheduled pipeline...")
    # pipeline.schedule_pipeline()

if __name__ == "__main__":
    main()
