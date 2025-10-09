"""
Training script for Mumbai Navigation System GNN Model.
This script handles the complete training pipeline for the Graph Neural Network.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ingest.tomtom_fetch import TomTomTrafficFetcher
from ingest.osm_overpass import OSMDataFetcher
from preprocess.build_graph import RoadNetworkProcessor
from models.model import TrafficGNNModel
from models.train import ModelTrainer
from utils.data_pipeline import DataPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GNNTrainer:
    """
    Complete GNN training pipeline for the Mumbai Navigation System.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize GNN trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.trainer = None
        self.training_history = None
        self.evaluation_results = None
        
        # Create necessary directories
        self._create_directories()
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load training configuration."""
        default_config = {
            'data': {
                'historical_days': 30,
                'test_split': 0.2,
                'validation_split': 0.2,
                'random_state': 42,
                'min_data_points': 1000
            },
            'model': {
                'hidden_dims': [64, 32, 16],
                'output_dim': 1,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'activation': 'relu'
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'early_stopping_patience': 15,
                'reduce_lr_patience': 8,
                'min_delta': 0.001
            },
            'evaluation': {
                'metrics': ['mse', 'mae', 'mape', 'r2'],
                'cross_validation_folds': 5
            },
            'output': {
                'save_model': True,
                'save_predictions': True,
                'save_plots': True,
                'model_name': 'traffic_gnn_model'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with default config
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            'data/raw',
            'data/processed',
            'data/models',
            'logs',
            'results',
            'plots'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def collect_training_data(self) -> bool:
        """
        Collect and prepare training data.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("📊 Collecting training data...")
        
        try:
            # Step 1: Collect OSM road network data
            logger.info("🛣️  Collecting road network data...")
            if not self._collect_osm_data():
                return False
            
            # Step 2: Collect historical traffic data
            logger.info("🚦 Collecting historical traffic data...")
            if not self._collect_historical_data():
                return False
            
            # Step 3: Process data into graph format
            logger.info("🏗️  Processing data into graph format...")
            if not self._process_graph_data():
                return False
            
            logger.info("✅ Training data collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training data collection failed: {e}")
            return False
    
    def _collect_osm_data(self) -> bool:
        """Collect OpenStreetMap road network data."""
        try:
            # Check if we already have recent OSM data
            osm_file = "data/raw/mumbai_roads.geojson"
            if os.path.exists(osm_file):
                # Check file age
                file_age = datetime.now().timestamp() - os.path.getmtime(osm_file)
                if file_age < 86400:  # Less than 24 hours old
                    logger.info("   ✅ Using existing OSM data")
                    return True
            
            # Fetch new OSM data
            osm_fetcher = OSMDataFetcher()
            roads = osm_fetcher.get_road_network()
            
            # Save road network
            roads.to_file(osm_file, driver='GeoJSON')
            logger.info(f"   ✅ Collected {len(roads)} road segments")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ OSM data collection failed: {e}")
            return False
    
    def _collect_historical_data(self) -> bool:
        """Collect historical traffic data for training."""
        try:
            # Check if we have enough historical data
            historical_file = "data/raw/historical_traffic.csv"
            if os.path.exists(historical_file):
                # Check data size
                df = pd.read_csv(historical_file)
                if len(df) >= self.config['data']['min_data_points']:
                    logger.info(f"   ✅ Using existing historical data ({len(df)} records)")
                    return True
            
            # Fetch historical data
            tomtom_fetcher = TomTomTrafficFetcher()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=self.config['data']['historical_days'])
            
            logger.info(f"   📅 Fetching data from {start_date.date()} to {end_date.date()}")
            
            historical_data = tomtom_fetcher.get_historical_traffic_data(start_date, end_date)
            
            # Save historical data
            historical_data.to_csv(historical_file, index=False)
            logger.info(f"   ✅ Collected {len(historical_data)} historical records")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Historical data collection failed: {e}")
            return False
    
    def _process_graph_data(self) -> bool:
        """Process data into graph format for training."""
        try:
            # Load OSM data
            processor = RoadNetworkProcessor()
            roads = processor.load_osm_data("data/raw/mumbai_roads.geojson")
            
            # Load traffic data
            traffic_data = processor.load_traffic_data("data/raw/current_traffic.csv")
            
            # Build graph
            graph = processor.build_networkx_graph(roads)
            logger.info(f"   🗺️  Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            # Extract features
            node_features = processor.extract_node_features(graph, traffic_data)
            edge_features = processor.extract_edge_features(graph, traffic_data)
            
            # Create dataset
            dataset = processor.create_gnn_dataset(graph, node_features, edge_features)
            
            # Save processed dataset
            processor.save_processed_data(dataset, "data/processed/processed_graph.pkl")
            logger.info(f"   ✅ Created dataset with {dataset['num_nodes']} nodes and {dataset['num_edges']} edges")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Graph data processing failed: {e}")
            return False
    
    def train_model(self) -> bool:
        """
        Train the GNN model.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("🤖 Training GNN model...")
        
        try:
            # Initialize trainer
            self.trainer = ModelTrainer(self.config)
            
            # Train model
            logger.info("🚀 Starting model training...")
            self.training_history = self.trainer.train_model()
            
            # Get evaluation results
            self.evaluation_results = self.trainer.evaluation_results
            
            logger.info("✅ Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    def evaluate_model(self) -> dict:
        """
        Evaluate the trained model.
        
        Returns:
            Evaluation results dictionary
        """
        logger.info("📊 Evaluating model...")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return {}
        
        # Extract metrics
        metrics = {
            'mse': self.evaluation_results.get('mse', 0),
            'mae': self.evaluation_results.get('mae', 0),
            'r2': self.evaluation_results.get('r2', 0),
            'mape': self.evaluation_results.get('mape', 0)
        }
        
        # Log results
        logger.info("📈 Model Performance:")
        logger.info(f"   MSE: {metrics['mse']:.4f}")
        logger.info(f"   MAE: {metrics['mae']:.4f}")
        logger.info(f"   R²: {metrics['r2']:.4f}")
        logger.info(f"   MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def save_results(self) -> str:
        """
        Save training results and model.
        
        Returns:
            Path to saved results
        """
        logger.info("💾 Saving training results...")
        
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/training_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # Save model
            if self.config['output']['save_model'] and self.trainer and self.trainer.model:
                model_path = os.path.join(results_dir, self.config['output']['model_name'])
                self.trainer.model.save_model(model_path)
                logger.info(f"   ✅ Model saved to {model_path}")
            
            # Save training history
            if self.training_history:
                history_path = os.path.join(results_dir, "training_history.json")
                with open(history_path, 'w') as f:
                    json.dump(self.training_history.history, f, indent=2)
                logger.info(f"   ✅ Training history saved to {history_path}")
            
            # Save evaluation results
            if self.evaluation_results:
                eval_path = os.path.join(results_dir, "evaluation_results.json")
                eval_data = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in self.evaluation_results.items() 
                           if k not in ['predictions', 'true_values']}
                with open(eval_path, 'w') as f:
                    json.dump(eval_data, f, indent=2)
                logger.info(f"   ✅ Evaluation results saved to {eval_path}")
            
            # Save configuration
            config_path = os.path.join(results_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"   ✅ Configuration saved to {config_path}")
            
            # Save plots
            if self.config['output']['save_plots']:
                self._save_training_plots(results_dir)
            
            # Save predictions
            if self.config['output']['save_predictions'] and self.evaluation_results:
                self._save_predictions(results_dir)
            
            logger.info(f"✅ All results saved to {results_dir}")
            return results_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return ""
    
    def _save_training_plots(self, results_dir: str):
        """Save training plots."""
        try:
            plots_dir = os.path.join(results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Training history plot
            if self.training_history:
                self.trainer.plot_training_history(os.path.join(plots_dir, "training_history.png"))
            
            # Evaluation results plot
            if self.evaluation_results:
                self.trainer.plot_evaluation_results(os.path.join(plots_dir, "evaluation_results.png"))
            
            logger.info(f"   ✅ Plots saved to {plots_dir}")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to save plots: {e}")
    
    def _save_predictions(self, results_dir: str):
        """Save model predictions."""
        try:
            if 'predictions' in self.evaluation_results and 'true_values' in self.evaluation_results:
                predictions_df = pd.DataFrame({
                    'true_values': self.evaluation_results['true_values'].flatten(),
                    'predictions': self.evaluation_results['predictions'].flatten(),
                    'residuals': (self.evaluation_results['true_values'] - self.evaluation_results['predictions']).flatten()
                })
                
                predictions_path = os.path.join(results_dir, "predictions.csv")
                predictions_df.to_csv(predictions_path, index=False)
                logger.info(f"   ✅ Predictions saved to {predictions_path}")
                
        except Exception as e:
            logger.error(f"   ❌ Failed to save predictions: {e}")
    
    def run_complete_training(self) -> bool:
        """
        Run the complete training pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Starting Complete GNN Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Collect training data
            logger.info("📊 Step 1: Collecting training data...")
            if not self.collect_training_data():
                logger.error("❌ Training data collection failed")
                return False
            
            # Step 2: Train model
            logger.info("\n🤖 Step 2: Training GNN model...")
            if not self.train_model():
                logger.error("❌ Model training failed")
                return False
            
            # Step 3: Evaluate model
            logger.info("\n📊 Step 3: Evaluating model...")
            metrics = self.evaluate_model()
            
            # Step 4: Save results
            logger.info("\n💾 Step 4: Saving results...")
            results_dir = self.save_results()
            
            # Step 5: Summary
            logger.info("\n📋 Step 5: Training Summary")
            logger.info("=" * 60)
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"📁 Results saved to: {results_dir}")
            logger.info(f"📈 Final Performance:")
            for metric, value in metrics.items():
                logger.info(f"   {metric.upper()}: {value:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete training pipeline failed: {e}")
            return False
    
    def run_quick_training(self) -> bool:
        """
        Run a quick training for testing purposes.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("⚡ Running Quick Training (Testing Mode)")
        logger.info("=" * 60)
        
        # Modify config for quick training
        quick_config = self.config.copy()
        quick_config['data']['historical_days'] = 3
        quick_config['training']['epochs'] = 10
        quick_config['training']['batch_size'] = 16
        
        # Create temporary trainer with quick config
        temp_trainer = ModelTrainer(quick_config)
        
        try:
            # Use existing data if available
            if not os.path.exists("data/processed/processed_graph.pkl"):
                logger.info("📊 Collecting minimal training data...")
                if not self.collect_training_data():
                    return False
            
            # Quick training
            logger.info("🤖 Quick model training...")
            history = temp_trainer.train_model()
            
            # Quick evaluation
            if temp_trainer.evaluation_results:
                metrics = temp_trainer.evaluation_results
                logger.info("📈 Quick Training Results:")
                logger.info(f"   MSE: {metrics.get('mse', 0):.4f}")
                logger.info(f"   MAE: {metrics.get('mae', 0):.4f}")
                logger.info(f"   R²: {metrics.get('r2', 0):.4f}")
            
            logger.info("✅ Quick training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quick training failed: {e}")
            return False

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GNN model for Mumbai Navigation System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--quick', action='store_true', help='Run quick training for testing')
    parser.add_argument('--data-only', action='store_true', help='Only collect training data')
    parser.add_argument('--train-only', action='store_true', help='Only train model (assumes data exists)')
    
    args = parser.parse_args()
    
    print("🤖 Mumbai Navigation System - GNN Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = GNNTrainer(args.config)
    
    try:
        if args.data_only:
            # Only collect data
            print("📊 Data Collection Mode")
            success = trainer.collect_training_data()
            
        elif args.train_only:
            # Only train model
            print("🤖 Training Mode")
            success = trainer.train_model()
            if success:
                trainer.evaluate_model()
                trainer.save_results()
                
        elif args.quick:
            # Quick training
            print("⚡ Quick Training Mode")
            success = trainer.run_quick_training()
            
        else:
            # Complete training pipeline
            print("🚀 Complete Training Mode")
            success = trainer.run_complete_training()
        
        if success:
            print("\n🎉 Training completed successfully!")
            print("📁 Check the results/ directory for outputs")
            print("🌐 You can now run main.py to use the trained model")
        else:
            print("\n❌ Training failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.exception("Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
