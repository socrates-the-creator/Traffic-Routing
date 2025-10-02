"""
Model training pipeline for the Traffic GNN model.
This module handles data preparation, model training, and evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import TrafficGNNModel, TrafficPredictor
from preprocess.build_graph import RoadNetworkProcessor
from ingest.tomtom_fetch import TomTomTrafficFetcher
from ingest.osm_overpass import OSMDataFetcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles training of the Traffic GNN model.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.training_history = None
        self.evaluation_results = None
        
    def _get_default_config(self) -> Dict:
        """Get default training configuration."""
        return {
            'model': {
                'hidden_dims': [64, 32, 16],
                'output_dim': 1,
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.2,
                'early_stopping_patience': 10,
                'reduce_lr_patience': 5
            },
            'data': {
                'historical_days': 30,
                'test_split': 0.2,
                'random_state': 42
            }
        }
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from historical traffic data.
        
        Returns:
            Tuple of (node_features, adjacency_matrix, edge_features, targets)
        """
        logger.info("Preparing training data...")
        
        # Load or generate historical data
        historical_data = self._load_historical_data()
        
        # Process data into time series
        time_series_data = self._create_time_series_data(historical_data)
        
        # Create training samples
        X_node, X_adj, X_edge, y = self._create_training_samples(time_series_data)
        
        logger.info(f"Prepared {len(X_node)} training samples")
        
        return X_node, X_adj, X_edge, y
    
    def _load_historical_data(self) -> pd.DataFrame:
        """
        Load historical traffic data for training.
        
        Returns:
            Historical traffic data DataFrame
        """
        # Try to load existing historical data
        hist_file = "data/raw/historical_traffic.csv"
        
        if os.path.exists(hist_file):
            logger.info(f"Loading historical data from {hist_file}")
            return pd.read_csv(hist_file)
        else:
            # Generate historical data
            logger.info("Generating historical traffic data...")
            fetcher = TomTomTrafficFetcher()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['data']['historical_days'])
            
            historical_data = fetcher.get_historical_traffic_data(start_date, end_date)
            
            # Save for future use
            os.makedirs(os.path.dirname(hist_file), exist_ok=True)
            historical_data.to_csv(hist_file, index=False)
            
            return historical_data
    
    def _create_time_series_data(self, historical_data: pd.DataFrame) -> Dict:
        """
        Create time series data from historical traffic data.
        
        Args:
            historical_data: Historical traffic data
            
        Returns:
            Time series data dictionary
        """
        # Group by segment and time
        time_series = historical_data.groupby(['segment_id', 'timestamp']).agg({
            'speed': 'mean',
            'volume': 'sum',
            'day_of_week': 'first',
            'hour': 'first',
            'is_weekend': 'first'
        }).reset_index()
        
        # Sort by timestamp
        time_series['timestamp'] = pd.to_datetime(time_series['timestamp'])
        time_series = time_series.sort_values(['segment_id', 'timestamp'])
        
        # Create features and targets
        features = []
        targets = []
        
        for segment_id in time_series['segment_id'].unique():
            segment_data = time_series[time_series['segment_id'] == segment_id].copy()
            
            # Create lagged features
            for i in range(len(segment_data) - 1):
                current_row = segment_data.iloc[i]
                next_row = segment_data.iloc[i + 1]
                
                # Features: current speed, volume, time features
                feature = [
                    current_row['speed'],
                    current_row['volume'],
                    current_row['day_of_week'],
                    current_row['hour'],
                    current_row['is_weekend']
                ]
                
                # Target: next speed
                target = next_row['speed']
                
                features.append(feature)
                targets.append(target)
        
        return {
            'features': np.array(features),
            'targets': np.array(targets)
        }
    
    def _create_training_samples(self, time_series_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create training samples from time series data.
        
        Args:
            time_series_data: Time series data dictionary
            
        Returns:
            Tuple of training data
        """
        # Load graph structure
        graph_data = self._load_graph_data()
        
        # Create node features from time series
        node_features = self._create_node_features_from_timeseries(
            time_series_data, graph_data
        )
        
        # Create adjacency matrix
        adjacency_matrix = self._create_adjacency_matrix(graph_data)
        
        # Create edge features
        edge_features = self._create_edge_features(graph_data)
        
        # Create targets
        targets = time_series_data['targets'].reshape(-1, 1)
        
        return node_features, adjacency_matrix, edge_features, targets
    
    def _load_graph_data(self) -> Dict:
        """
        Load or create graph data structure.
        
        Returns:
            Graph data dictionary
        """
        # Try to load existing processed graph
        processed_file = "data/processed/processed_graph.pkl"
        
        if os.path.exists(processed_file):
            logger.info(f"Loading processed graph from {processed_file}")
            with open(processed_file, 'rb') as f:
                return pickle.load(f)
        else:
            # Create new graph data
            logger.info("Creating new graph data...")
            return self._create_graph_data()
    
    def _create_graph_data(self) -> Dict:
        """
        Create graph data from OSM and traffic data.
        
        Returns:
            Graph data dictionary
        """
        # Fetch OSM data
        osm_fetcher = OSMDataFetcher()
        roads = osm_fetcher.get_road_network()
        
        # Fetch current traffic data
        tomtom_fetcher = TomTomTrafficFetcher()
        traffic = tomtom_fetcher.get_traffic_flow_data()
        
        # Process into graph
        processor = RoadNetworkProcessor()
        graph = processor.build_networkx_graph(roads)
        node_features = processor.extract_node_features(graph, traffic)
        edge_features = processor.extract_edge_features(graph, traffic)
        
        # Create dataset
        dataset = processor.create_gnn_dataset(graph, node_features, edge_features)
        
        # Save for future use
        os.makedirs(os.path.dirname("data/processed/processed_graph.pkl"), exist_ok=True)
        with open("data/processed/processed_graph.pkl", 'wb') as f:
            pickle.dump(dataset, f)
        
        return dataset
    
    def _create_node_features_from_timeseries(self, time_series_data: Dict, graph_data: Dict) -> np.ndarray:
        """
        Create node features from time series data.
        
        Args:
            time_series_data: Time series data
            graph_data: Graph data
            
        Returns:
            Node features array
        """
        # Use existing node features as base
        base_features = graph_data['node_features']
        
        # Add time series features
        time_features = time_series_data['features']
        
        # Combine features (simplified approach)
        if len(time_features) > 0:
            # Repeat time features for all nodes
            repeated_time_features = np.tile(time_features[0], (base_features.shape[0], 1))
            combined_features = np.concatenate([base_features, repeated_time_features], axis=1)
        else:
            combined_features = base_features
        
        return combined_features
    
    def _create_adjacency_matrix(self, graph_data: Dict) -> np.ndarray:
        """Create adjacency matrix from graph data."""
        return self._create_adjacency_matrix_from_edges(
            graph_data['edge_list'], graph_data['num_nodes']
        )
    
    def _create_adjacency_matrix_from_edges(self, edge_list: np.ndarray, num_nodes: int) -> np.ndarray:
        """Create adjacency matrix from edge list."""
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        for edge in edge_list:
            source, target = edge[0], edge[1]
            adjacency_matrix[source, target] = 1
            adjacency_matrix[target, source] = 1  # Undirected graph
        
        return adjacency_matrix
    
    def _create_edge_features(self, graph_data: Dict) -> np.ndarray:
        """Create edge features from graph data."""
        edge_features = graph_data['edge_features']
        edge_list = graph_data['edge_list']
        num_nodes = graph_data['num_nodes']
        
        edge_feature_dim = edge_features.shape[1] if len(edge_features) > 0 else 1
        edge_features_matrix = np.zeros((num_nodes, num_nodes, edge_feature_dim))
        
        for i, edge in enumerate(edge_list):
            source, target = edge[0], edge[1]
            if i < len(edge_features):
                edge_features_matrix[source, target] = edge_features[i]
                edge_features_matrix[target, source] = edge_features[i]
        
        return edge_features_matrix
    
    def train_model(self) -> keras.callbacks.History:
        """
        Train the Traffic GNN model.
        
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        # Prepare training data
        X_node, X_adj, X_edge, y = self.prepare_training_data()
        
        # Split data
        X_node_train, X_node_test, X_adj_train, X_adj_test, X_edge_train, X_edge_test, y_train, y_test = train_test_split(
            X_node, X_adj, X_edge, y,
            test_size=self.config['data']['test_split'],
            random_state=self.config['data']['random_state']
        )
        
        # Initialize model
        self.model = TrafficGNNModel(
            num_nodes=X_node.shape[1],
            node_feature_dim=X_node.shape[2],
            edge_feature_dim=X_edge.shape[3],
            hidden_dims=self.config['model']['hidden_dims'],
            output_dim=self.config['model']['output_dim'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        # Build model
        self.model.build_model()
        
        # Train model
        self.training_history = self.model.train(
            X_node_train, X_adj_train, X_edge_train, y_train,
            validation_split=self.config['training']['validation_split'],
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size']
        )
        
        # Evaluate model
        self.evaluation_results = self.evaluate_model(
            X_node_test, X_adj_test, X_edge_test, y_test
        )
        
        logger.info("Model training completed")
        
        return self.training_history
    
    def evaluate_model(self, 
                      X_node: np.ndarray,
                      X_adj: np.ndarray,
                      X_edge: np.ndarray,
                      y_true: np.ndarray) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X_node: Node features
            X_adj: Adjacency matrix
            X_edge: Edge features
            y_true: True target values
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_node, X_adj, X_edge)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        results = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred,
            'true_values': y_true
        }
        
        logger.info(f"Evaluation results: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        
        return results
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.training_history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.training_history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.training_history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # MAPE
        axes[1, 0].plot(self.training_history.history['mape'], label='Training MAPE')
        axes[1, 0].plot(self.training_history.history['val_mape'], label='Validation MAPE')
        axes[1, 0].set_title('Mean Absolute Percentage Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].legend()
        
        # Learning rate
        if 'lr' in self.training_history.history:
            axes[1, 1].plot(self.training_history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_evaluation_results(self, save_path: str = None):
        """
        Plot evaluation results.
        
        Args:
            save_path: Path to save the plot
        """
        if self.evaluation_results is None:
            logger.warning("No evaluation results available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Predictions vs True values
        y_true = self.evaluation_results['true_values']
        y_pred = self.evaluation_results['predictions']
        
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predictions')
        axes[0].set_title('Predictions vs True Values')
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predictions')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation results plot saved to {save_path}")
        
        plt.show()
    
    def save_training_results(self, output_dir: str = None):
        """
        Save training results and model.
        
        Args:
            output_dir: Output directory
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"data/models/training_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if self.model is not None:
            model_path = os.path.join(output_dir, "model")
            self.model.save_model(model_path)
        
        # Save training history
        if self.training_history is not None:
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.training_history.history, f, indent=2)
        
        # Save evaluation results
        if self.evaluation_results is not None:
            eval_path = os.path.join(output_dir, "evaluation_results.json")
            eval_data = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in self.evaluation_results.items() 
                        if k not in ['predictions', 'true_values']}
            with open(eval_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save plots
        self.plot_training_history(os.path.join(output_dir, "training_history.png"))
        self.plot_evaluation_results(os.path.join(output_dir, "evaluation_results.png"))
        
        logger.info(f"Training results saved to {output_dir}")

def main():
    """
    Main training pipeline.
    """
    # Configuration
    config = {
        'model': {
            'hidden_dims': [64, 32, 16],
            'output_dim': 1,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        },
        'training': {
            'epochs': 50,  # Reduced for demo
            'batch_size': 16,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5
        },
        'data': {
            'historical_days': 7,  # Reduced for demo
            'test_split': 0.2,
            'random_state': 42
        }
    }
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train model
    print("Starting model training...")
    history = trainer.train_model()
    
    # Save results
    print("Saving training results...")
    trainer.save_training_results()
    
    print("Training pipeline completed!")

if __name__ == "__main__":
    main()
