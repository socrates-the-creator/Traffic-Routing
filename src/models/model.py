"""
Graph Neural Network model for traffic prediction and routing optimization.
This module implements a GNN architecture using TensorFlow for traffic flow prediction.
"""

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    # Fallback for older TensorFlow versions
    import tensorflow as tf
    import keras
    from keras import layers
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
import pickle
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphConvolutionLayer(layers.Layer):
    """
    Graph Convolution Layer implementation for traffic prediction.
    """
    
    def __init__(self, units: int, activation: str = 'relu', use_bias: bool = True, **kwargs):
        """
        Initialize Graph Convolution Layer.
        
        Args:
            units: Number of output units
            activation: Activation function
            use_bias: Whether to use bias
        """
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        
    def build(self, input_shape):
        """Build the layer weights."""
        # Node features weight matrix
        self.node_weight = self.add_weight(
            name='node_weight',
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Neighbor aggregation weight matrix
        self.neighbor_weight = self.add_weight(
            name='neighbor_weight',
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
        
        super(GraphConvolutionLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass of the graph convolution layer.
        
        Args:
            inputs: Tuple of (node_features, adjacency_matrix)
            training: Training mode flag
            
        Returns:
            Updated node features
        """
        node_features, adjacency_matrix = inputs
        
        # Self-connection
        self_features = tf.matmul(node_features, self.node_weight)
        
        # Neighbor aggregation
        neighbor_features = tf.matmul(node_features, self.neighbor_weight)
        aggregated_neighbors = tf.matmul(adjacency_matrix, neighbor_features)
        
        # Combine self and neighbor features
        output = self_features + aggregated_neighbors
        
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        
        return self.activation(output)
    
    def get_config(self):
        """Get layer configuration."""
        config = super(GraphConvolutionLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config

class TrafficGNNModel:
    """
    Graph Neural Network model for traffic prediction and routing optimization.
    """
    
    def __init__(self, 
                 num_nodes: int,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dims: List[int] = [64, 32, 16],
                 output_dim: int = 1,
                 dropout_rate: float = 0.2):
        """
        Initialize the Traffic GNN model.
        
        Args:
            num_nodes: Number of nodes in the graph
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout_rate: Dropout rate
        """
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the GNN model architecture."""
        
        # Input layers
        node_features_input = layers.Input(shape=(self.num_nodes, self.node_feature_dim), name='node_features')
        adjacency_matrix_input = layers.Input(shape=(self.num_nodes, self.num_nodes), name='adjacency_matrix')
        edge_features_input = layers.Input(shape=(self.num_nodes, self.num_nodes, self.edge_feature_dim), name='edge_features')
        
        # Graph Convolution layers
        x = node_features_input
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Graph convolution
            x = GraphConvolutionLayer(
                units=hidden_dim,
                activation='relu',
                name=f'gcn_{i}'
            )([x, adjacency_matrix_input])
            
            # Dropout
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
            
            # Layer normalization
            x = layers.LayerNormalization(name=f'norm_{i}')(x)
        
        # Global pooling to get graph-level representation
        graph_embedding = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers for final prediction
        x = layers.Dense(32, activation='relu', name='dense_1')(graph_embedding)
        x = layers.Dropout(self.dropout_rate, name='dropout_final')(x)
        x = layers.Dense(16, activation='relu', name='dense_2')(x)
        
        # Output layer
        output = layers.Dense(self.output_dim, activation='linear', name='output')(x)
        
        # Create model
        self.model = keras.Model(
            inputs=[node_features_input, adjacency_matrix_input, edge_features_input],
            outputs=output,
            name='TrafficGNN'
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("Built Traffic GNN model")
        self.model.summary()
        
        return self.model
    
    def prepare_data(self, dataset: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Tuple of (node_features, adjacency_matrix, edge_features, targets)
        """
        # Node features
        node_features = dataset['node_features']
        
        # Create adjacency matrix
        adjacency_matrix = self._create_adjacency_matrix(dataset['edge_list'], dataset['num_nodes'])
        
        # Edge features (simplified - using average edge features for each node pair)
        edge_features = self._create_edge_features(dataset['edge_features'], dataset['edge_list'], dataset['num_nodes'])
        
        # Create targets (traffic speed prediction)
        targets = self._create_targets(node_features)
        
        return node_features, adjacency_matrix, edge_features, targets
    
    def _create_adjacency_matrix(self, edge_list: np.ndarray, num_nodes: int) -> np.ndarray:
        """
        Create adjacency matrix from edge list.
        
        Args:
            edge_list: List of edges
            num_nodes: Number of nodes
            
        Returns:
            Adjacency matrix
        """
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        for edge in edge_list:
            source, target = edge[0], edge[1]
            adjacency_matrix[source, target] = 1
            adjacency_matrix[target, source] = 1  # Undirected graph
        
        return adjacency_matrix
    
    def _create_edge_features(self, edge_features: np.ndarray, edge_list: np.ndarray, num_nodes: int) -> np.ndarray:
        """
        Create edge features matrix.
        
        Args:
            edge_features: Edge features array
            edge_list: List of edges
            num_nodes: Number of nodes
            
        Returns:
            Edge features matrix
        """
        edge_feature_dim = edge_features.shape[1] if len(edge_features) > 0 else 1
        edge_features_matrix = np.zeros((num_nodes, num_nodes, edge_feature_dim))
        
        for i, edge in enumerate(edge_list):
            source, target = edge[0], edge[1]
            if i < len(edge_features):
                edge_features_matrix[source, target] = edge_features[i]
                edge_features_matrix[target, source] = edge_features[i]  # Undirected graph
        
        return edge_features_matrix
    
    def _create_targets(self, node_features: np.ndarray) -> np.ndarray:
        """
        Create target values for training.
        
        Args:
            node_features: Node features array
            
        Returns:
            Target values
        """
        # Use current speed as target (simplified)
        if 'current_speed' in range(node_features.shape[1]):
            targets = node_features[:, 8]  # Assuming current_speed is at index 8
        else:
            # Generate synthetic targets
            targets = np.random.uniform(20, 80, node_features.shape[0])
        
        return targets.reshape(-1, 1)
    
    def train(self, 
              node_features: np.ndarray,
              adjacency_matrix: np.ndarray,
              edge_features: np.ndarray,
              targets: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1) -> keras.callbacks.History:
        """
        Train the GNN model.
        
        Args:
            node_features: Node features
            adjacency_matrix: Adjacency matrix
            edge_features: Edge features
            targets: Target values
            validation_split: Validation split ratio
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X = [node_features, adjacency_matrix, edge_features]
        y = targets
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("Training completed")
        
        return self.history
    
    def predict(self, 
                node_features: np.ndarray,
                adjacency_matrix: np.ndarray,
                edge_features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            node_features: Node features
            adjacency_matrix: Adjacency matrix
            edge_features: Edge features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = [node_features, adjacency_matrix, edge_features]
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_traffic_flow(self, dataset: Dict) -> Dict:
        """
        Predict traffic flow for the entire graph.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Dictionary with traffic predictions
        """
        # Prepare data
        node_features, adjacency_matrix, edge_features, _ = self.prepare_data(dataset)
        
        # Make predictions
        predictions = self.predict(node_features, adjacency_matrix, edge_features)
        
        # Create results
        results = {
            'node_predictions': predictions.flatten(),
            'node_mapping': dataset['node_mapping'],
            'graph': dataset.get('graph', None)
        }
        
        return results
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path to saved model
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"traffic_gnn_model_{timestamp}"
        
        filepath = os.path.join("data", "models", filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'GraphConvolutionLayer': GraphConvolutionLayer}
        )
        logger.info(f"Model loaded from {filepath}")

class TrafficPredictor:
    """
    High-level interface for traffic prediction using GNN.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize traffic predictor.
        
        Args:
            model_path: Path to trained model
        """
        self.model = None
        self.dataset = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model."""
        # Load model architecture and weights
        # This would be implemented based on your model saving format
        pass
    
    def load_dataset(self, dataset_path: str):
        """Load processed dataset."""
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
    
    def predict_current_traffic(self) -> Dict:
        """
        Predict current traffic conditions.
        
        Returns:
            Dictionary with traffic predictions
        """
        if self.model is None or self.dataset is None:
            raise ValueError("Model and dataset must be loaded first")
        
        # Use the model to predict traffic
        predictions = self.model.predict_traffic_flow(self.dataset)
        
        return predictions
    
    def get_route_recommendations(self, start_node: str, end_node: str) -> Dict:
        """
        Get route recommendations based on traffic predictions.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            
        Returns:
            Dictionary with route recommendations
        """
        # This would integrate with the routing module
        # For now, return a placeholder
        return {
            'start_node': start_node,
            'end_node': end_node,
            'recommended_route': [],
            'estimated_time': 0,
            'traffic_conditions': 'unknown'
        }

def main():
    """
    Example usage of TrafficGNNModel.
    """
    # Load processed dataset
    print("Loading processed dataset...")
    with open("data/processed/processed_graph.pkl", 'rb') as f:
        dataset = pickle.load(f)
    
    # Initialize model
    print("Initializing GNN model...")
    model = TrafficGNNModel(
        num_nodes=dataset['num_nodes'],
        node_feature_dim=dataset['node_features'].shape[1],
        edge_feature_dim=dataset['edge_features'].shape[1] if len(dataset['edge_features']) > 0 else 1,
        hidden_dims=[64, 32, 16],
        output_dim=1
    )
    
    # Build model
    print("Building model architecture...")
    model.build_model()
    
    # Prepare data
    print("Preparing training data...")
    node_features, adjacency_matrix, edge_features, targets = model.prepare_data(dataset)
    
    # Train model
    print("Training model...")
    history = model.train(
        node_features, adjacency_matrix, edge_features, targets,
        epochs=50,
        batch_size=16,
        verbose=1
    )
    
    # Save model
    print("Saving model...")
    model_path = model.save_model()
    print(f"Model saved to {model_path}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict_traffic_flow(dataset)
    print(f"Generated predictions for {len(predictions['node_predictions'])} nodes")

if __name__ == "__main__":
    main()
