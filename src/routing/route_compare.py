"""
Route comparison system for evaluating different routing algorithms.
This module compares A*, Dijkstra, and GNN-based routing performance.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import json
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from routing.a_star import AStarRouter, DijkstraRouter, RouteResult
from routing.gnn_router import GNNRouter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Result of route comparison."""
    algorithm: str
    route: RouteResult
    success: bool
    error_message: str = ""
    computation_time: float = 0.0
    memory_usage: float = 0.0

@dataclass
class RouteComparison:
    """Complete route comparison results."""
    start_node: str
    end_node: str
    timestamp: datetime
    results: List[ComparisonResult]
    best_route: Optional[RouteResult] = None
    performance_metrics: Dict = None

class RouteComparator:
    """
    System for comparing different routing algorithms.
    """
    
    def __init__(self, 
                 graph: nx.Graph, 
                 traffic_data: Dict = None,
                 gnn_model_path: str = None,
                 gnn_dataset_path: str = None):
        """
        Initialize route comparator.
        
        Args:
            graph: NetworkX graph
            traffic_data: Traffic data dictionary
            gnn_model_path: Path to GNN model
            gnn_dataset_path: Path to GNN dataset
        """
        self.graph = graph
        self.traffic_data = traffic_data or {}
        
        # Initialize routers
        self.astar_router = AStarRouter(graph, traffic_data)
        self.dijkstra_router = DijkstraRouter(graph, traffic_data)
        self.gnn_router = GNNRouter(graph, gnn_model_path, gnn_dataset_path, traffic_data)
        
        # Performance tracking
        self.comparison_history = []
        
    def compare_routes(self, 
                      start_node: str, 
                      end_node: str,
                      cost_function: str = 'time',
                      traffic_weight: float = 1.0,
                      include_gnn: bool = True) -> RouteComparison:
        """
        Compare routes from different algorithms.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            cost_function: Cost function type
            traffic_weight: Traffic weight
            include_gnn: Whether to include GNN routing
            
        Returns:
            RouteComparison object with results
        """
        logger.info(f"Comparing routes from {start_node} to {end_node}")
        
        results = []
        start_time = datetime.now()
        
        # Test A* algorithm
        astar_result = self._test_algorithm(
            'A*', self.astar_router, start_node, end_node, cost_function, traffic_weight
        )
        results.append(astar_result)
        
        # Test Dijkstra algorithm
        dijkstra_result = self._test_algorithm(
            'Dijkstra', self.dijkstra_router, start_node, end_node, cost_function, traffic_weight
        )
        results.append(dijkstra_result)
        
        # Test GNN algorithm (if available)
        if include_gnn and self.gnn_router.model:
            gnn_result = self._test_algorithm(
                'GNN-Enhanced', self.gnn_router, start_node, end_node, cost_function, traffic_weight
            )
            results.append(gnn_result)
        
        # Find best route
        successful_results = [r for r in results if r.success]
        best_route = None
        if successful_results:
            if cost_function == 'time':
                best_route = min(successful_results, key=lambda x: x.route.total_time).route
            elif cost_function == 'distance':
                best_route = min(successful_results, key=lambda x: x.route.total_distance).route
            else:
                best_route = min(successful_results, key=lambda x: x.route.total_cost).route
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(results)
        
        comparison = RouteComparison(
            start_node=start_node,
            end_node=end_node,
            timestamp=start_time,
            results=results,
            best_route=best_route,
            performance_metrics=performance_metrics
        )
        
        # Store in history
        self.comparison_history.append(comparison)
        
        return comparison
    
    def _test_algorithm(self, 
                       algorithm_name: str,
                       router,
                       start_node: str,
                       end_node: str,
                       cost_function: str,
                       traffic_weight: float) -> ComparisonResult:
        """
        Test a specific routing algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            router: Router instance
            start_node: Starting node
            end_node: Ending node
            cost_function: Cost function
            traffic_weight: Traffic weight
            
        Returns:
            ComparisonResult object
        """
        start_time = datetime.now()
        
        try:
            # Find route
            route = router.find_route(start_node, end_node, cost_function, traffic_weight)
            
            # Calculate computation time
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return ComparisonResult(
                algorithm=algorithm_name,
                route=route,
                success=True,
                computation_time=computation_time
            )
            
        except Exception as e:
            computation_time = (datetime.now() - start_time).total_seconds()
            return ComparisonResult(
                algorithm=algorithm_name,
                route=None,
                success=False,
                error_message=str(e),
                computation_time=computation_time
            )
    
    def _calculate_performance_metrics(self, results: List[ComparisonResult]) -> Dict:
        """
        Calculate performance metrics from comparison results.
        
        Args:
            results: List of comparison results
            
        Returns:
            Performance metrics dictionary
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {'error': 'No successful routes found'}
        
        metrics = {
            'total_algorithms_tested': len(results),
            'successful_algorithms': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'computation_times': {},
            'route_metrics': {},
            'algorithm_rankings': {}
        }
        
        # Computation times
        for result in results:
            metrics['computation_times'][result.algorithm] = result.computation_time
        
        # Route metrics
        for result in successful_results:
            route = result.route
            metrics['route_metrics'][result.algorithm] = {
                'distance': route.total_distance,
                'time': route.total_time,
                'cost': route.total_cost,
                'path_length': len(route.path)
            }
        
        # Algorithm rankings
        if successful_results:
            # Rank by time
            time_ranking = sorted(successful_results, key=lambda x: x.route.total_time)
            metrics['algorithm_rankings']['by_time'] = [r.algorithm for r in time_ranking]
            
            # Rank by distance
            distance_ranking = sorted(successful_results, key=lambda x: x.route.total_distance)
            metrics['algorithm_rankings']['by_distance'] = [r.algorithm for r in distance_ranking]
            
            # Rank by cost
            cost_ranking = sorted(successful_results, key=lambda x: x.route.total_cost)
            metrics['algorithm_rankings']['by_cost'] = [r.algorithm for r in cost_ranking]
            
            # Rank by computation time
            speed_ranking = sorted(successful_results, key=lambda x: x.computation_time)
            metrics['algorithm_rankings']['by_speed'] = [r.algorithm for r in speed_ranking]
        
        return metrics
    
    def batch_compare_routes(self, 
                           route_pairs: List[Tuple[str, str]],
                           cost_function: str = 'time',
                           traffic_weight: float = 1.0,
                           include_gnn: bool = True) -> List[RouteComparison]:
        """
        Compare routes for multiple start-end pairs.
        
        Args:
            route_pairs: List of (start_node, end_node) tuples
            cost_function: Cost function type
            traffic_weight: Traffic weight
            include_gnn: Whether to include GNN routing
            
        Returns:
            List of RouteComparison objects
        """
        logger.info(f"Batch comparing {len(route_pairs)} route pairs")
        
        comparisons = []
        for start_node, end_node in route_pairs:
            try:
                comparison = self.compare_routes(
                    start_node, end_node, cost_function, traffic_weight, include_gnn
                )
                comparisons.append(comparison)
            except Exception as e:
                logger.error(f"Error comparing route {start_node} -> {end_node}: {e}")
        
        return comparisons
    
    def generate_performance_report(self, 
                                  comparisons: List[RouteComparison] = None) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            comparisons: List of comparisons (uses history if None)
            
        Returns:
            Performance report dictionary
        """
        if comparisons is None:
            comparisons = self.comparison_history
        
        if not comparisons:
            return {'error': 'No comparisons available'}
        
        report = {
            'summary': self._generate_summary_statistics(comparisons),
            'algorithm_performance': self._analyze_algorithm_performance(comparisons),
            'route_quality_analysis': self._analyze_route_quality(comparisons),
            'computational_efficiency': self._analyze_computational_efficiency(comparisons),
            'recommendations': self._generate_recommendations(comparisons)
        }
        
        return report
    
    def _generate_summary_statistics(self, comparisons: List[RouteComparison]) -> Dict:
        """Generate summary statistics."""
        total_comparisons = len(comparisons)
        successful_comparisons = len([c for c in comparisons if c.best_route])
        
        # Algorithm success rates
        algorithm_success = {}
        for comparison in comparisons:
            for result in comparison.results:
                if result.algorithm not in algorithm_success:
                    algorithm_success[result.algorithm] = {'success': 0, 'total': 0}
                algorithm_success[result.algorithm]['total'] += 1
                if result.success:
                    algorithm_success[result.algorithm]['success'] += 1
        
        # Calculate success rates
        for alg in algorithm_success:
            success_rate = algorithm_success[alg]['success'] / algorithm_success[alg]['total']
            algorithm_success[alg]['success_rate'] = success_rate
        
        return {
            'total_comparisons': total_comparisons,
            'successful_comparisons': successful_comparisons,
            'overall_success_rate': successful_comparisons / total_comparisons if total_comparisons > 0 else 0,
            'algorithm_success_rates': algorithm_success
        }
    
    def _analyze_algorithm_performance(self, comparisons: List[RouteComparison]) -> Dict:
        """Analyze algorithm performance."""
        algorithm_stats = {}
        
        for comparison in comparisons:
            for result in comparison.results:
                if not result.success:
                    continue
                
                alg = result.algorithm
                if alg not in algorithm_stats:
                    algorithm_stats[alg] = {
                        'total_distance': [],
                        'total_time': [],
                        'total_cost': [],
                        'computation_time': [],
                        'path_length': []
                    }
                
                route = result.route
                algorithm_stats[alg]['total_distance'].append(route.total_distance)
                algorithm_stats[alg]['total_time'].append(route.total_time)
                algorithm_stats[alg]['total_cost'].append(route.total_cost)
                algorithm_stats[alg]['computation_time'].append(result.computation_time)
                algorithm_stats[alg]['path_length'].append(len(route.path))
        
        # Calculate statistics
        for alg in algorithm_stats:
            stats = algorithm_stats[alg]
            for metric in stats:
                values = stats[metric]
                if values:
                    stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
        
        return algorithm_stats
    
    def _analyze_route_quality(self, comparisons: List[RouteComparison]) -> Dict:
        """Analyze route quality."""
        quality_metrics = {
            'best_algorithm_by_metric': {},
            'route_improvements': {},
            'consistency_analysis': {}
        }
        
        # Find best algorithm for each metric
        metrics = ['total_distance', 'total_time', 'total_cost']
        for metric in metrics:
            best_algorithm = None
            best_value = float('inf')
            
            for comparison in comparisons:
                for result in comparison.results:
                    if result.success:
                        value = getattr(result.route, metric)
                        if value < best_value:
                            best_value = value
                            best_algorithm = result.algorithm
            
            quality_metrics['best_algorithm_by_metric'][metric] = {
                'algorithm': best_algorithm,
                'value': best_value
            }
        
        return quality_metrics
    
    def _analyze_computational_efficiency(self, comparisons: List[RouteComparison]) -> Dict:
        """Analyze computational efficiency."""
        efficiency_stats = {}
        
        for comparison in comparisons:
            for result in comparison.results:
                if not result.success:
                    continue
                
                alg = result.algorithm
                if alg not in efficiency_stats:
                    efficiency_stats[alg] = []
                
                efficiency_stats[alg].append(result.computation_time)
        
        # Calculate efficiency metrics
        for alg in efficiency_stats:
            times = efficiency_stats[alg]
            efficiency_stats[alg] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_routes': len(times)
            }
        
        return efficiency_stats
    
    def _generate_recommendations(self, comparisons: List[RouteComparison]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Analyze success rates
        summary = self._generate_summary_statistics(comparisons)
        success_rates = summary['algorithm_success_rates']
        
        if success_rates:
            best_algorithm = max(success_rates.keys(), 
                               key=lambda x: success_rates[x]['success_rate'])
            recommendations.append(f"Use {best_algorithm} for highest success rate "
                                f"({success_rates[best_algorithm]['success_rate']:.2%})")
        
        # Analyze performance
        performance = self._analyze_algorithm_performance(comparisons)
        
        if performance:
            # Find fastest algorithm
            fastest_alg = min(performance.keys(), 
                            key=lambda x: performance[x]['computation_time']['mean'])
            recommendations.append(f"Use {fastest_alg} for fastest computation time")
            
            # Find most consistent algorithm
            most_consistent = min(performance.keys(),
                                key=lambda x: performance[x]['total_time']['std'])
            recommendations.append(f"Use {most_consistent} for most consistent results")
        
        return recommendations
    
    def plot_comparison_results(self, 
                              comparisons: List[RouteComparison] = None,
                              save_path: str = None):
        """
        Plot comparison results.
        
        Args:
            comparisons: List of comparisons
            save_path: Path to save the plot
        """
        if comparisons is None:
            comparisons = self.comparison_history
        
        if not comparisons:
            logger.warning("No comparisons to plot")
            return
        
        # Prepare data for plotting
        plot_data = []
        for comparison in comparisons:
            for result in comparison.results:
                if result.success:
                    plot_data.append({
                        'Algorithm': result.algorithm,
                        'Distance (m)': result.route.total_distance,
                        'Time (s)': result.route.total_time,
                        'Cost': result.route.total_cost,
                        'Computation Time (s)': result.computation_time,
                        'Path Length': len(result.route.path)
                    })
        
        if not plot_data:
            logger.warning("No successful routes to plot")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Distance comparison
        sns.boxplot(data=df, x='Algorithm', y='Distance (m)', ax=axes[0, 0])
        axes[0, 0].set_title('Route Distance Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Time comparison
        sns.boxplot(data=df, x='Algorithm', y='Time (s)', ax=axes[0, 1])
        axes[0, 1].set_title('Route Time Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cost comparison
        sns.boxplot(data=df, x='Algorithm', y='Cost', ax=axes[0, 2])
        axes[0, 2].set_title('Route Cost Comparison')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Computation time comparison
        sns.boxplot(data=df, x='Algorithm', y='Computation Time (s)', ax=axes[1, 0])
        axes[1, 0].set_title('Computation Time Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Path length comparison
        sns.boxplot(data=df, x='Algorithm', y='Path Length', ax=axes[1, 1])
        axes[1, 1].set_title('Path Length Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        success_data = []
        for comparison in comparisons:
            for result in comparison.results:
                success_data.append({
                    'Algorithm': result.algorithm,
                    'Success': result.success
                })
        
        success_df = pd.DataFrame(success_data)
        success_rates = success_df.groupby('Algorithm')['Success'].mean()
        success_rates.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Success Rate Comparison')
        axes[1, 2].set_ylabel('Success Rate')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_comparison_results(self, 
                              comparisons: List[RouteComparison] = None,
                              output_path: str = None):
        """
        Save comparison results to file.
        
        Args:
            comparisons: List of comparisons
            output_path: Output file path
        """
        if comparisons is None:
            comparisons = self.comparison_history
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed/route_comparison_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to serializable format
        serializable_comparisons = []
        for comparison in comparisons:
            comp_dict = asdict(comparison)
            # Convert datetime to string
            comp_dict['timestamp'] = comp_dict['timestamp'].isoformat()
            # Convert RouteResult objects
            for result in comp_dict['results']:
                if result['route']:
                    result['route'] = asdict(result['route'])
                    result['route']['timestamp'] = result['route']['timestamp'].isoformat()
            serializable_comparisons.append(comp_dict)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_comparisons, f, indent=2)
        
        logger.info(f"Comparison results saved to {output_path}")
        
        return output_path

def main():
    """
    Example usage of RouteComparator.
    """
    # Create a test graph
    G = nx.Graph()
    
    # Add nodes
    nodes = [
        ('A', {'lat': 19.0760, 'lon': 72.8777}),
        ('B', {'lat': 19.0760, 'lon': 72.8778}),
        ('C', {'lat': 19.0761, 'lon': 72.8777}),
        ('D', {'lat': 19.0761, 'lon': 72.8778}),
        ('E', {'lat': 19.0762, 'lon': 72.8777}),
    ]
    
    G.add_nodes_from(nodes)
    
    # Add edges
    edges = [
        ('A', 'B', {'length': 100, 'maxspeed': '50'}),
        ('A', 'C', {'length': 150, 'maxspeed': '40'}),
        ('B', 'D', {'length': 150, 'maxspeed': '40'}),
        ('C', 'D', {'length': 100, 'maxspeed': '50'}),
        ('D', 'E', {'length': 200, 'maxspeed': '60'}),
    ]
    
    G.add_edges_from(edges)
    
    # Create traffic data
    traffic_data = {
        'A_B': {'current_speed': 30, 'jam_factor': 0.4},
        'A_C': {'current_speed': 45, 'jam_factor': 0.1},
        'B_D': {'current_speed': 45, 'jam_factor': 0.1},
        'C_D': {'current_speed': 25, 'jam_factor': 0.5},
        'D_E': {'current_speed': 50, 'jam_factor': 0.0},
    }
    
    # Initialize comparator
    print("Initializing route comparator...")
    comparator = RouteComparator(G, traffic_data)
    
    # Compare single route
    print("Comparing single route...")
    comparison = comparator.compare_routes('A', 'E', cost_function='time')
    
    print(f"Best route: {comparison.best_route.algorithm}")
    print(f"Distance: {comparison.best_route.total_distance:.2f}m")
    print(f"Time: {comparison.best_route.total_time:.2f}s")
    
    # Batch compare multiple routes
    print("\nBatch comparing routes...")
    route_pairs = [('A', 'E'), ('B', 'D'), ('C', 'E')]
    batch_comparisons = comparator.batch_compare_routes(route_pairs)
    
    # Generate performance report
    print("\nGenerating performance report...")
    report = comparator.generate_performance_report()
    
    print("Performance Report Summary:")
    print(f"Total comparisons: {report['summary']['total_comparisons']}")
    print(f"Success rate: {report['summary']['overall_success_rate']:.2%}")
    
    # Save results
    print("\nSaving results...")
    output_file = comparator.save_comparison_results()
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
