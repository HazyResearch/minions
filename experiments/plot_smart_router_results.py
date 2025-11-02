"""
Smart Router Results Visualization

This script creates comprehensive plots from smart router experiment results.

Usage:
    python experiments/plot_smart_router_results.py real_results.json
    python experiments/plot_smart_router_results.py --input results.json --output plots/
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SmartRouterPlotter:
    """Create comprehensive visualizations from smart router experiment results."""
    
    def __init__(self, results_path: str, output_dir: str = "plots"):
        """
        Initialize the plotter with results data.
        
        Args:
            results_path: Path to the JSON results file
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        with open(results_path, 'r') as f:
            self.results = json.load(f)
            
        print(f"📊 Loaded results from {results_path}")
        print(f"🎨 Plots will be saved to {output_dir}/")
    
    def plot_classification_accuracy(self):
        """Plot query classification accuracy results."""
        classification = self.results.get('classification', {})
        if not classification:
            print("⚠️  No classification results found")
            return
            
        results = classification.get('results', [])
        if not results:
            return
            
        # Create confusion matrix data
        df = pd.DataFrame(results)
        confusion_data = df.groupby(['expected_type', 'predicted_type']).size().unstack(fill_value=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix
        sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Query Classification Confusion Matrix')
        ax1.set_xlabel('Predicted Type')
        ax1.set_ylabel('Expected Type')
        
        # Accuracy by type
        accuracy_by_type = df.groupby('expected_type')['correct'].mean()
        accuracy_by_type.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title('Classification Accuracy by Query Type')
        ax2.set_xlabel('Query Type')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add overall accuracy text
        overall_accuracy = classification.get('accuracy', 0)
        fig.suptitle(f'Query Classification Results (Overall Accuracy: {overall_accuracy:.1%})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classification_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved classification_accuracy.png")
    
    def plot_routing_simulation(self):
        """Plot routing simulation results."""
        routing_sim = self.results.get('routing_simulation', {})
        if not routing_sim:
            print("⚠️  No routing simulation results found")
            return
            
        results = routing_sim.get('results', [])
        if not results:
            return
            
        df = pd.DataFrame(results)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Routing decision distribution
        route_counts = df['predicted_route'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        route_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Routing Decisions Distribution')
        ax1.set_ylabel('')
        
        # Uncertainty vs Threshold by Query Type
        for query_type in df['query_type'].unique():
            subset = df[df['query_type'] == query_type]
            ax2.scatter(subset['threshold'], subset['simulated_uncertainty'], 
                       label=query_type, alpha=0.7, s=60)
        
        # Add diagonal line where uncertainty = threshold
        min_val = min(df['threshold'].min(), df['simulated_uncertainty'].min())
        max_val = max(df['threshold'].max(), df['simulated_uncertainty'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Threshold Line')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Simulated Uncertainty')
        ax2.set_title('Uncertainty vs Threshold by Query Type')
        ax2.legend()
        
        # Routing accuracy by query type
        accuracy_by_type = df.groupby('query_type')['correct'].mean()
        accuracy_by_type.plot(kind='bar', ax=ax3, color='lightblue')
        ax3.set_title('Routing Accuracy by Query Type')
        ax3.set_xlabel('Query Type')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        
        # Query type distribution
        type_counts = df['query_type'].value_counts()
        type_counts.plot(kind='bar', ax=ax4, color='orange')
        ax4.set_title('Query Type Distribution')
        ax4.set_xlabel('Query Type')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        
        overall_accuracy = routing_sim.get('accuracy', 0)
        fig.suptitle(f'Routing Simulation Results (Overall Accuracy: {overall_accuracy:.1%})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'routing_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved routing_simulation.png")
    
    def plot_live_experiments(self):
        """Plot live experiment results with comprehensive metrics."""
        live_exp = self.results.get('live_experiments', {})
        if not live_exp or live_exp.get('skipped'):
            print("⚠️  No live experiment results found")
            return
            
        results = live_exp.get('results', [])
        if not results:
            return
            
        df = pd.DataFrame(results)
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Time Performance Comparison
        ax1 = plt.subplot(3, 3, 1)
        if 'total_time_seconds' in df.columns:
            time_by_type = df.groupby('query_type')['total_time_seconds'].mean()
            time_by_type.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Average Execution Time by Query Type')
            ax1.set_xlabel('Query Type')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Routing Decisions
        ax2 = plt.subplot(3, 3, 2)
        if 'actual_route' in df.columns:
            route_counts = df['actual_route'].value_counts()
            colors = ['lightgreen', 'lightcoral', 'lightgray']
            route_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=colors[:len(route_counts)])
            ax2.set_title('Actual Routing Decisions')
            ax2.set_ylabel('')
        
        # 3. Token Usage Comparison
        ax3 = plt.subplot(3, 3, 3)
        if 'local_tokens' in df.columns and 'remote_tokens' in df.columns:
            token_data = df.groupby('query_type')[['local_tokens', 'remote_tokens']].sum()
            token_data.plot(kind='bar', ax=ax3, stacked=True)
            ax3.set_title('Token Usage by Query Type')
            ax3.set_xlabel('Query Type')
            ax3.set_ylabel('Total Tokens')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend(['Local', 'Remote'])
        
        # 4. Time Breakdown (Local vs Remote vs Overhead)
        ax4 = plt.subplot(3, 3, 4)
        if all(col in df.columns for col in ['local_time_seconds', 'remote_time_seconds', 'total_time_seconds']):
            df['overhead_time'] = df['total_time_seconds'] - df['local_time_seconds'] - df['remote_time_seconds']
            time_breakdown = df.groupby('query_type')[['local_time_seconds', 'remote_time_seconds', 'overhead_time']].mean()
            time_breakdown.plot(kind='bar', ax=ax4, stacked=True)
            ax4.set_title('Time Breakdown by Query Type')
            ax4.set_xlabel('Query Type')
            ax4.set_ylabel('Time (seconds)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend(['Local', 'Remote', 'Overhead'])
        
        # 5. Success Rate by Query Type
        ax5 = plt.subplot(3, 3, 5)
        if 'success' in df.columns:
            success_rate = df.groupby('query_type')['success'].mean()
            success_rate.plot(kind='bar', ax=ax5, color='lightgreen')
            ax5.set_title('Success Rate by Query Type')
            ax5.set_xlabel('Query Type')
            ax5.set_ylabel('Success Rate')
            ax5.tick_params(axis='x', rotation=45)
            ax5.set_ylim(0, 1)
        
        # 6. Time Saved Analysis
        ax6 = plt.subplot(3, 3, 6)
        if 'time_saved_seconds' in df.columns:
            time_saved = df.groupby('query_type')['time_saved_seconds'].sum()
            time_saved.plot(kind='bar', ax=ax6, color='gold')
            ax6.set_title('Total Time Saved by Query Type')
            ax6.set_xlabel('Query Type')
            ax6.set_ylabel('Time Saved (seconds)')
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. Local vs Remote Route Accuracy
        ax7 = plt.subplot(3, 3, 7)
        if all(col in df.columns for col in ['expected_route', 'actual_route']):
            accuracy_data = []
            for query_type in df['query_type'].unique():
                subset = df[df['query_type'] == query_type]
                if len(subset) > 0:
                    accuracy = (subset['expected_route'] == subset['actual_route']).mean()
                    accuracy_data.append({'query_type': query_type, 'accuracy': accuracy})
            
            if accuracy_data:
                acc_df = pd.DataFrame(accuracy_data)
                acc_df.plot(x='query_type', y='accuracy', kind='bar', ax=ax7, color='purple')
                ax7.set_title('Routing Accuracy by Query Type')
                ax7.set_xlabel('Query Type')
                ax7.set_ylabel('Accuracy')
                ax7.tick_params(axis='x', rotation=45)
                ax7.set_ylim(0, 1)
        
        # 8. Performance Improvement (Local-only vs Full Protocol)
        ax8 = plt.subplot(3, 3, 8)
        if 'was_local_only' in df.columns:
            local_only_df = df[df['was_local_only'] == True]
            full_protocol_df = df[df['was_local_only'] == False]
            
            performance_data = {
                'Local-only Avg Time': local_only_df['total_time_seconds'].mean() if len(local_only_df) > 0 else 0,
                'Full Protocol Avg Time': full_protocol_df['total_time_seconds'].mean() if len(full_protocol_df) > 0 else 0
            }
            
            bars = ax8.bar(performance_data.keys(), performance_data.values(), 
                          color=['lightgreen', 'lightcoral'])
            ax8.set_title('Performance: Local-only vs Full Protocol')
            ax8.set_ylabel('Average Time (seconds)')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s', ha='center', va='bottom')
        
        # 9. Cost Analysis (if token data available)
        ax9 = plt.subplot(3, 3, 9)
        if 'remote_tokens' in df.columns:
            # Estimate costs (using rough pricing)
            df_cost = df.copy()
            df_cost['estimated_cost'] = df_cost['remote_tokens'] * 0.00001  # ~$0.01 per 1K tokens
            cost_by_type = df_cost.groupby('query_type')['estimated_cost'].sum()
            cost_by_type.plot(kind='bar', ax=ax9, color='red')
            ax9.set_title('Estimated Cost by Query Type')
            ax9.set_xlabel('Query Type')
            ax9.set_ylabel('Estimated Cost ($)')
            ax9.tick_params(axis='x', rotation=45)
        
        # Add summary statistics
        total_queries = live_exp.get('total_queries', 0)
        successful_queries = live_exp.get('successful_queries', 0)
        total_time_saved = live_exp.get('total_time_saved_seconds', 0)
        local_only_queries = live_exp.get('local_only_queries', 0)
        
        fig.suptitle(f'Live Experiments Results\n'
                    f'Total: {total_queries} queries | Successful: {successful_queries} | '
                    f'Local-only: {local_only_queries} | Time Saved: {total_time_saved:.1f}s', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'live_experiments.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved live_experiments.png")
    
    def plot_summary_dashboard(self):
        """Create a summary dashboard with key metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall Performance Summary
        classification_acc = self.results.get('classification', {}).get('accuracy', 0)
        routing_sim_acc = self.results.get('routing_simulation', {}).get('accuracy', 0)
        
        metrics = ['Classification\nAccuracy', 'Routing Simulation\nAccuracy']
        values = [classification_acc, routing_sim_acc]
        colors = ['lightblue', 'lightgreen']
        
        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_title('Smart Router Performance Metrics', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add percentage labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                    f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Live Experiments Summary
        live_exp = self.results.get('live_experiments', {})
        if not live_exp.get('skipped'):
            total_queries = live_exp.get('total_queries', 0)
            successful_queries = live_exp.get('successful_queries', 0)
            local_only_queries = live_exp.get('local_only_queries', 0)
            
            categories = ['Total\nQueries', 'Successful\nQueries', 'Local-only\nQueries']
            counts = [total_queries, successful_queries, local_only_queries]
            colors = ['gray', 'green', 'blue']
            
            bars = ax2.bar(categories, counts, color=colors)
            ax2.set_title('Live Experiments Summary', fontweight='bold')
            ax2.set_ylabel('Count')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax2.text(bar.get_x() + bar.get_width()/2., count + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Live Experiments\nRan', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=16)
            ax2.set_title('Live Experiments Summary', fontweight='bold')
        
        # Time and Cost Savings
        time_saved = live_exp.get('total_time_saved_seconds', 0)
        total_local_tokens = live_exp.get('total_local_tokens', 0)
        total_remote_tokens = live_exp.get('total_remote_tokens', 0)
        
        savings_data = {
            'Time Saved\n(seconds)': time_saved,
            'Local Tokens\n(thousands)': total_local_tokens / 1000,
            'Remote Tokens\n(thousands)': total_remote_tokens / 1000
        }
        
        bars = ax3.bar(savings_data.keys(), savings_data.values(), 
                      color=['gold', 'lightblue', 'lightcoral'])
        ax3.set_title('Resource Usage & Savings', fontweight='bold')
        ax3.set_ylabel('Count')
        
        # Add value labels
        for bar, (key, value) in zip(bars, savings_data.items()):
            label = f'{value:.1f}' if 'seconds' in key else f'{value:.0f}K'
            ax3.text(bar.get_x() + bar.get_width()/2., value + max(savings_data.values()) * 0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        # Dataset Summary
        dataset_summary = self.results.get('dataset_summary', {})
        total_queries = dataset_summary.get('total_queries', 0)
        by_type = dataset_summary.get('by_type', {})
        
        if by_type:
            types = list(by_type.keys())
            counts = list(by_type.values())
            
            ax4.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Test Dataset Distribution', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Dataset\nInformation', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=16)
            ax4.set_title('Test Dataset Distribution', fontweight='bold')
        
        # Main title with timestamp
        import datetime
        timestamp = self.results.get('timestamp', 0)
        if timestamp:
            date_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            fig.suptitle(f'Smart Router Experiment Dashboard\nRun on: {date_str}', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Smart Router Experiment Dashboard', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved summary_dashboard.png")
    
    def create_all_plots(self):
        """Create all available plots."""
        print("🎨 Creating comprehensive smart router visualizations...")
        print("=" * 60)
        
        self.plot_summary_dashboard()
        self.plot_classification_accuracy()
        self.plot_routing_simulation()
        self.plot_live_experiments()
        
        print("\n" + "=" * 60)
        print(f"🎉 All plots saved to {self.output_dir}/")
        print("📊 Generated plots:")
        print("   • summary_dashboard.png - Key metrics overview")
        print("   • classification_accuracy.png - Query classification results")
        print("   • routing_simulation.png - Routing decision analysis")
        print("   • live_experiments.png - Complete protocol performance")


def main():
    parser = argparse.ArgumentParser(description="Plot smart router experiment results")
    parser.add_argument(
        "input",
        nargs="?",
        default="smart_router_results.json",
        help="Input JSON results file (default: smart_router_results.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots/)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Results file '{args.input}' not found!")
        print("💡 Make sure to run the experiments first:")
        print("   python experiments/run_smart_router_experiments.py --mode full --output results.json")
        return
    
    # Create plotter and generate all plots
    plotter = SmartRouterPlotter(args.input, args.output)
    plotter.create_all_plots()


if __name__ == "__main__":
    main()