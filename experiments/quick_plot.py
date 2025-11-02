"""
Quick plotting script for your smart router results.

Just run: python experiments/quick_plot.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.plot_smart_router_results import SmartRouterPlotter

def main():
    """Quick plot generation."""
    results_file = "real_results.json"
    
    if not Path(results_file).exists():
        print(f"❌ Results file '{results_file}' not found!")
        print("💡 Make sure you've run the experiments first")
        return
    
    print("🎨 Creating plots from your experiment results...")
    plotter = SmartRouterPlotter(results_file, "plots")
    plotter.create_all_plots()

if __name__ == "__main__":
    main()