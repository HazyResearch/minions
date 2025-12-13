import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_visualizations(data, output_file='experiment_results.png'):
    results = data['results']
    analysis = data['analysis']

    # Extract data for plotting
    queries = [r['query'][:20] + '...' for r in results] # Truncate for labels
    local_times = [r['local_time'] for r in results]
    full_times = [r['full_time'] for r in results]
    winners = [r['winner'] for r in results]
    query_types = [r['query_type'] for r in results]
    speedups = [r.get('speedup_factor', 0) for r in results]

    matplotlib.rcParams["font.size"] = 14
    matplotlib.rcParams['figure.dpi'] = 300

    # Constants from analysis (using literals as requested)
    TOTAL_QUERIES = 120
    TOTAL_TIME_SAVED = 2462.47
    AVG_SPEEDUP = 14.33

    # Setup the figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Minions Experiment Results\nTotal Time Saved: {TOTAL_TIME_SAVED:.2f}s | Avg Speedup: {AVG_SPEEDUP:.2f}x | N={TOTAL_QUERIES}', fontsize=16)
    
    # 1. Win Rate Distribution
    ax1 = plt.subplot(2, 2, 1)
    unique_winners, counts = np.unique(winners, return_counts=True)
    colors = {'local': '#2ecc71', 'full': '#e74c3c', 'tie': '#95a5a6'}
    bar_colors = [colors.get(w, 'gray') for w in unique_winners]
    
    bars = ax1.bar(unique_winners, counts, color=bar_colors)
    ax1.set_title('Win Rate Distribution')
    ax1.set_ylabel('Number of Queries')
    ax1.bar_label(bars)

    # 2. Latency Comparison (Log Scale)
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(local_times, full_times, alpha=0.7, c='#3498db')
    
    # Add diagonal line for equal latency
    max_val = max(max(local_times), max(full_times))
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Latency')
    
    ax2.set_title('Latency Comparison: Local vs Full Strategy')
    ax2.set_xlabel('Local Time (s)')
    ax2.set_ylabel('Full Time (s)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Speedup Factor Histogram
    ax3 = plt.subplot(2, 2, 3)
    # Filter out reasonable outliers for visualization if needed, but here we show all
    # Using a log scale for bins might be better given the spread
    ax3.hist(speedups, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax3.set_title('Distribution of Speedup Factors')
    ax3.set_xlabel('Speedup Factor (Full Time / Local Time)')
    ax3.set_ylabel('Frequency')
    
    # Add mean line
    ax3.axvline(AVG_SPEEDUP, color='r', linestyle='dashed', linewidth=1, label=f'Avg: {AVG_SPEEDUP:.1f}x')
    ax3.legend()

    # 4. Winners by Query Type
    ax4 = plt.subplot(2, 2, 4)
    
    # Aggregate data
    type_winner_map = {}
    for qt, winner in zip(query_types, winners):
        if qt not in type_winner_map:
            type_winner_map[qt] = {'local': 0, 'full': 0, 'tie': 0}
        type_winner_map[qt][winner] = type_winner_map[qt].get(winner, 0) + 1
    
    q_types = list(type_winner_map.keys())
    local_counts = [type_winner_map[qt].get('local', 0) for qt in q_types]
    full_counts = [type_winner_map[qt].get('full', 0) for qt in q_types]
    tie_counts = [type_winner_map[qt].get('tie', 0) for qt in q_types]
    
    x = np.arange(len(q_types))
    width = 0.6
    
    ax4.bar(x, local_counts, width, label='Local', color=colors['local'])
    ax4.bar(x, full_counts, width, bottom=local_counts, label='Full', color=colors['full'])
    # Stack tie on top of local + full
    bottom_tie = np.array(local_counts) + np.array(full_counts)
    ax4.bar(x, tie_counts, width, bottom=bottom_tie, label='Tie', color=colors['tie'])
    
    ax4.set_title('Winners by Query Type')
    ax4.set_xticks(x)
    ax4.set_xticklabels(q_types, rotation=45, ha='right')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Successfully created visualization: {output_file}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'enhanced_results.json')
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
    else:
        data = load_data(json_path)
        create_visualizations(data, os.path.join(current_dir, 'experiment_visualization.png'))
