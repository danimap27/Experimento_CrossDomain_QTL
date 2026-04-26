import matplotlib.pyplot as plt
import numpy as np

def plot_topology_decay():
    # Metrics based on experimental results for Task A retention
    ansatze = ['A (Strongly Entangling)', 'B (Basic Entangler)', 'C (Tree Tensor Network)']
    
    # Accuracy on Task A immediately after learning Task A (Initial)
    initial_accs = [96.0, 75.0, 96.0]
    # Accuracy on Task A after learning Task B (Final Retained)
    final_accs = [90.0, 59.5, 90.5]
    
    # Calculate the drop (Forgetting)
    drops = [init - final for init, final in zip(initial_accs, final_accs)]
    
    x = np.arange(len(ansatze))
    width = 0.25  # Thinner bars for 3 items
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot grouped bars (3 per Ansatz)
    bars_initial = ax.bar(x - width, initial_accs, width, label='Initial Accuracy', color='#06d6a0')
    bars_retained = ax.bar(x, final_accs, width, label='Retained Accuracy', color='#3a86ff')
    bars_drop = ax.bar(x + width, drops, width, label='Accuracy Drop (Forgetting)', color='#e63946')
    
    # Styling and Labels
    ax.set_ylabel(r'Accuracy (\%)', fontsize=12)
    ax.set_title('Topology Resilience: Initial, Retained, and Catastrophic Drop', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ansatze, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 115) # Increased to make room for legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add numerical labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=10)

    autolabel(bars_initial)
    autolabel(bars_retained)
    autolabel(bars_drop)

    fig.tight_layout()
    plt.savefig('figure2_ansatz_decay.png', dpi=300)
    print("Standalone Figure saved as figure2_ansatz_decay.png")
    plt.show(block=False)

if __name__ == "__main__":
    plot_topology_decay()
