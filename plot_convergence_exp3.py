import matplotlib.pyplot as plt
import numpy as np

def plot_exp3_clear_advantage():
    # Data extracted from the most recent Exp 3 stdout log
    epochs = np.arange(1, 16)
    
    # Loss curves
    loss_scratch = [0.4364, 0.2092, 0.1983, 0.1775, 0.1731, 0.1637, 0.1638, 0.1697, 0.1772, 0.1773, 0.1579, 0.1658, 0.1676, 0.1717, 0.1770]
    loss_syn = [0.3249, 0.2049, 0.1982, 0.1753, 0.1730, 0.1670, 0.1730, 0.1664, 0.1849, 0.1629, 0.1615, 0.1680, 0.1663, 0.1569, 0.1588]
    loss_mob = [0.7933, 0.2563, 0.1877, 0.1986, 0.1887, 0.1672, 0.1715, 0.1730, 0.1586, 0.1650, 0.1699, 0.1719, 0.1713, 0.1602, 0.1707]
    
    # Final Accuracies
    models = ['Baseline\n(Scratch)', 'QTL\n(Synthetic Source)', 'QTL\n(MobileNet Source)']
    accs = [91.00, 90.50, 92.50]
    
    # Create a 1x2 layout to explain BOTH advantages visually
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PANEL 1: Convergence (Zoomed on early epochs to show Synthetic advantage) ---
    ax1.plot(epochs, loss_scratch, label='Scratch (Random Init)', color='#000000', linestyle='--', linewidth=2.5)
    ax1.plot(epochs, loss_syn, label='QTL (Synthetic Init)', color='#06d6a0', marker='o', linewidth=2.5)
    ax1.plot(epochs, loss_mob, label='QTL (MobileNet Init)', color='#118ab2', marker='s', linewidth=2.5)
    
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Objective 1: Convergence Acceleration', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=11)
    
    # Add annotation for the Synthetic advantage at epoch 1
    ax1.annotate('Initial Loss Advantage:\nFaster Convergence', 
                 xy=(1.2, 0.33), xycoords='data',
                 xytext=(3, 0.5), textcoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#06d6a0", lw=2))

    # --- PANEL 2: Final Accuracy (Shows MobileNet advantage) ---
    bars = ax2.bar(models, accs, color=['#343a40', '#06d6a0', '#118ab2'], width=0.5)
    ax2.set_ylabel(r'Target Accuracy (\%)', fontsize=12)
    ax2.set_title('Objective 2: Final Generalization Quality', fontsize=14, fontweight='bold')
    ax2.set_ylim(85, 95) # Zoomed in to highlight the differences among the tops
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add a global super title explaining the overall experiment
    fig.suptitle('Experiment 3: Cross-Domain Quantum Transfer Learning (Target: MNIST)', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('figure3_convergence.png', dpi=300, bbox_inches='tight')
    print("Standalone Figure saved as figure3_convergence.png")
    plt.show(block=False)

if __name__ == "__main__":
    plot_exp3_clear_advantage()
