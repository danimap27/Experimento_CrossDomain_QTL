import matplotlib.pyplot as plt
from data_module import DataModule
from quantum_net import HybridQuantumNet
from experiment_runner import ExperimentRunner

def plot_results(exp1_data, exp2_data, exp3_data):
    """Generates the 3 requested plots."""
    print("Generating Matplotlib visualizations...")
    
    # ---------------------------
    # Figure 1: Baseline vs QTL Forgetting
    # Figure 1: Forgetting Drop (Bar Chart)
    # ---------------------------
    drop_base, drop_qtl = exp1_data
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Baseline (No Transfer)', 'QTL (Synthetic Prior)'], [drop_base, drop_qtl], color=['#e63946', '#457b9d'])
    
    plt.ylabel(r'Forgetting Drop (\% Accuracy Loss) $\downarrow$', fontsize=12)
    plt.title('Experiment 1: Mitigation of Catastrophic Forgetting', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figure1_forgetting.png', dpi=300)
    plt.show(block=False)

    # ---------------------------
    # Figure 2: Ansatz Final Accuracy Retention (Bar Chart)
    # ---------------------------
    plt.figure(figsize=(8, 5))
    ansatze = ['A (Strongly Entangling)', 'B (Basic Entangler)', 'C (Tree Tensor Network)']
    final_accs = [exp2_data['A']['acc_A'], exp2_data['B']['acc_A'], exp2_data['C']['acc_A']]
    colors = ['#8338ec', '#ffbe0b', '#3a86ff']
    
    bars2 = plt.bar(ansatze, final_accs, color=colors)
    
    plt.ylabel(r'Retained Task A Accuracy (\%) $\uparrow$', fontsize=12)
    plt.title('Experiment 2: Topology Resilience against Interference', fontsize=14)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figure2_ansatz_decay.png', dpi=300)
    plt.show(block=False)

    # ---------------------------
    # Figure 3: Scratch vs QTL Convergence
    # ---------------------------
    scr_metrics, qtl_metrics, pretrain_metrics, mob_qtl_metrics, mob_pretrain_metrics = exp3_data
    scratch_loss = scr_metrics[0]
    qtl_loss = qtl_metrics[0]
    mob_qtl_loss = mob_qtl_metrics[0]
    
    epochs = range(1, len(scratch_loss) + 1)
    
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, scratch_loss, marker='s', label='Target from Scratch', color='#d62828', linestyle='--', linewidth=2)
    plt.plot(epochs, qtl_loss, marker='o', label='QTL (Synthetic Source)', color='#003049', linewidth=3)
    plt.plot(epochs, mob_qtl_loss, marker='^', label='QTL (MobileNetV2 Source)', color='#2a9d8f', linewidth=3)
    
    plt.xlabel('Target Training Epochs', fontsize=12)
    plt.ylabel(r'Cross-Entropy Loss $\downarrow$', fontsize=12)
    plt.title('Experiment 3: Faster Convergence via Cross-Domain Prior', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figure3_convergence.png', dpi=300)
    plt.show(block=False)
    
    print("\n\n=== OVERALL METRICS TABLE ===")
    print("Experimento 2 (Topologías):")
    for ans, met in exp2_data.items():
        print(f" Ansatz {ans} | Tr.A: {met['train_time_A']:.2f}s | Tst.A: {met['test_time_A']:.2f}s | Acc.A: {met['acc_A']:.2f}% | Tr.B: {met['train_time_B']:.2f}s | Tst.B: {met['test_time_B']:.2f}s | Acc.B: {met['acc_B']:.2f}%")
        
    print("\nExperimento 3 (Cross-Domain):")
    print(f" Pre-Training (Syn)| Tr: {pretrain_metrics[0]:.2f}s | Acc: {pretrain_metrics[1]:.2f}% | Tst: {pretrain_metrics[2]:.2f}s")
    print(f" Pre-Train (Mobile)| Tr: {mob_pretrain_metrics[0]:.2f}s | Acc: {mob_pretrain_metrics[1]:.2f}% | Tst: {mob_pretrain_metrics[2]:.2f}s")
    print(f" Scratch Target    | Tr: {scr_metrics[1]:.2f}s | Acc: {scr_metrics[2]:.2f}% | Tst: {scr_metrics[3]:.2f}s")
    print(f" QTL Target (Syn)  | Tr: {qtl_metrics[1]:.2f}s | Acc: {qtl_metrics[2]:.2f}% | Tst: {qtl_metrics[3]:.2f}s")
    print(f" QTL Target (Mob)  | Tr: {mob_qtl_metrics[1]:.2f}s | Acc: {mob_qtl_metrics[2]:.2f}% | Tst: {mob_qtl_metrics[3]:.2f}s")
    
    print("Figures saved successfully as PNGs.")

def main():
    print("Initializing Quantum Modules and Loading Data...")
    data_mod = DataModule(batch_size=32)
    
    # Let's drastically limit datasets for testing/dry-run speeds:
    # However, to be empirically sound, we use standard loading.
    syn_train, syn_test = data_mod.get_synthetic_task(n_samples=2500)
    source_loader = (syn_train, syn_test)
    
    fa_train, fa_test, _ = data_mod.get_fashion_mnist_task(classes=(0, 1))
    task_a_loader = (fa_train, fa_test)
    
    # Task B: MNIST Digits 2 and 3 (Target for Exp 3)
    mn_train, mn_test, _ = data_mod.get_mnist_task(classes=(2, 3))
    target_loader = (mn_train, mn_test)
    
    # Source 2: CIFAR-10 via MobileNetV2
    mob_train, mob_test = data_mod.get_mobilenet_features_task(classes=(2, 3), n_samples=1000)
    mob_source_loader = (mob_train, mob_test)
    
    runner = ExperimentRunner(data_module=data_mod, epochs=10, lr=0.05)

    # Experiment 1
    # Exp 1 measures Forgetting Drop on Task A after training Task B.
    # We use Fashion-MNIST (0,1) as Task A, and MNIST (2,3) as Task B.
    drop_base, drop_qtl = runner.run_experiment_1(HybridQuantumNet, source_loader, task_a_loader, target_loader)
    
    # Experiment 2
    # Measuring Forgetting on Task A when trained on Task B for Ansatz A, B, C.
    exp2_results = runner.run_experiment_2(HybridQuantumNet, task_a_loader, target_loader)
    
    # Experiment 3
    # Compare convergence on MNIST Target domain (Classes 2 vs 3).
    # Synthetic dataset serves as Source.
    scr_res, qtl_res, pre_res, mob_qtl_res, mob_pre_res = runner.run_experiment_3(HybridQuantumNet, source_loader, mob_source_loader, target_loader)
    
    # Plot Everything
    plot_results(
        exp1_data=(drop_base, drop_qtl),
        exp2_data=exp2_results,
        exp3_data=(scr_res, qtl_res, pre_res, mob_qtl_res, mob_pre_res)
    )

if __name__ == "__main__":
    main()
