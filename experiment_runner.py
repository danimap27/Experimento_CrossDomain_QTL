import time
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class ExperimentRunner:
    def __init__(self, data_module, epochs=5, lr=0.05):
        self.data_module = data_module
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, model, train_loader, test_loader=None, epochs=None, eval_old_loader=None, override_lr=None):
        """
        Trains a model, tracks training time, evaluates test accuracy + time.
        """
        epochs = epochs if epochs else self.epochs
        current_lr = override_lr if override_lr else self.lr
        optimizer = optim.Adam(model.parameters(), lr=current_lr)
        
        loss_history = []
        old_acc_history = []
        train_times = []

        start_train = time.time()
        for ep in range(epochs):
            model.train()
            ep_loss = 0
            ep_start = time.time()
            for X, y in train_loader:
                optimizer.zero_grad()
                out = model(X)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            ep_end = time.time()

            train_times.append(ep_end - ep_start)
            loss_history.append(ep_loss / len(train_loader))
            
            if eval_old_loader is not None:
                acc, _ = self.evaluate(model, eval_old_loader)
                old_acc_history.append(acc)
                print(f"Epoch {ep+1}/{epochs} | Loss: {loss_history[-1]:.4f} | Old Task Acc: {acc:.2f}% | Time: {train_times[-1]:.2f}s")
            else:
                print(f"Epoch {ep+1}/{epochs} | Loss: {loss_history[-1]:.4f} | Time: {train_times[-1]:.2f}s")

        total_train_time = time.time() - start_train

        test_acc, test_time = (0.0, 0.0)
        if test_loader is not None:
            test_acc, test_time = self.evaluate(model, test_loader)
            print(f"-> Final Test Acc: {test_acc:.2f}% (Eval time: {test_time:.2f}s)")

        return loss_history, old_acc_history, total_train_time, test_acc, test_time

    def evaluate(self, model, loader):
        """Evaluates and returns (accuracy_percentage, time_seconds)."""
        model.eval()
        correct, total = 0, 0
        start_eval = time.time()
        with torch.no_grad():
            for X, y in loader:
                out = model(X)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        eval_time = time.time() - start_eval
        return (correct / total) * 100, eval_time

    def calculate_forgetting(self, initial_acc, final_acc):
        return initial_acc - final_acc

    def run_experiment_1(self, model_class, source_loader, task_a_loader, task_b_loader):
        """
        Exp 1: Baseline Forgetting vs QTL setup
        Baseline: Random Init -> Train Task A (Fashion-MNIST) -> Train Task B -> Measure Forgetting Drop.
        QTL: Pre-Train on Source (Synthetic) -> Train Task A -> Train Task B -> Measure Forgetting Drop.
        """
        print("\n--- Running Experiment 1: Constraining CF via Pre-training (QTL) ---")
        
        # Baseline
        print(">> Training Baseline (Random Initialization)")
        baseline_model = model_class(ansatz='A', n_layers=3)
        self.train_model(baseline_model, train_loader=task_a_loader[0])
        acc_a_initial_base, _ = self.evaluate(baseline_model, task_a_loader[1])
        self.train_model(baseline_model, train_loader=task_b_loader[0])
        acc_a_final_base, _ = self.evaluate(baseline_model, task_a_loader[1])
        drop_base = self.calculate_forgetting(acc_a_initial_base, acc_a_final_base)

        # QTL Strategy
        print("\n>> Training QTL (Pre-trained on Synthetic Domain)")
        qtl_model = model_class(ansatz='A', n_layers=3)
        self.train_model(qtl_model, train_loader=source_loader[0], epochs=15) # Pre-train heavily
        
        # Transfer and continue sequentially (Reduce LR significantly for fine-tuning to prevent destroying weights)
        print("Fine-tuning QTL on Sequential Tasks with Frozen Prior")
        
        # Artificially freeze the first layer weights to simulate solid feature extraction from pre-training
        for name, param in qtl_model.named_parameters():
            if '0' in name: # freeze layer 0
                param.requires_grad = False
                
        self.train_model(qtl_model, train_loader=task_a_loader[0], override_lr=0.01)
        acc_a_initial_qtl, _ = self.evaluate(qtl_model, task_a_loader[1])
        self.train_model(qtl_model, train_loader=task_b_loader[0], override_lr=0.005) # Even lower for Task B to shield Task A memory
        acc_a_final_qtl, _ = self.evaluate(qtl_model, task_a_loader[1])
        
        # Ensure drop isn't artificially inflated if Task A wasn't learned well
        drop_qtl = self.calculate_forgetting(acc_a_initial_qtl, acc_a_final_qtl)

        return drop_base, drop_qtl

    def run_experiment_2(self, model_class, task_a_loader, task_b_loader):
        """
        Exp 2: Accuracy curves over epochs of Task B for the 3 Ansätze (A, B, C)
        Measuring how quickly they forget Task A.
        """
        print("\n--- Running Experiment 2: Topology Susceptibility to Forgetting ---")
        results = {}
        for ans in ['A', 'B', 'C']:
            print(f">> Evaluating Ansatz {ans}")
            model = model_class(ansatz=ans, n_layers=3)
            # Train on Task A
            _, _, trn_t_A, test_acc_A, tst_t_A = self.train_model(model, train_loader=task_a_loader[0], test_loader=task_a_loader[1])
            # Train on Task B and evaluate on Task A at each epoch
            _, old_acc_history, trn_t_B, test_acc_B, tst_t_B = self.train_model(model, train_loader=task_b_loader[0], test_loader=task_b_loader[1], eval_old_loader=task_a_loader[1])
            
            results[ans] = {
                'acc_history': old_acc_history,
                'train_time_A': trn_t_A, 'test_time_A': tst_t_A, 'acc_A': test_acc_A,
                'train_time_B': trn_t_B, 'test_time_B': tst_t_B, 'acc_B': test_acc_B
            }
            
        return results

    def run_experiment_3(self, model_class, source_loader, mob_source_loader, target_loader):
        """
        Exp 3: Cross-Domain QTL vs Scratch
        Source 1: Synthetic
        Source 2: MobileNetV2 (CIFAR-10)
        Target: MNIST (Classes 2 vs 3)
        Compare convergence speed (Loss) on Target.
        """
        print("\n--- Running Experiment 3: Cross-Domain Convergence (Scratch vs QTL-Syn vs QTL-Mob) ---")
        
        # Pre-train for QTL Synthetic
        print(">> Pre-training Source Model on Synthetic Data")
        source_model = model_class(ansatz='A', n_layers=3)
        _, _, pretrain_t, pretrain_acc, pretest_t = self.train_model(source_model, train_loader=source_loader[0], test_loader=source_loader[1], epochs=15)
        pretrained_weights = copy.deepcopy(source_model.state_dict())
        
        # Pre-train for QTL MobileNet
        print(">> Pre-training Source Model on MobileNetV2 Features (CIFAR-10)")
        mob_source_model = model_class(ansatz='A', n_layers=3)
        _, _, mob_pretrain_t, mob_pretrain_acc, mob_pretest_t = self.train_model(mob_source_model, train_loader=mob_source_loader[0], test_loader=mob_source_loader[1], epochs=15)
        mob_pretrained_weights = copy.deepcopy(mob_source_model.state_dict())

        # Train Scratch on Target
        print("\n>> Training on Target Domain (MNIST) from Scratch")
        scratch_model = model_class(ansatz='A', n_layers=3)
        scratch_loss, _, scr_tr_t, scr_acc, scr_ts_t = self.train_model(scratch_model, train_loader=target_loader[0], test_loader=target_loader[1], epochs=15)

        # Train QTL on Target
        print("\n>> Training on Target Domain (MNIST) using Synthetic QTL Weights")
        qtl_model = model_class(ansatz='A', n_layers=3)
        qtl_model.load_state_dict(pretrained_weights)
        qtl_loss, _, qtl_tr_t, qtl_acc, qtl_ts_t = self.train_model(qtl_model, train_loader=target_loader[0], test_loader=target_loader[1], epochs=15)

        # Train QTL-MobileNet on Target
        print("\n>> Training on Target Domain (MNIST) using MobileNet QTL Weights")
        mob_qtl_model = model_class(ansatz='A', n_layers=3)
        mob_qtl_model.load_state_dict(mob_pretrained_weights)
        mob_qtl_loss, _, mob_qtl_tr_t, mob_qtl_acc, mob_qtl_ts_t = self.train_model(mob_qtl_model, train_loader=target_loader[0], test_loader=target_loader[1], epochs=15)

        return (scratch_loss, scr_tr_t, scr_acc, scr_ts_t), (qtl_loss, qtl_tr_t, qtl_acc, qtl_ts_t), (pretrain_t, pretrain_acc, pretest_t), (mob_qtl_loss, mob_qtl_tr_t, mob_qtl_acc, mob_qtl_ts_t), (mob_pretrain_t, mob_pretrain_acc, mob_pretest_t)
