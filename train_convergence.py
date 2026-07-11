"""
ProfileSplit - End-to-End Training Convergence Experiment (v2)
===============================================================
Answers three reviewer questions:
  Q1: Does dynamic cut switching affect convergence?
  Q2: Does it hurt final test accuracy?
  Q3: Does latency reduction translate to faster time-to-accuracy?

Runs on a single device (no network needed).
Three scenarios: stable-low (3Mbps), wifi6 (300Mbps), fluctuating-wide (5-200Mbps)
Three strategies: Fixed-C4, Fixed-C1, ProfileSplit

Usage:
    # Quick test (10 epochs):
    python3 train_convergence.py --quick

    # Full run (50 epochs, recommended):
    python3 train_convergence.py --epochs 50 --batch-size 32

    # On Jetson Nano (4GB RAM):
    python3 train_convergence.py --epochs 50 --batch-size 32
    
    # On Orin Nano/NX (8-16GB RAM):
    python3 train_convergence.py --epochs 50 --batch-size 64
"""

import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
    from torchvision.models import resnet18
    _t = resnet18(weights=None); del _t
    NEW_API = True
except TypeError:
    NEW_API = False


# ============================================================
# 1. Splittable ResNet-18
# ============================================================

class SplitResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        if NEW_API:
            base = resnet18(weights=None)
        else:
            base = resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        base.maxpool = nn.Identity()
        
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, num_classes),
        )
        self.segments = [self.stem, self.layer1, self.layer2,
                         self.layer3, self.layer4]

    def forward_split(self, x, cut):
        """Split forward: client runs segments[0:cut], server runs rest."""
        # Client forward
        for i in range(cut):
            x = self.segments[i](x)
        act = x
        # Detach to simulate network transfer
        act_server = act.detach().requires_grad_(True)
        # Server forward
        x = act_server
        for i in range(cut, len(self.segments)):
            x = self.segments[i](x)
        out = self.head(x)
        return out, act, act_server

    def forward(self, x):
        for seg in self.segments:
            x = seg(x)
        return self.head(x)


# ============================================================
# 2. Channel Simulation (per-round uplink rate)
# ============================================================

def generate_channel_for_training(scenario, num_rounds, seed=42):
    """Generate per-round uplink rates for training."""
    rng = random.Random(seed)
    
    if scenario == "stable_low":
        return [max(0.5, 3.0 + rng.gauss(0, 0.5)) for _ in range(num_rounds)]
    elif scenario == "wifi6":
        return [max(20.0, 300.0 + rng.gauss(0, 30)) for _ in range(num_rounds)]
    elif scenario == "fluctuating_wide":
        return [max(1.0, 102 + 97 * math.sin(2 * math.pi * t / 300) + rng.gauss(0, 10))
                for t in range(num_rounds)]
    else:
        return [50.0] * num_rounds


# ============================================================
# 3. ProfileSplit Cut Selector (channel-driven)
# ============================================================

class ProfileSplitSelector:
    """
    Simplified ProfileSplit for training experiment.
    Uses profiling data + current uplink rate to select cut.
    """
    def __init__(self, device_type="jetson_nano"):
        # Real profiling data from Jetson Nano (batch=4)
        # comp_ms = client forward + backward
        self.profiles = {
            1: {"comp_ms": 3.10,  "act_bytes": 1048576},  # C1: 1024KB
            2: {"comp_ms": 21.72, "act_bytes": 1048576},  # C2: 1024KB
            3: {"comp_ms": 43.63, "act_bytes": 524288},   # C3: 512KB
            4: {"comp_ms": 76.77, "act_bytes": 262144},   # C4: 256KB
        }
        self.prev_cut = 4
        self.eta = 15.0   # switch penalty (ms)
        self.tau = 0.10   # stability threshold
    
    def select(self, uplink_rate_mbps):
        rate_bpms = (uplink_rate_mbps * 1e6 / 8) / 1000
        
        best_cut, best_cost = self.prev_cut, float('inf')
        costs = {}
        for c, p in self.profiles.items():
            tx = p["act_bytes"] / max(rate_bpms, 1.0)
            penalty = self.eta if c != self.prev_cut else 0
            cost = p["comp_ms"] + tx + penalty
            costs[c] = cost
            if cost < best_cost:
                best_cost = cost
                best_cut = c
        
        # Stability check
        if best_cut != self.prev_cut:
            prev_cost = costs[self.prev_cut]
            improvement = (prev_cost - best_cost) / max(prev_cost, 1e-6)
            if improvement < self.tau:
                best_cut = self.prev_cut
        
        self.prev_cut = best_cut
        return best_cut
    
    def estimate_round_latency(self, cut, uplink_rate_mbps):
        """Estimate round latency for this cut at this rate."""
        rate_bpms = (uplink_rate_mbps * 1e6 / 8) / 1000
        p = self.profiles[cut]
        comp = p["comp_ms"]
        tx = p["act_bytes"] / max(rate_bpms, 1.0)
        server = 3.0  # estimated HP server time
        return comp + tx + server


# ============================================================
# 4. Training Functions
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, cut, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        out, act_client, act_server = model.forward_split(inputs, cut)
        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Propagate gradient back through client
        if act_server.grad is not None:
            act_client.backward(act_server.grad)
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += inputs.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += inputs.size(0)
    return correct / total


# ============================================================
# 5. Run One Complete Training Experiment
# ============================================================

def run_experiment(strategy_name, cut_fn, latency_fn, epochs, batch_size,
                   lr, device_str, channel_trace, seed=42):
    """
    Run training with a given cut strategy.
    Tracks both real compute time and simulated wall-clock time
    (which includes estimated network latency).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = torch.device(device_str)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_batches = len(train_loader)
    
    model = SplitResNet18(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    results = []
    cumulative_wall_clock_ms = 0.0
    total_switches = 0
    prev_cut = None
    round_idx = 0
    
    print(f"\n{'='*65}")
    print(f"Training: {strategy_name} | {epochs} epochs | batch={batch_size}")
    print(f"{'='*65}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0, 0, 0
        epoch_cuts = []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Select cut for this round
            rate = channel_trace[round_idx % len(channel_trace)]
            cut = cut_fn(rate)
            
            # Track switches
            if prev_cut is not None and cut != prev_cut:
                total_switches += 1
            prev_cut = cut
            epoch_cuts.append(cut)
            
            # Estimate simulated wall-clock (includes network latency)
            simulated_latency_ms = latency_fn(cut, rate)
            cumulative_wall_clock_ms += simulated_latency_ms
            
            # Actual training
            inputs, labels = inputs.to(device), labels.to(device)
            out, act_client, act_server = model.forward_split(inputs, cut)
            loss = criterion(out, labels)
            
            optimizer.zero_grad()
            loss.backward()
            if act_server.grad is not None:
                act_client.backward(act_server.grad)
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            _, pred = out.max(1)
            epoch_correct += pred.eq(labels).sum().item()
            epoch_total += inputs.size(0)
            round_idx += 1
        
        scheduler.step()
        
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        test_acc = evaluate(model, test_loader, device)
        wall_clock_sec = cumulative_wall_clock_ms / 1000.0
        
        # Most common cut this epoch
        from collections import Counter
        cut_dist = Counter(epoch_cuts)
        dominant_cut = cut_dist.most_common(1)[0][0]
        
        results.append({
            "epoch": epoch + 1,
            "strategy": strategy_name,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "wall_clock_sec": round(wall_clock_sec, 1),
            "dominant_cut": dominant_cut,
            "switches_total": total_switches,
        })
        
        print(f"  Epoch {epoch+1:>3}/{epochs} | Cut=C{dominant_cut} | "
              f"Loss={train_loss:.4f} | Train={train_acc:.3f} | "
              f"Test={test_acc:.3f} | WallClock={wall_clock_sec:.0f}s | "
              f"Sw={total_switches}")
    
    return results


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="./convergence_results")
    parser.add_argument("--quick", action="store_true", help="10 epochs quick test")
    args = parser.parse_args()
    
    epochs = 10 if args.quick else args.epochs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Scenarios to test
    scenarios = ["stable_low", "wifi6", "fluctuating_wide"]
    
    # Estimate total rounds per epoch
    num_train_samples = 50000
    rounds_per_epoch = num_train_samples // args.batch_size
    total_rounds = rounds_per_epoch * epochs
    
    # ProfileSplit selector
    ps = ProfileSplitSelector(device_type="jetson_nano")
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n\n{'#'*70}")
        print(f"SCENARIO: {scenario}")
        print(f"{'#'*70}")
        
        channel_trace = generate_channel_for_training(scenario, total_rounds)
        
        # Strategy definitions
        strategies = {
            "Fixed-C4": {
                "cut_fn": lambda rate: 4,
                "latency_fn": lambda cut, rate: ps.estimate_round_latency(4, rate),
            },
            "Fixed-C1": {
                "cut_fn": lambda rate: 1,
                "latency_fn": lambda cut, rate: ps.estimate_round_latency(1, rate),
            },
            "ProfileSplit": {
                "cut_fn": lambda rate: ps.select(rate),
                "latency_fn": lambda cut, rate: ps.estimate_round_latency(cut, rate),
            },
        }
        
        for strat_name, strat in strategies.items():
            # Reset ProfileSplit state for each run
            if strat_name == "ProfileSplit":
                ps.prev_cut = 4
            
            results = run_experiment(
                strategy_name=f"{strat_name}_{scenario}",
                cut_fn=strat["cut_fn"],
                latency_fn=strat["latency_fn"],
                epochs=epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device_str=args.device,
                channel_trace=channel_trace,
                seed=42,
            )
            all_results.extend(results)
    
    # Save all results
    csv_path = os.path.join(args.output_dir, "convergence_all.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    # Summary
    print("\n\n" + "=" * 90)
    print("CONVERGENCE SUMMARY")
    print("=" * 90)
    print(f"{'Strategy':<35} {'Final Acc':>10} {'Best Acc':>10} "
          f"{'WallClock(s)':>13} {'Switches':>9}")
    print("-" * 80)
    
    strat_names = list(dict.fromkeys(r["strategy"] for r in all_results))
    for sname in strat_names:
        sr = [r for r in all_results if r["strategy"] == sname]
        final = sr[-1]
        best = max(r["test_acc"] for r in sr)
        print(f"{sname:<35} {final['test_acc']:>9.3f} {best:>9.3f} "
              f"{final['wall_clock_sec']:>12.0f} {final['switches_total']:>9}")
    
    # Time-to-target analysis
    print("\n" + "=" * 90)
    print("TIME-TO-TARGET ACCURACY")
    print("=" * 90)
    targets = [0.70, 0.80, 0.85, 0.90]
    print(f"{'Strategy':<35}", end="")
    for t in targets:
        print(f" {'t@'+str(int(t*100))+'%':>10}", end="")
    print()
    print("-" * 80)
    
    for sname in strat_names:
        sr = [r for r in all_results if r["strategy"] == sname]
        row = f"{sname:<35}"
        for target in targets:
            reached = [r for r in sr if r["test_acc"] >= target]
            if reached:
                row += f" {reached[0]['wall_clock_sec']:>9.0f}s"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    print(f"\nResults saved to {csv_path}")
    print("Use this CSV to create convergence figures for the paper.")


if __name__ == "__main__":
    main()