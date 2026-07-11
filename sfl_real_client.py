"""
ProfileSplit - Distributed Training CLIENT (v7)
=================================================
Fixes applied:
  - Explicit torch.autograd.backward for stability
  - Cosine LR decay (manual, survives optimizer rebuild)
  - Only sends client-owned segment weights
  - Client holds segments[0:cut], optimizes only those

Usage:
    python3 sfl_real_client.py --server-ip 192.168.0.99 --strategy fixed_c4 --channel wifi6 --epochs 5 --batch-size 32
"""

import argparse
import csv
import math
import os
import pickle
import random
import socket
import struct
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
    from torchvision.models import resnet18 as _r
    _t = _r(weights=None); del _t; NEW_API = True
except TypeError:
    NEW_API = False


def build_segments(num_classes=10):
    if NEW_API:
        from torchvision.models import resnet18
        base = resnet18(weights=None)
    else:
        from torchvision.models import resnet18
        base = resnet18(pretrained=False)
    base.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    base.maxpool = nn.Identity()
    return nn.ModuleList([
        nn.Sequential(base.conv1, base.bn1, base.relu),
        base.layer1, base.layer2, base.layer3, base.layer4,
    ])


def cosine_lr(initial_lr, epoch, total_epochs):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def send_msg(sock, data):
    raw = pickle.dumps(data)
    sock.sendall(struct.pack(">I", len(raw)) + raw)

def recv_msg(sock):
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk: return None
        raw_len += chunk
    msg_len = struct.unpack(">I", raw_len)[0]
    raw = b""
    while len(raw) < msg_len:
        chunk = sock.recv(min(msg_len - len(raw), 65536))
        if not chunk: return None
        raw += chunk
    return pickle.loads(raw)


def generate_channel(scenario, num_rounds, seed=42):
    rng = random.Random(seed)
    if scenario == "stable_low":
        return [max(0.5, 3.0 + rng.gauss(0, 0.5)) for _ in range(num_rounds)]
    elif scenario == "wifi6":
        return [max(20.0, 300.0 + rng.gauss(0, 30)) for _ in range(num_rounds)]
    elif scenario == "fluctuating_wide":
        return [max(1.0, 102 + 97 * math.sin(2 * math.pi * t / 300) + rng.gauss(0, 10))
                for t in range(num_rounds)]
    elif scenario == "stable_high":
        return [max(1.0, 50.0 + rng.gauss(0, 3)) for _ in range(num_rounds)]
    else:
        return [50.0] * num_rounds


class ProfileSplitSelector:
    def __init__(self):
        self.profiles = {
            1: {"comp_ms": 3.10,  "act_bytes": 1048576},
            2: {"comp_ms": 21.72, "act_bytes": 1048576},
            3: {"comp_ms": 43.63, "act_bytes": 524288},
            4: {"comp_ms": 76.77, "act_bytes": 262144},
        }
        self.prev_cut = 4
        self.eta = 15.0
        self.tau = 0.10
    def select(self, rate_mbps):
        r = (rate_mbps * 1e6 / 8) / 1000
        best_c, best_cost = self.prev_cut, float('inf')
        costs = {}
        for c, p in self.profiles.items():
            tx = p["act_bytes"] / max(r, 1.0)
            pen = self.eta if c != self.prev_cut else 0
            cost = p["comp_ms"] + tx + pen
            costs[c] = cost
            if cost < best_cost: best_cost, best_c = cost, c
        if best_c != self.prev_cut:
            if (costs[self.prev_cut] - best_cost) / max(costs[self.prev_cut], 1e-6) < self.tau:
                best_c = self.prev_cut
        self.prev_cut = best_c
        return best_c


def run_client(server_ip, port, strategy, channel_scenario, epochs,
               batch_size, lr, output_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Client] Device: {device}")
    if torch.cuda.is_available():
        print(f"[Client] GPU: {torch.cuda.get_device_name()}")

    segments = build_segments().to(device)
    initial_lr = lr

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    total_rounds = len(train_loader) * epochs
    channel_trace = generate_channel(channel_scenario, total_rounds)
    ps = ProfileSplitSelector()

    print(f"[Client] Connecting to {server_ip}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, port))
    print(f"[Client] Connected!")

    send_msg(sock, {"type": "config", "strategy": strategy,
                    "channel": channel_scenario, "epochs": epochs, "batch_size": batch_size})

    msg = recv_msg(sock)
    segments.load_state_dict(msg["segments_state"])
    print("[Client] Loaded initial segment weights")

    current_cut = None
    optimizer = None

    os.makedirs(output_dir, exist_ok=True)
    results = []
    round_idx = 0
    total_switches = 0
    prev_cut = None
    cumulative_wall_clock = 0.0

    for epoch in range(epochs):
        epoch_lr = cosine_lr(initial_lr, epoch, epochs)
        epoch_loss, epoch_correct, epoch_total = 0, 0, 0
        epoch_cuts = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)

            # Select cut
            rate = channel_trace[round_idx % len(channel_trace)]
            if strategy == "fixed_c4": cut = 4
            elif strategy == "fixed_c1": cut = 1
            elif strategy == "fixed_c2": cut = 2
            elif strategy == "profilesplit": cut = ps.select(rate)
            else: cut = 4

            if prev_cut is not None and cut != prev_cut:
                total_switches += 1
            prev_cut = cut
            epoch_cuts.append(cut)

            # Rebuild optimizer if cut changed (with current epoch's LR)
            if cut != current_cut:
                current_cut = cut
                client_params = []
                for i in range(cut):
                    client_params.extend(segments[i].parameters())
                optimizer = optim.SGD(client_params, lr=epoch_lr, momentum=0.9, weight_decay=5e-4) if client_params else None
                print(f"  [Client] Cut=C{cut}, rebuilt optimizer ({sum(p.numel() for p in client_params)} params, lr={epoch_lr:.4f})")
            else:
                # Update LR in existing optimizer
                if optimizer is not None:
                    for pg in optimizer.param_groups:
                        pg['lr'] = epoch_lr

            # Train mode for client segments
            for i in range(cut):
                segments[i].train()

            # Client forward
            t_start = time.time()
            x = inputs
            for i in range(cut):
                x = segments[i](x)
            activation = x
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Send to server
            send_msg(sock, {"type": "activation", "cut": cut,
                            "activation": activation.detach().cpu(), "labels": labels})

            # Receive gradient
            reply = recv_msg(sock)
            grad = reply["gradient"].to(device)

            # *** Client backward — explicit autograd.backward ***
            if optimizer is not None:
                optimizer.zero_grad()
            torch.autograd.backward(activation, grad)
            if optimizer is not None:
                optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.time()

            cumulative_wall_clock += (t_end - t_start) * 1000
            epoch_loss += reply["loss"] * reply["total"]
            epoch_correct += reply["correct"]
            epoch_total += reply["total"]
            round_idx += 1

        # Epoch done
        send_msg(sock, {"type": "epoch_done", "epoch": epoch + 1})

        # Send ONLY client-owned segments
        client_seg_state = {}
        for i in range(current_cut):
            for k, v in segments[i].state_dict().items():
                client_seg_state[f"{i}.{k}"] = v.cpu()
        send_msg(sock, {"type": "client_weights", "segments_state": client_seg_state, "cut": current_cut})

        epoch_results = recv_msg(sock)
        test_acc = epoch_results.get("test_acc", 0)

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        wall_sec = cumulative_wall_clock / 1000.0
        dominant_cut = Counter(epoch_cuts).most_common(1)[0][0]

        results.append({
            "epoch": epoch + 1, "strategy": f"{strategy}_{channel_scenario}",
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4), "wall_clock_sec": round(wall_sec, 1),
            "dominant_cut": dominant_cut, "switches_total": total_switches,
            "lr": round(epoch_lr, 6),
        })

        print(f"  Epoch {epoch+1}/{epochs} | Cut=C{dominant_cut} | "
              f"Loss={train_loss:.4f} | Train={train_acc:.3f} | "
              f"Test={test_acc:.3f} | Wall={wall_sec:.0f}s | Sw={total_switches} | LR={epoch_lr:.4f}")

    final_msg = recv_msg(sock)
    sock.close()

    csv_path = os.path.join(output_dir, f"client_{strategy}_{channel_scenario}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)

    print(f"\n[Client] Done! Final={results[-1]['test_acc']:.3f} "
          f"Best={max(r['test_acc'] for r in results):.3f} "
          f"Wall={results[-1]['wall_clock_sec']:.0f}s Sw={total_switches}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server-ip", type=str, required=True)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--strategy", type=str, default="fixed_c4",
                   choices=["fixed_c4", "fixed_c1", "fixed_c2", "profilesplit"])
    p.add_argument("--channel", type=str, default="wifi6",
                   choices=["stable_low", "stable_high", "wifi6", "fluctuating_wide"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="./convergence_results")
    a = p.parse_args()
    run_client(a.server_ip, a.port, a.strategy, a.channel, a.epochs, a.batch_size, a.lr, a.output_dir)
