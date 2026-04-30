"""
ProfileSplit - Distributed Training SERVER (v7)
=================================================
Fixes applied:
  - eval() before test evaluation
  - Cosine LR decay (manual, survives optimizer rebuild)
  - Clean protocol: only receives client-owned segments
  - Server holds segments[cut:] + head, optimizes only those

Usage:
    python3 sfl_real_server.py --port 9000 --epochs 5
"""

import argparse
import csv
import math
import os
import pickle
import socket
import struct

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


def build_segments_and_head(num_classes=10):
    if NEW_API:
       from torchvision.models import resnet18
       base = resnet18(weights=None)
    else:
        from torchvision.models import resnet18
        base = resnet18(pretrained=False)
    base.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    base.maxpool = nn.Identity()
    segments = nn.ModuleList([
        nn.Sequential(base.conv1, base.bn1, base.relu),
        base.layer1, base.layer2, base.layer3, base.layer4,
    ])
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.Linear(512, num_classes),
    )
    return segments, head


def cosine_lr(initial_lr, epoch, total_epochs):
    """Compute learning rate with cosine annealing."""
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


def run_server(port, epochs, lr, output_dir):
    device = torch.device("cpu")
    segments, head = build_segments_and_head()
    segments, head = segments.to(device), head.to(device)
    criterion = nn.CrossEntropyLoss()
    initial_lr = lr

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(1)
    print(f"[Server] Listening on port {port}...")
    conn, addr = sock.accept()
    print(f"[Server] Client connected from {addr}")

    config = recv_msg(conn)
    print(f"[Server] Config: {config}")
    send_msg(conn, {"type": "init_state", "segments_state": segments.state_dict()})
    print("[Server] Sent initial segment weights to client")

    os.makedirs(output_dir, exist_ok=True)
    results = []
    current_cut = None
    optimizer = None

    for epoch in range(epochs):
        # Cosine LR for this epoch
        epoch_lr = cosine_lr(initial_lr, epoch, epochs)
        
        epoch_loss, epoch_correct, epoch_total = 0, 0, 0

        while True:
            msg = recv_msg(conn)
            if msg is None:
                print("[Server] Disconnected"); conn.close(); sock.close(); return
            if msg["type"] == "epoch_done":
                break
            if msg["type"] == "activation":
                cut = msg["cut"]

                # Rebuild optimizer if cut changed (with current epoch's LR)
                if cut != current_cut:
                    current_cut = cut
                    tail_params = []
                    for i in range(cut, len(segments)):
                        tail_params.extend(segments[i].parameters())
                    tail_params.extend(head.parameters())
                    optimizer = optim.SGD(tail_params, lr=epoch_lr, momentum=0.9, weight_decay=5e-4)
                    print(f"  [Server] Cut=C{cut}, rebuilt optimizer ({sum(p.numel() for p in tail_params)} params, lr={epoch_lr:.4f})")
                else:
                    # Update LR in existing optimizer
                    for pg in optimizer.param_groups:
                        pg['lr'] = epoch_lr

                # Train mode for tail
                for i in range(cut, len(segments)):
                    segments[i].train()
                head.train()

                activation = msg["activation"].to(device).requires_grad_(True)
                labels = msg["labels"].to(device)

                x = activation
                for i in range(cut, len(segments)):
                    x = segments[i](x)
                output = head(x)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                grad = activation.grad.clone()
                _, pred = output.max(1)
                correct = pred.eq(labels).sum().item()

                send_msg(conn, {
                    "type": "gradient", "gradient": grad,
                    "loss": loss.item(), "correct": correct, "total": labels.size(0),
                })
                epoch_loss += loss.item() * labels.size(0)
                epoch_correct += correct
                epoch_total += labels.size(0)

        # Receive client weights for evaluation
        client_msg = recv_msg(conn)
        if client_msg["type"] == "client_weights":
            cut_used = client_msg["cut"]
            client_seg_state = client_msg["segments_state"]
            for i in range(cut_used):
                prefix = f"{i}."
                seg_state = {k[len(prefix):]: v for k, v in client_seg_state.items() if k.startswith(prefix)}
                if seg_state:
                    segments[i].load_state_dict(seg_state)

        # *** CRITICAL: eval() before testing ***
        for seg in segments:
            seg.eval()
        head.eval()

        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                x = inputs
                for seg in segments:
                    x = seg(x)
                out = head(x)
                _, pred = out.max(1)
                test_correct += pred.eq(labels).sum().item()
                test_total += labels.size(0)

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        test_acc = test_correct / test_total

        results.append({"epoch": epoch+1, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4),
                        "lr": round(epoch_lr, 6)})

        print(f"  Epoch {epoch+1}/{epochs} | Loss={train_loss:.4f} | "
              f"Train={train_acc:.3f} | Test={test_acc:.3f} | LR={epoch_lr:.4f}")

        send_msg(conn, {"type": "epoch_results", "epoch": epoch+1,
                        "test_acc": test_acc, "train_loss": train_loss})

    csv_path = os.path.join(output_dir, "server_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    send_msg(conn, {"type": "training_done"})
    conn.close(); sock.close()
    print(f"[Server] Done. Saved to {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="./convergence_results")
    a = p.parse_args()
    run_server(a.port, a.epochs, a.lr, a.output_dir)
