import socket
import struct
import pickle
import time
import argparse
import csv
import os
import statistics

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from split_model import get_split_models


SERVER_HOST = "192.168.0.99"
SERVER_PORT = 5001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LR = 0.01
NUM_BATCHES = 50
WARMUP_BATCHES = 5


def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = recvall(sock, msg_len)
    return pickle.loads(data)


def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    return loader


def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def append_batch_csv(device_name, cut_id, rows, out_file="results_client_batches.csv"):
    file_exists = os.path.exists(out_file)
    with open(out_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "device", "cut", "batch_idx", "is_warmup",
                "loss", "acc", "forward_ms", "backward_ms",
                "total_ms", "activation_bytes"
            ])
        for row in rows:
            writer.writerow([
                device_name,
                cut_id,
                row["batch_idx"],
                row["is_warmup"],
                row["loss"],
                row["acc"],
                row["forward_ms"],
                row["backward_ms"],
                row["total_ms"],
                row["activation_bytes"],
            ])


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def safe_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_metric(rows, key):
    vals = [r[key] for r in rows]
    return {
        f"{key}_mean": safe_mean(vals),
        f"{key}_std": safe_std(vals),
        f"{key}_min": min(vals) if vals else float("nan"),
        f"{key}_max": max(vals) if vals else float("nan"),
    }


def append_summary_csv(device_name, cut_id, valid_rows, out_file="results_client_summary.csv"):
    file_exists = os.path.exists(out_file)

    forward_stats = summarize_metric(valid_rows, "forward_ms")
    backward_stats = summarize_metric(valid_rows, "backward_ms")
    total_stats = summarize_metric(valid_rows, "total_ms")
    loss_stats = summarize_metric(valid_rows, "loss")
    acc_stats = summarize_metric(valid_rows, "acc")

    activation_vals = [r["activation_bytes"] for r in valid_rows]
    activation_mean = safe_mean(activation_vals)
    activation_std = safe_std(activation_vals)

    with open(out_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "device", "cut",
                "num_valid_batches",
                "activation_bytes_mean", "activation_bytes_std",
                "forward_ms_mean", "forward_ms_std", "forward_ms_min", "forward_ms_max",
                "backward_ms_mean", "backward_ms_std", "backward_ms_min", "backward_ms_max",
                "total_ms_mean", "total_ms_std", "total_ms_min", "total_ms_max",
                "loss_mean", "loss_std", "loss_min", "loss_max",
                "acc_mean", "acc_std", "acc_min", "acc_max",
            ])

        writer.writerow([
            device_name,
            cut_id,
            len(valid_rows),
            activation_mean,
            activation_std,
            forward_stats["forward_ms_mean"],
            forward_stats["forward_ms_std"],
            forward_stats["forward_ms_min"],
            forward_stats["forward_ms_max"],
            backward_stats["backward_ms_mean"],
            backward_stats["backward_ms_std"],
            backward_stats["backward_ms_min"],
            backward_stats["backward_ms_max"],
            total_stats["total_ms_mean"],
            total_stats["total_ms_std"],
            total_stats["total_ms_min"],
            total_stats["total_ms_max"],
            loss_stats["loss_mean"],
            loss_stats["loss_std"],
            loss_stats["loss_min"],
            loss_stats["loss_max"],
            acc_stats["acc_mean"],
            acc_stats["acc_std"],
            acc_stats["acc_min"],
            acc_stats["acc_max"],
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", type=str, default="C2", choices=["C1", "C2", "C3", "C4"])
    parser.add_argument("--device_name", type=str, default="nx")
    parser.add_argument("--num_batches", type=int, default=NUM_BATCHES)
    parser.add_argument("--warmup_batches", type=int, default=WARMUP_BATCHES)
    args = parser.parse_args()

    cut_id = args.cut
    device_name = args.device_name
    num_batches = args.num_batches
    warmup_batches = args.warmup_batches

    if warmup_batches >= num_batches:
        raise ValueError("warmup_batches must be smaller than num_batches")

    client_model, _ = get_split_models(cut_id, num_classes=10)
    client_model = client_model.to(DEVICE)
    client_model.train()

    optimizer = optim.SGD(client_model.parameters(), lr=LR, momentum=0.9)
    loader = get_dataloader()

    all_rows = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_HOST, SERVER_PORT))
        print(f"[CLIENT] Connected to server {SERVER_HOST}:{SERVER_PORT} with cut={cut_id}")

        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= num_batches:
                break

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            t0 = time.perf_counter()
            smashed = client_model(images)
            t1 = time.perf_counter()

            smashed_detached = smashed.detach().cpu()
            labels_cpu = labels.detach().cpu()
            act_bytes = tensor_bytes(smashed_detached)

            send_msg(sock, {
                "type": "forward",
                "activation": smashed_detached,
                "labels": labels_cpu,
            })

            reply = recv_msg(sock)
            if reply is None:
                print("[CLIENT] Server disconnected.")
                break

            grad = reply["grad"].to(DEVICE)

            t2 = time.perf_counter()
            smashed.backward(grad)
            optimizer.step()
            t3 = time.perf_counter()

            forward_ms = (t1 - t0) * 1000
            backward_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000
            is_warmup = batch_idx < warmup_batches

            row = {
                "batch_idx": batch_idx,
                "is_warmup": is_warmup,
                "loss": reply["loss"],
                "acc": reply["acc"],
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "total_ms": total_ms,
                "activation_bytes": act_bytes,
            }
            all_rows.append(row)

            print(
                f"[CLIENT] cut={cut_id} batch={batch_idx} warmup={is_warmup} "
                f"loss={reply['loss']:.4f} acc={reply['acc']:.4f} "
                f"forward_ms={forward_ms:.2f} "
                f"backward_ms={backward_ms:.2f} "
                f"total_ms={total_ms:.2f} "
                f"activation_bytes={act_bytes}"
            )

        send_msg(sock, {"type": "close"})

    append_batch_csv(device_name, cut_id, all_rows)

    valid_rows = [r for r in all_rows if not r["is_warmup"]]
    append_summary_csv(device_name, cut_id, valid_rows)

    print(
        f"[CLIENT] Saved {len(all_rows)} batch rows to results_client_batches.csv "
        f"and {len(valid_rows)} valid rows summarized in results_client_summary.csv "
        f"for device={device_name}, cut={cut_id}"
    )


if __name__ == "__main__":
    main()