import socket
import struct
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from split_model import get_split_models


HOST = "0.0.0.0"
PORT = 5001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
LR = 0.01


def send_msg(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(conn):
    raw_len = recvall(conn, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = recvall(conn, msg_len)
    return pickle.loads(data)


def recvall(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", type=str, default="C2", choices=["C1", "C2", "C3", "C4"])
    args = parser.parse_args()

    cut_id = args.cut

    _, server_model = get_split_models(cut_id, num_classes=NUM_CLASSES)
    server_model = server_model.to(DEVICE)
    server_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(server_model.parameters(), lr=LR, momentum=0.9)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[SERVER] Listening on {HOST}:{PORT} with cut={cut_id} ...")

        conn, addr = s.accept()
        with conn:
            print(f"[SERVER] Connected by {addr}")

            while True:
                msg = recv_msg(conn)
                if msg is None:
                    print("[SERVER] Client disconnected.")
                    break

                if msg.get("type") == "close":
                    print("[SERVER] Close signal received.")
                    break

                if msg.get("type") != "forward":
                    print("[SERVER] Unknown message type.")
                    continue

                smashed = msg["activation"].to(DEVICE)
                labels = msg["labels"].to(DEVICE)

                smashed.requires_grad_(True)

                optimizer.zero_grad()
                outputs = server_model(smashed)
                loss = criterion(outputs, labels)
                loss.backward()

                grad_to_client = smashed.grad.detach().cpu()
                optimizer.step()

                pred = outputs.argmax(dim=1)
                acc = (pred == labels).float().mean().item()

                reply = {
                    "type": "backward",
                    "grad": grad_to_client,
                    "loss": float(loss.item()),
                    "acc": float(acc),
                }
                send_msg(conn, reply)
                print(f"[SERVER] cut={cut_id} loss={loss.item():.4f}, acc={acc:.4f}")


if __name__ == "__main__":
    main()