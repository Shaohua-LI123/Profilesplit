"""
ProfileSplit - Phase 1: Device Profiling Infrastructure
========================================================

Run on EACH Jetson device to measure per-cut-point costs.

Hardware:
  - "jetson_nano": NVIDIA Jetson Nano Developer Kit (4GB)
  - "orin_nano":   NVIDIA Jetson Orin Nano Developer Kit (8GB)
  - "orin_nx":     NVIDIA Jetson Orin NX

Usage:
  python profiler.py --device jetson_nano --model resnet18 --batch-size 4
  python profiler.py --device orin_nano   --model resnet18 --batch-size 8
  python profiler.py --device orin_nx     --model resnet18 --batch-size 8

  # Without power measurement (if no sudo access):
  python profiler.py --device orin_nano --model resnet18 --no-power
"""

import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# 1. Model Definitions with Explicit Cut Points
# ============================================================

class SplittableResNet18:
    """
    ResNet-18 split at natural cut points for CIFAR-10 (32x32).

    Cut points:
        C1: after stem (conv1 + bn1 + relu)
        C2: after layer1
        C3: after layer2
        C4: after layer3
    """

    def __init__(self, num_classes=10):
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.segments = nn.ModuleList([
            nn.Sequential(base.conv1, base.bn1, base.relu),
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        self.cut_names = ["C1_after_stem", "C2_after_layer1",
                          "C3_after_layer2", "C4_after_layer3"]
        self.num_cuts = 4

    def get_client_model(self, cut_index: int) -> nn.Sequential:
        return nn.Sequential(*list(self.segments[:cut_index]))

    def get_server_model(self, cut_index: int) -> nn.Sequential:
        return nn.Sequential(*list(self.segments[cut_index:]), self.head)


class SplittableMobileNetV3:
    """MobileNetV3-Small split at block boundaries."""

    def __init__(self, num_classes=10):
        base = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
        features = base.features

        self.segments = nn.ModuleList([
            nn.Sequential(*features[0:3]),
            nn.Sequential(*features[3:6]),
            nn.Sequential(*features[6:9]),
            nn.Sequential(*features[9:11]),
            nn.Sequential(features[11], base.avgpool, nn.Flatten()),
        ])
        self.head = base.classifier

        self.cut_names = ["C1_early", "C2_mid_early",
                          "C3_mid_late", "C4_late"]
        self.num_cuts = 4

    def get_client_model(self, cut_index: int) -> nn.Sequential:
        return nn.Sequential(*list(self.segments[:cut_index]))

    def get_server_model(self, cut_index: int) -> nn.Sequential:
        return nn.Sequential(*list(self.segments[cut_index:]), self.head)


# ============================================================
# 2. Power Monitoring via tegrastats
# ============================================================

class TegrastatsMonitor:
    """
    Read power from tegrastats (works on all Jetson devices).
    Parses VDD_IN or POM_5V_IN (total board power) from tegrastats output.
    Runs tegrastats in a background thread during profiling.

    NOTE: Requires sudo. Use --no-power flag if sudo not available.
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.power_samples = []
        self._process = None
        self._thread = None
        self._running = False

    def _parse_power(self, line: str) -> dict:
        """Extract power readings from a tegrastats line."""
        result = {}

        # --- Orin devices: VDD_IN 4788mW/4788mW ---
        m = re.search(r'VDD_IN\s+(\d+)mW', line)
        if m:
            result["vdd_in_mw"] = float(m.group(1))

        # --- Nano (Maxwell): POM_5V_IN 2118/2134 ---
        m = re.search(r'POM_5V_IN\s+(\d+)/(\d+)', line)
        if m:
            result["vdd_in_mw"] = float(m.group(1))

        # VDD_CPU_GPU_CV 536mW/536mW (Orin only)
        m = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
        if m:
            result["vdd_cpu_gpu_mw"] = float(m.group(1))

        # VDD_SOC 1403mW/1403mW (Orin only)
        m = re.search(r'VDD_SOC\s+(\d+)mW', line)
        if m:
            result["vdd_soc_mw"] = float(m.group(1))

        # POM_5V_GPU 166/166 (Nano only)
        m = re.search(r'POM_5V_GPU\s+(\d+)/(\d+)', line)
        if m:
            result["gpu_power_mw"] = float(m.group(1))

        # POM_5V_CPU 332/345 (Nano only)
        m = re.search(r'POM_5V_CPU\s+(\d+)/(\d+)', line)
        if m:
            result["cpu_power_mw"] = float(m.group(1))

        return result

    def _reader_thread(self):
        try:
            self._process = subprocess.Popen(
                ["sudo", "tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True,  # Compatible with Python 3.6+
            )
            while self._running and self._process.poll() is None:
                line = self._process.stdout.readline()
                if line:
                    power = self._parse_power(line)
                    if power:
                        power["timestamp"] = time.time()
                        self.power_samples.append(power)
        except Exception as e:
            print(f"[Power] tegrastats error: {e}")

    def start(self):
        self.power_samples = []
        self._running = True
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()
        time.sleep(0.5)

    def stop(self) -> dict:
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except:
                self._process.kill()
        if self._thread:
            self._thread.join(timeout=3)

        if not self.power_samples:
            return {"avg_power_mw": None, "samples": 0}

        vdd_in = [s["vdd_in_mw"] for s in self.power_samples if "vdd_in_mw" in s]
        cpu_gpu = [s["vdd_cpu_gpu_mw"] for s in self.power_samples if "vdd_cpu_gpu_mw" in s]

        return {
            "avg_vdd_in_mw": sum(vdd_in) / len(vdd_in) if vdd_in else None,
            "avg_cpu_gpu_mw": sum(cpu_gpu) / len(cpu_gpu) if cpu_gpu else None,
            "max_vdd_in_mw": max(vdd_in) if vdd_in else None,
            "samples": len(self.power_samples),
        }

    def measure_block(self, func, *args, **kwargs):
        self.start()
        result = func(*args, **kwargs)
        summary = self.stop()
        return result, summary


# ============================================================
# 3. Profiling Data Structure
# ============================================================

@dataclass
class CutPointProfile:
    device: str
    model_name: str
    cut_name: str
    cut_index: int
    batch_size: int
    client_forward_ms: float
    client_backward_ms: float
    server_forward_ms: float
    server_backward_ms: float
    activation_shape: str
    activation_size_bytes: int
    activation_elements: int
    client_peak_memory_mb: float
    client_model_params: int
    server_model_params: int
    client_avg_power_mw: Optional[float] = None
    client_max_power_mw: Optional[float] = None
    client_energy_per_round_mj: Optional[float] = None


# ============================================================
# 4. Profiling Engine
# ============================================================

def profile_cut_point(
    splittable_model, cut_index, device_name, model_name,
    input_shape=(8, 3, 32, 32), num_warmup=20, num_measure=100,
    use_gpu=True, measure_power=True,
) -> CutPointProfile:

    hw = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    cut_name = splittable_model.cut_names[cut_index - 1]
    print(f"\n  Profiling {cut_name} on {hw}...")

    client_model = splittable_model.get_client_model(cut_index).to(hw)
    server_model = splittable_model.get_server_model(cut_index).to(hw)
    client_model.train()
    server_model.train()

    client_params = sum(p.numel() for p in client_model.parameters())
    server_params = sum(p.numel() for p in server_model.parameters())

    x = torch.randn(*input_shape, device=hw)
    y = torch.zeros(input_shape[0], dtype=torch.long, device=hw)
    criterion = nn.CrossEntropyLoss()

    # ---- Warm-up ----
    print(f"    Warming up ({num_warmup} iters)...")
    for _ in range(num_warmup):
        a = client_model(x)
        o = server_model(a)
        loss = criterion(o, y)
        loss.backward()
        client_model.zero_grad()
        server_model.zero_grad()

    if use_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ---- Client Forward ----
    print(f"    Measuring client forward ({num_measure} iters)...")
    fwd_times = []
    for _ in range(num_measure):
        if use_gpu: torch.cuda.synchronize()
        t0 = time.perf_counter()
        act = client_model(x)
        if use_gpu: torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000)

    act_shape = tuple(act.shape)
    act_bytes = act.numel() * 4

    # ---- Server Forward ----
    print(f"    Measuring server forward ({num_measure} iters)...")
    sfwd_times = []
    for _ in range(num_measure):
        ad = act.detach().requires_grad_(True)
        if use_gpu: torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = server_model(ad)
        loss = criterion(o, y)
        if use_gpu: torch.cuda.synchronize()
        sfwd_times.append((time.perf_counter() - t0) * 1000)

    # ---- Backward (separate client and server) ----
    print(f"    Measuring backward ({num_measure} iters)...")
    cbwd_times = []
    sbwd_times = []
    for _ in range(num_measure):
        act_c = client_model(x)
        act_c.retain_grad()
        ad = act_c.detach().requires_grad_(True)
        o = server_model(ad)
        loss = criterion(o, y)

        # Server backward
        if use_gpu: torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        if use_gpu: torch.cuda.synchronize()
        sbwd_times.append((time.perf_counter() - t0) * 1000)

        # Client backward (with gradient from server)
        grad = ad.grad.clone()
        client_model.zero_grad()
        if use_gpu: torch.cuda.synchronize()
        t0 = time.perf_counter()
        act_c.backward(grad)
        if use_gpu: torch.cuda.synchronize()
        cbwd_times.append((time.perf_counter() - t0) * 1000)

        client_model.zero_grad()
        server_model.zero_grad()

    # ---- Peak memory ----
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if use_gpu else 0

    # ---- Power measurement ----
    avg_power, max_power, energy = None, None, None
    if measure_power:
        print(f"    Measuring power (50 rounds under tegrastats)...")
        monitor = TegrastatsMonitor(interval_ms=100)

        def run_rounds():
            for _ in range(50):
                a = client_model(x); a.retain_grad()
                ad = a.detach().requires_grad_(True)
                o = server_model(ad); l = criterion(o, y)
                l.backward(); g = ad.grad.clone()
                client_model.zero_grad()
                a.backward(g)
                client_model.zero_grad(); server_model.zero_grad()
                if use_gpu: torch.cuda.synchronize()

        t_start = time.perf_counter()
        _, psummary = monitor.measure_block(run_rounds)
        t_elapsed = time.perf_counter() - t_start

        if psummary["samples"] > 0 and psummary.get("avg_vdd_in_mw"):
            avg_power = psummary["avg_vdd_in_mw"]
            max_power = psummary["max_vdd_in_mw"]
            energy = avg_power * (t_elapsed / 50)  # mW * s = mJ per round
            print(f"    Power: avg={avg_power:.0f}mW, max={max_power:.0f}mW, "
                  f"energy/round={energy:.1f}mJ")
        else:
            print(f"    Power: no samples (need sudo?)")

    # ---- Median helper ----
    def med(lst):
        s = sorted(lst); n = len(s)
        return (s[n//2-1] + s[n//2]) / 2 if n % 2 == 0 else s[n//2]

    return CutPointProfile(
        device=device_name, model_name=model_name,
        cut_name=cut_name, cut_index=cut_index,
        batch_size=input_shape[0],
        client_forward_ms=med(fwd_times),
        client_backward_ms=med(cbwd_times),
        server_forward_ms=med(sfwd_times),
        server_backward_ms=med(sbwd_times),
        activation_shape=str(act_shape),
        activation_size_bytes=act_bytes,
        activation_elements=act.numel(),
        client_peak_memory_mb=peak_mem,
        client_model_params=client_params,
        server_model_params=server_params,
        client_avg_power_mw=avg_power,
        client_max_power_mw=max_power,
        client_energy_per_round_mj=energy,
    )


# ============================================================
# 5. Main
# ============================================================

def run_full_profiling(device_name, model_name="resnet18", batch_size=8,
                       num_warmup=20, num_measure=100, measure_power=True):

    print(f"\n{'='*60}")
    print(f"ProfileSplit Profiling: {device_name} / {model_name}")
    print(f"{'='*60}")

    try:
        hw_model = open("/proc/device-tree/model").read().strip('\x00').strip()
        print(f"  Hardware: {hw_model}")
    except:
        pass

    if model_name == "resnet18":
        splittable = SplittableResNet18(num_classes=10)
    else:
        splittable = SplittableMobileNetV3(num_classes=10)

    use_gpu = torch.cuda.is_available()
    print(f"  GPU: {torch.cuda.get_device_name() if use_gpu else 'None'}")
    print(f"  Batch: {batch_size}, Warmup: {num_warmup}, Measure: {num_measure}")

    profiles = []
    for ci in range(1, splittable.num_cuts + 1):
        p = profile_cut_point(splittable, ci, device_name, model_name,
                              (batch_size, 3, 32, 32), num_warmup, num_measure,
                              use_gpu, measure_power)
        profiles.append(p)
        total = p.client_forward_ms + p.client_backward_ms
        print(f"    → Client: {total:.2f}ms | Server: "
              f"{p.server_forward_ms + p.server_backward_ms:.2f}ms | "
              f"Act: {p.activation_size_bytes/1024:.1f}KB")

    return profiles


def save_profiles(profiles, output_dir="./profiling_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = profiles[0].device
    model = profiles[0].model_name

    json_path = os.path.join(output_dir, f"profile_{device}_{model}.json")
    with open(json_path, 'w') as f:
        json.dump([asdict(p) for p in profiles], f, indent=2)

    csv_path = os.path.join(output_dir, f"profile_{device}_{model}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(profiles[0]).keys()))
        w.writeheader()
        for p in profiles:
            w.writerow(asdict(p))

    print(f"\n{'='*90}")
    print(f"RESULTS: {device} / {model} (batch={profiles[0].batch_size})")
    print(f"{'='*90}")
    print(f"{'Cut':<18} {'Fwd(ms)':<10} {'Bwd(ms)':<10} {'Total(ms)':<11} "
          f"{'Act(KB)':<10} {'Power(mW)':<11} {'Energy(mJ)':<12}")
    print("-" * 90)
    for p in profiles:
        t = p.client_forward_ms + p.client_backward_ms
        pw = f"{p.client_avg_power_mw:.0f}" if p.client_avg_power_mw else "N/A"
        en = f"{p.client_energy_per_round_mj:.1f}" if p.client_energy_per_round_mj else "N/A"
        print(f"{p.cut_name:<18} {p.client_forward_ms:<10.2f} {p.client_backward_ms:<10.2f} "
              f"{t:<11.2f} {p.activation_size_bytes/1024:<10.1f} {pw:<11} {en:<12}")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProfileSplit Profiler")
    parser.add_argument("--device", type=str, required=True,
                        choices=["jetson_nano", "orin_nano", "orin_nx"])
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "mobilenetv3"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--no-power", action="store_true",
                        help="Skip power measurement (if no sudo)")
    parser.add_argument("--output-dir", type=str, default="./profiling_results")

    args = parser.parse_args()
    profiles = run_full_profiling(
        args.device, args.model, args.batch_size,
        args.warmup, args.measure, not args.no_power)
    save_profiles(profiles, args.output_dir)