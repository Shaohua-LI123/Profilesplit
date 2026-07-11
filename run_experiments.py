"""
ProfileSplit - Unified Experiment Runner (v3 - Extended Channel Range)
======================================================================
Includes higher uplink rate scenarios (up to 200 Mbps) to capture
the crossover regime where Jetson Nano and Orin NX diverge.

Usage:
    cd ~/profilesplit
    python run_experiments.py
"""

import json
import math
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase4_controller'))

from controller import (
    ProfileSplitController, ProfileSplitConfig, CutProfile,
    ProfileSplitNoAdaptive, ProfileSplitNoPenalty, ProfileSplitNoStability
)
from baselines import (
    FixedShallow, FixedMid, FixedDeep, ChannelOnly, Oracle, RoundRobin
)


def load_profiles(profiling_dir="./profiling_results"):
    profiles = {}
    for device in ["jetson_nano", "orin_nano", "orin_nx"]:
        path = os.path.join(profiling_dir, f"profile_{device}_resnet18.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping {device}")
            continue
        with open(path, 'r') as f:
            profiles[device] = json.load(f)
        print(f"Loaded {len(profiles[device])} cut profiles for {device}")
    if not profiles:
        print("ERROR: No profiling data found!")
        sys.exit(1)
    return profiles


# ============================================================
# Channel Scenarios — Extended Range
# ============================================================
# Key insight: the Nano crossover happens at ~80-100 Mbps.
# Wi-Fi 5 (802.11ac) can reach 200+ Mbps in good conditions.
# Wi-Fi 6 (802.11ax) can reach 600+ Mbps.
# These are realistic rates for indoor edge deployments.

def generate_channel_trace(scenario, num_rounds, seed=42):
    rng = random.Random(seed)

    if scenario == "stable_high":
        # Good Wi-Fi 5: ~50 Mbps
        return [max(1.0, 50.0 + rng.gauss(0, 3)) for _ in range(num_rounds)]

    elif scenario == "stable_low":
        # Poor connection: ~3 Mbps
        return [max(0.5, 3.0 + rng.gauss(0, 0.5)) for _ in range(num_rounds)]

    elif scenario == "wifi5_good":
        # Strong Wi-Fi 5: ~150 Mbps (triggers crossover on Nano)
        return [max(10.0, 150.0 + rng.gauss(0, 15)) for _ in range(num_rounds)]

    elif scenario == "wifi6":
        # Wi-Fi 6: ~300 Mbps (deep in crossover regime)
        return [max(20.0, 300.0 + rng.gauss(0, 30)) for _ in range(num_rounds)]

    elif scenario == "fluctuating_wide":
        # Wide fluctuation: 5-200 Mbps sinusoidal (crosses the crossover boundary)
        return [max(1.0, 102 + 97 * math.sin(2 * math.pi * t / 200) + rng.gauss(0, 10))
                for t in range(num_rounds)]

    elif scenario == "fluctuating":
        # Original: 3-50 Mbps
        return [max(1.0, 25 + 22 * math.sin(2 * math.pi * t / 200) + rng.gauss(0, 3))
                for t in range(num_rounds)]

    elif scenario == "degrading":
        # 200 → 3 Mbps (crosses crossover on the way down)
        return [max(0.5, 200.0 - 197.0 * t / num_rounds + rng.gauss(0, 5))
                for t in range(num_rounds)]

    elif scenario == "bursty_wide":
        # Bursty between 5-200 Mbps (random jumps across crossover)
        trace, cur = [], 100.0
        for _ in range(num_rounds):
            if rng.random() < 0.05:
                cur = rng.choice([5.0, 20.0, 50.0, 100.0, 150.0, 200.0])
            cur = max(3.0, min(250.0, cur + rng.gauss(0, 5)))
            trace.append(cur)
        return trace

    elif scenario == "bursty":
        # Original: 3-50 Mbps
        trace, cur = [], 30.0
        for _ in range(num_rounds):
            if rng.random() < 0.05:
                cur = rng.choice([3.0, 10.0, 30.0, 50.0])
            cur = max(1.0, min(60.0, cur + rng.gauss(0, 1)))
            trace.append(cur)
        return trace

    elif scenario == "recovery":
        # 3 → 200 Mbps (crosses crossover on the way up)
        trace = []
        for t in range(num_rounds):
            if t < num_rounds * 0.3:
                base = 3.0
            elif t < num_rounds * 0.6:
                base = 3.0 + 197.0 * (t - num_rounds * 0.3) / (num_rounds * 0.3)
            else:
                base = 200.0
            trace.append(max(0.5, base + rng.gauss(0, 5)))
        return trace

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


SERVER_TIME_HP = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0}

def compute_round_cost(cut_index, device_type, uplink_rate_mbps, profiles):
    prof = [p for p in profiles[device_type] if p["cut_index"] == cut_index][0]
    comp_ms = prof["client_forward_ms"] + prof["client_backward_ms"]
    rate_bytes_per_ms = (uplink_rate_mbps * 1e6 / 8) / 1000
    tx_ms = prof["activation_size_bytes"] / max(rate_bytes_per_ms, 1.0)
    srv_ms = SERVER_TIME_HP[cut_index]
    return {
        "comp_ms": comp_ms,
        "tx_ms": tx_ms,
        "srv_ms": srv_ms,
        "total_latency_ms": comp_ms + tx_ms + srv_ms,
        "activation_bytes": prof["activation_size_bytes"],
    }


def run_experiment(strategy, device_type, channel_trace, profiles, scenario_name):
    results = {
        "strategy": strategy.name if hasattr(strategy, 'name') else type(strategy).__name__,
        "device": device_type,
        "scenario": scenario_name,
        "rounds": [],
    }
    total_latency, total_tx = 0.0, 0
    switches, prev_cut = 0, None

    for t, rate in enumerate(channel_trace):
        cut = strategy.select_cut(device_type, t, rate)
        cost = compute_round_cost(cut, device_type, rate, profiles)
        if prev_cut is not None and cut != prev_cut:
            switches += 1
        prev_cut = cut
        total_latency += cost["total_latency_ms"]
        total_tx += cost["activation_bytes"]
        results["rounds"].append({
            "round": t, "cut": cut,
            "rate_mbps": round(rate, 2),
            "latency_ms": round(cost["total_latency_ms"], 2),
        })

    n = len(channel_trace)
    results["summary"] = {
        "avg_latency_ms": round(total_latency / n, 2),
        "avg_tx_kb": round(total_tx / (1024 * n), 1),
        "num_switches": switches,
    }
    return results


def build_all_strategies(profiles):
    strategies = []
    strategies.append(FixedShallow())
    strategies.append(FixedMid())
    strategies.append(FixedDeep())

    first_dev = list(profiles.keys())[0]
    act_sizes = {p["cut_index"]: p["activation_size_bytes"] for p in profiles[first_dev]}
    strategies.append(ChannelOnly(activation_sizes=act_sizes))

    oracle = Oracle()
    for device in profiles:
        oracle.set_profiles_direct(device, {
            p["cut_index"]: {
                "comp_time_ms": p["client_forward_ms"] + p["client_backward_ms"],
                "activation_bytes": p["activation_size_bytes"],
                "energy_mj": p.get("client_energy_per_round_mj") or 0,
            } for p in profiles[device]
        })
    strategies.append(oracle)
    strategies.append(RoundRobin())

    # ProfileSplit
    ps = ProfileSplitController(ProfileSplitConfig(
        use_adaptive_weights=True, eta=15.0, tau=0.10))
    for device in profiles:
        ps.set_profiles_manual(device, {
            p["cut_index"]: CutProfile(
                cut_index=p["cut_index"],
                comp_time_ms=p["client_forward_ms"] + p["client_backward_ms"],
                activation_bytes=p["activation_size_bytes"],
                energy_mj=p.get("client_energy_per_round_mj") or 0,
            ) for p in profiles[device]
        })
    strategies.append(ps)

    # Ablation: Fixed Weights
    ps_fw = ProfileSplitNoAdaptive(alpha=1.0, beta=1.0, gamma=0.5, eta=15.0, tau=0.10)
    for device in profiles:
        ps_fw.set_profiles_manual(device, {
            p["cut_index"]: CutProfile(
                cut_index=p["cut_index"],
                comp_time_ms=p["client_forward_ms"] + p["client_backward_ms"],
                activation_bytes=p["activation_size_bytes"],
                energy_mj=p.get("client_energy_per_round_mj") or 0,
            ) for p in profiles[device]
        })
    strategies.append(ps_fw)

    # Ablation: No Penalty
    ps_np = ProfileSplitNoPenalty(use_adaptive_weights=True)
    for device in profiles:
        ps_np.set_profiles_manual(device, {
            p["cut_index"]: CutProfile(
                cut_index=p["cut_index"],
                comp_time_ms=p["client_forward_ms"] + p["client_backward_ms"],
                activation_bytes=p["activation_size_bytes"],
                energy_mj=p.get("client_energy_per_round_mj") or 0,
            ) for p in profiles[device]
        })
    strategies.append(ps_np)

    return strategies


def main():
    print("=" * 70)
    print("ProfileSplit - Three-Device Extended Experiment Suite")
    print("=" * 70)

    profiles = load_profiles("./profiling_results")

    for device in profiles:
        print(f"\n  {device} (batch={profiles[device][0]['batch_size']}):")
        for p in profiles[device]:
            comp = p["client_forward_ms"] + p["client_backward_ms"]
            print(f"    C{p['cut_index']}: comp={comp:.2f}ms, act={p['activation_size_bytes']/1024:.0f}KB")

    # Extended scenarios including high-rate ones
    scenarios = [
        "stable_low",        # 3 Mbps
        "stable_high",       # 50 Mbps
        "wifi5_good",        # 150 Mbps (triggers crossover)
        "wifi6",             # 300 Mbps (deep crossover)
        "fluctuating",       # 3-50 Mbps
        "fluctuating_wide",  # 5-200 Mbps (crosses boundary)
        "degrading",         # 200→3 Mbps
        "bursty",            # 3-50 Mbps
        "bursty_wide",       # 5-200 Mbps
        "recovery",          # 3→200 Mbps
    ]
    devices = list(profiles.keys())
    num_rounds = 500

    print(f"\nDevices:    {devices}")
    print(f"Scenarios:  {scenarios}")
    print(f"Rounds:     {num_rounds}")

    all_results = []

    for scenario in scenarios:
        channel_trace = generate_channel_trace(scenario, num_rounds)
        for device in devices:
            print(f"\n--- {scenario} / {device} ---")
            fresh = build_all_strategies(profiles)
            for strategy in fresh:
                result = run_experiment(strategy, device, channel_trace, profiles, scenario)
                all_results.append(result)
                s = result["summary"]
                print(f"  {result['strategy']:<28} "
                      f"Lat: {s['avg_latency_ms']:>8.1f}ms  "
                      f"TX: {s['avg_tx_kb']:>8.1f}KB  "
                      f"Sw: {s['num_switches']:>4}")

    # Summary tables
    seen = set()
    ordered = []
    for r in all_results:
        if r["strategy"] not in seen:
            ordered.append(r["strategy"])
            seen.add(r["strategy"])

    print("\n\n" + "=" * 130)
    print("LATENCY TABLE: Average Round Latency (ms)")
    print("=" * 130)
    for device in devices:
        print(f"\nDevice: {device}")
        header = f"  {'Strategy':<28}"
        for sc in scenarios:
            header += f" {sc[:11]:>12}"
        print(header)
        print("  " + "-" * (28 + 12 * len(scenarios)))
        for sname in ordered:
            row = f"  {sname:<28}"
            for sc in scenarios:
                match = [r for r in all_results
                         if r["strategy"] == sname and r["scenario"] == sc and r["device"] == device]
                row += f" {match[0]['summary']['avg_latency_ms']:>12.1f}" if match else f" {'N/A':>12}"
            print(row)

    # Key comparison
    print("\n\n" + "=" * 120)
    print("ProfileSplit vs Fixed-Deep (% improvement)")
    print("=" * 120)
    for device in devices:
        print(f"\nDevice: {device}")
        for sc in scenarios:
            ps = [r for r in all_results if r["strategy"] == "ProfileSplit" and r["scenario"] == sc and r["device"] == device]
            fd = [r for r in all_results if r["strategy"] == "Fixed-Deep" and r["scenario"] == sc and r["device"] == device]
            orc = [r for r in all_results if r["strategy"] == "Oracle" and r["scenario"] == sc and r["device"] == device]
            if ps and fd and orc:
                ps_l, fd_l, or_l = ps[0]["summary"]["avg_latency_ms"], fd[0]["summary"]["avg_latency_ms"], orc[0]["summary"]["avg_latency_ms"]
                ps_sw = ps[0]["summary"]["num_switches"]
                imp = (fd_l - ps_l) / fd_l * 100 if fd_l > 0 else 0
                gap = (ps_l - or_l) / or_l * 100 if or_l > 0 else 0
                m = " <<<" if imp > 1 else ""
                print(f"  {sc:<16} FD={fd_l:>7.1f}  PS={ps_l:>7.1f}  ORC={or_l:>7.1f}  "
                      f"imp={imp:>5.1f}%  gap={gap:>5.1f}%  sw={ps_sw}{m}")

    # Save
    output_dir = "./experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(output_dir, "summary.csv"), 'w') as f:
        f.write("strategy,device,scenario,avg_latency_ms,avg_tx_kb,num_switches\n")
        for r in all_results:
            s = r["summary"]
            f.write(f"{r['strategy']},{r['device']},{r['scenario']},"
                    f"{s['avg_latency_ms']},{s['avg_tx_kb']},{s['num_switches']}\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()