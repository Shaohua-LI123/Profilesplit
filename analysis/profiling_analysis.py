"""
ProfileSplit - Profiling Analysis & Visualization
===================================================
Loads REAL profiling data from Phase 1 JSON files and produces
paper-ready tables, key findings, and analysis.

Usage:
    cd ~/profilesplit
    python analysis/profiling_analysis.py
"""

import json
import os
import csv
from typing import Dict, List


# ============================================================
# 1. Load Real Profiling Data
# ============================================================

def load_real_profiles(profiling_dir: str = "./profiling_results") -> Dict[str, List[Dict]]:
    """
    Load real profiling data from Phase 1 JSON outputs.
    Expects files named: profile_<device>_resnet18.json
    """
    all_profiles = {}
    
    for device in ["orin_nano", "orin_nx"]:
        path = os.path.join(profiling_dir, f"profile_{device}_resnet18.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_profiles[device] = json.load(f)
            print(f"Loaded {len(all_profiles[device])} profiles from {path}")
        else:
            print(f"WARNING: {path} not found!")
    
    return all_profiles


# ============================================================
# 2. Analysis Functions
# ============================================================

def compute_optimal_cut_per_rate(profiles: List[Dict],
                                 rates_mbps: List[float]) -> List[Dict]:
    """
    For a given device's profiles, compute which cut is optimal
    at each uplink rate. Produces the key finding:
    "Optimal cut depends on both device type AND channel condition."
    """
    results = []
    for rate in rates_mbps:
        rate_bytes_per_ms = (rate * 1e6 / 8) / 1000

        best_cut = None
        best_cost = float('inf')
        cut_costs = {}

        for p in profiles:
            comp = p["client_forward_ms"] + p["client_backward_ms"]
            tx = p["activation_size_bytes"] / max(rate_bytes_per_ms, 1.0)
            total = comp + tx
            cut_costs[p["cut_index"]] = {"comp": comp, "tx": tx, "total": total}

            if total < best_cost:
                best_cost = total
                best_cut = p["cut_index"]

        results.append({
            "rate_mbps": rate,
            "optimal_cut": best_cut,
            "optimal_cost_ms": best_cost,
            "all_costs": cut_costs,
        })

    return results


def generate_profiling_table(all_profiles: Dict) -> str:
    """Generate Table I for the paper."""
    lines = []
    lines.append("=" * 105)
    lines.append("TABLE I: Per-Cut-Point Profiling Results (ResNet-18, CIFAR-10)")
    lines.append("=" * 105)

    header = (f"{'Device':<12} {'Cut':<18} {'Fwd(ms)':<10} {'Bwd(ms)':<10} "
              f"{'Total(ms)':<11} {'Act(KB)':<10} {'Power(mW)':<11} "
              f"{'Energy(mJ)':<12} {'Params':<10}")
    lines.append(header)
    lines.append("-" * 105)

    for device in ["orin_nano", "orin_nx"]:
        if device not in all_profiles:
            continue
        for p in all_profiles[device]:
            total_ms = p["client_forward_ms"] + p["client_backward_ms"]
            act_kb = p["activation_size_bytes"] / 1024
            power = p.get("client_avg_power_mw")
            energy = p.get("client_energy_per_round_mj")
            power_str = f"{power:.0f}" if power else "N/A"
            energy_str = f"{energy:.1f}" if energy else "N/A"
            params = p.get("client_model_params", 0)

            lines.append(
                f"{device:<12} {p['cut_name']:<18} "
                f"{p['client_forward_ms']:<10.2f} {p['client_backward_ms']:<10.2f} "
                f"{total_ms:<11.2f} {act_kb:<10.1f} "
                f"{power_str:<11} {energy_str:<12} {params:<10,}"
            )
        lines.append("-" * 105)

    return '\n'.join(lines)


def generate_key_findings(all_profiles: Dict) -> str:
    """Generate key empirical findings that motivate ProfileSplit."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("KEY EMPIRICAL FINDINGS FROM PROFILING")
    lines.append("=" * 80)

    devices = [d for d in ["orin_nano", "orin_nx"] if d in all_profiles]
    if len(devices) < 2:
        lines.append("  Need profiles from both devices for comparison.")
        return '\n'.join(lines)

    # Finding 1: Speed ratio
    lines.append("\nFinding 1: Computation speed ratio across devices")
    for cut_idx in [1, 2, 3, 4]:
        p_nano = [p for p in all_profiles["orin_nano"] if p["cut_index"] == cut_idx]
        p_nx = [p for p in all_profiles["orin_nx"] if p["cut_index"] == cut_idx]
        if not p_nano or not p_nx:
            continue
        nano_total = p_nano[0]["client_forward_ms"] + p_nano[0]["client_backward_ms"]
        nx_total = p_nx[0]["client_forward_ms"] + p_nx[0]["client_backward_ms"]
        if nx_total > 0:
            ratio = nano_total / nx_total
            lines.append(f"  C{cut_idx}: Orin Nano is {ratio:.1f}x slower than Orin NX "
                         f"({nano_total:.2f}ms vs {nx_total:.2f}ms)")

    # Finding 2: Optimal cut at different rates
    lines.append("\nFinding 2: Optimal cut at different uplink rates")
    test_rates = [1, 3, 5, 10, 25, 50, 100]
    lines.append(f"  {'Rate(Mbps)':<14} {'Orin Nano':<20} {'Orin NX':<20}")
    lines.append("  " + "-" * 52)

    for rate in test_rates:
        row = f"  {rate:<14}"
        for device in ["orin_nano", "orin_nx"]:
            if device in all_profiles:
                res = compute_optimal_cut_per_rate(all_profiles[device], [rate])[0]
                row += f" C{res['optimal_cut']} ({res['optimal_cost_ms']:.1f}ms)"
                row += " " * max(0, 20 - len(f"C{res['optimal_cut']} ({res['optimal_cost_ms']:.1f}ms)"))
            else:
                row += f" {'N/A':<20}"
        lines.append(row)

    # Finding 3: Cost of wrong cut
    lines.append("\nFinding 3: Cost of using worst cut vs optimal")
    for rate in [5, 10, 50]:
        for device in devices:
            opt = compute_optimal_cut_per_rate(all_profiles[device], [rate])[0]
            opt_cost = opt["optimal_cost_ms"]
            worst_cost = max(opt["all_costs"][c]["total"] for c in opt["all_costs"])
            overhead = ((worst_cost - opt_cost) / opt_cost) * 100
            lines.append(f"  {device} @ {rate}Mbps: worst-cut overhead = {overhead:.0f}%")

    # Finding 4: Energy comparison
    lines.append("\nFinding 4: Energy per round across cuts")
    for device in devices:
        for p in all_profiles[device]:
            energy = p.get("client_energy_per_round_mj")
            if energy:
                lines.append(f"  {device} C{p['cut_index']}: {energy:.1f} mJ/round")

    return '\n'.join(lines)


# ============================================================
# 3. Save Functions
# ============================================================

def save_profiles_csv(all_profiles: Dict, output_dir: str):
    """Save as CSV for LaTeX tables."""
    os.makedirs(output_dir, exist_ok=True)
    for device, profiles in all_profiles.items():
        path = os.path.join(output_dir, f"profile_{device}_resnet18_analysis.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=profiles[0].keys())
            writer.writeheader()
            writer.writerows(profiles)
        print(f"Saved: {path}")


# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":
    print("ProfileSplit - Profiling Analysis")
    print("Loading real profiling data...\n")

    all_profiles = load_real_profiles("./profiling_results")

    if not all_profiles:
        print("ERROR: No profiling data found in ./profiling_results/")
        print("Run profiler.py on each device first.")
        exit(1)

    # Print Table I
    print(generate_profiling_table(all_profiles))

    # Print key findings
    print(generate_key_findings(all_profiles))

    # Save CSV
    save_profiles_csv(all_profiles, "./profiling_results")

    print("\nAnalysis complete.")