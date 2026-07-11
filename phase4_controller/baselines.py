"""
ProfileSplit - Baseline Split Selection Strategies
====================================================
Implements all baseline methods for comparison with ProfileSplit.

Baselines:
    B1. Fixed-Shallow  : Always use C1 (after stem)
    B2. Fixed-Mid      : Always use C2 (after layer1)
    B3. Fixed-Deep     : Always use C4 (after layer3)
    B4. Channel-Only   : Pick cut based on uplink rate only, ignoring compute/energy
    B5. Oracle         : Perfect knowledge of all costs, no switching penalty
    B6. Round-Robin    : Cycle through cuts sequentially (sanity check)
    
    ProfileSplit: Our proposed method (in controller.py)

Each baseline implements the same interface:
    select_cut(device_type, round_num, uplink_rate_mbps, **kwargs) -> int

This makes it trivial to swap strategies in the training loop.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class SplitStrategy(ABC):
    """Base class for all split selection strategies."""

    def __init__(self, name: str, num_cuts: int = 4):
        self.name = name
        self.num_cuts = num_cuts
        self.selection_history = []  # Track decisions for analysis

    @abstractmethod
    def select_cut(self, device_type: str, round_num: int,
                   uplink_rate_mbps: float, **kwargs) -> int:
        """
        Select cut point for this round.
        
        Args:
            device_type: "nano" or "orin_nx"
            round_num: Current training round
            uplink_rate_mbps: Current estimated uplink rate
            **kwargs: Additional info (profiling data, previous cut, etc.)
            
        Returns:
            Cut index (1 to num_cuts)
        """
        pass

    def record_selection(self, round_num: int, cut_index: int,
                         uplink_rate_mbps: float, cost: Optional[float] = None):
        self.selection_history.append({
            "round": round_num,
            "cut": cut_index,
            "rate_mbps": uplink_rate_mbps,
            "cost": cost,
        })

    def get_switch_count(self) -> int:
        """Count how many times the cut point changed."""
        if len(self.selection_history) < 2:
            return 0
        return sum(
            1 for i in range(1, len(self.selection_history))
            if self.selection_history[i]["cut"] != self.selection_history[i - 1]["cut"]
        )

    def save_history(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "strategy": self.name,
                "total_rounds": len(self.selection_history),
                "total_switches": self.get_switch_count(),
                "history": self.selection_history,
            }, f, indent=2)


# ============================================================
# B1–B3: Fixed Cut Strategies
# ============================================================

class FixedShallow(SplitStrategy):
    """B1: Always cut at C1 (after stem). Maximum client-side offloading."""

    def __init__(self):
        super().__init__("Fixed-Shallow")

    def select_cut(self, device_type, round_num, uplink_rate_mbps, **kw):
        self.record_selection(round_num, 1, uplink_rate_mbps)
        return 1


class FixedMid(SplitStrategy):
    """B2: Always cut at C2 (after layer1). Balanced split."""

    def __init__(self):
        super().__init__("Fixed-Mid")

    def select_cut(self, device_type, round_num, uplink_rate_mbps, **kw):
        self.record_selection(round_num, 2, uplink_rate_mbps)
        return 2


class FixedDeep(SplitStrategy):
    """B3: Always cut at C4 (after layer3). Maximum client-side computation."""

    def __init__(self):
        super().__init__("Fixed-Deep")

    def select_cut(self, device_type, round_num, uplink_rate_mbps, **kw):
        self.record_selection(round_num, 4, uplink_rate_mbps)
        return 4


# ============================================================
# B4: Channel-Only Strategy
# ============================================================

class ChannelOnly(SplitStrategy):
    """
    B4: Select cut based purely on uplink rate thresholds.
    Does NOT use profiling data, energy, or switching penalty.
    
    Logic: When bandwidth is high, allow shallow cuts (large activations).
           When bandwidth is low, force deeper cuts (smaller activations).
    """

    def __init__(self, activation_sizes: Dict[int, int] = None):
        super().__init__("Channel-Only")
        # Default activation sizes (bytes) for ResNet-18 on CIFAR-10, batch=32
        # These are approximate; actual values come from profiling
        self.activation_sizes = activation_sizes or {
            1: 32 * 64 * 32 * 32 * 4,    # C1: ~8 MB
            2: 32 * 64 * 32 * 32 * 4,    # C2: ~8 MB
            3: 32 * 128 * 16 * 16 * 4,   # C3: ~4 MB
            4: 32 * 256 * 8 * 8 * 4,     # C4: ~2 MB
        }

    def select_cut(self, device_type, round_num, uplink_rate_mbps, **kw):
        # Simple: pick cut that keeps transmission time under a target
        target_tx_ms = 100.0  # target: activation sent within 100ms
        rate_bytes_per_ms = (uplink_rate_mbps * 1e6 / 8) / 1000  # bytes per ms

        best_cut = self.num_cuts  # default to deepest (smallest activation)
        for c in range(1, self.num_cuts + 1):
            tx_time = self.activation_sizes.get(c, 1e9) / rate_bytes_per_ms
            if tx_time <= target_tx_ms:
                best_cut = c
                break

        self.record_selection(round_num, best_cut, uplink_rate_mbps)
        return best_cut


# ============================================================
# B5: Oracle Strategy (Upper Bound)
# ============================================================

class Oracle(SplitStrategy):
    """
    B5: Oracle baseline with PERFECT knowledge.
    
    Evaluates the true cost of every cut point at every round using:
    - Exact profiled computation times
    - Exact current uplink rate
    - Exact activation sizes
    
    BUT: No switching penalty, no stability threshold.
    This represents the theoretical optimum for a greedy per-round selector.
    
    Purpose: Shows the ceiling of what any online method can achieve.
    If ProfileSplit is close to Oracle, it proves the lightweight
    approach loses little compared to perfect information.
    """

    def __init__(self, profiling_data: Dict = None):
        super().__init__("Oracle")
        self.profiling_data = profiling_data or {}

    def load_profiles(self, profile_path: str, device_type: str):
        """Load profiling JSON from Phase 1."""
        with open(profile_path, 'r') as f:
            profiles = json.load(f)
        self.profiling_data[device_type] = {}
        for p in profiles:
            cut_idx = p["cut_index"]
            self.profiling_data[device_type][cut_idx] = {
                "comp_time_ms": p["client_forward_ms"] + p["client_backward_ms"],
                "activation_bytes": p["activation_size_bytes"],
                "energy_mj": p.get("client_energy_per_round_mj", 0) or 0,
            }

    def set_profiles_direct(self, device_type: str, profiles: Dict):
        """Set profiles directly (for testing without files)."""
        self.profiling_data[device_type] = profiles

    def select_cut(self, device_type, round_num, uplink_rate_mbps, **kw):
        if device_type not in self.profiling_data:
            # Fallback: mid cut
            self.record_selection(round_num, 2, uplink_rate_mbps)
            return 2

        profiles = self.profiling_data[device_type]
        rate_bytes_per_ms = (uplink_rate_mbps * 1e6 / 8) / 1000

        best_cut = 1
        best_cost = float('inf')

        for c, p in profiles.items():
            c = int(c)
            tx_time_ms = p["activation_bytes"] / max(rate_bytes_per_ms, 1.0)
            # Oracle cost: just computation + transmission (no penalty terms)
            cost = p["comp_time_ms"] + tx_time_ms
            if cost < best_cost:
                best_cost = cost
                best_cut = c

        self.record_selection(round_num, best_cut, uplink_rate_mbps, best_cost)
        return best_cut


# ============================================================
# B6: Round-Robin (Sanity Check)
# ============================================================

class RoundRobin(SplitStrategy):
    """
    B6: Cycle through cuts 1→2→3→4→1→...
    
    Sanity check: tests whether diversity of cut points has
    any inherent benefit. Should generally perform poorly.
    """

    def __init__(self):
        super().__init__("Round-Robin")
        self._counter = 0

    def select_cut(self, device_type, round_num, uplink_rate_mbps, **kw):
        cut = (self._counter % self.num_cuts) + 1
        self._counter += 1
        self.record_selection(round_num, cut, uplink_rate_mbps)
        return cut


# ============================================================
# Factory
# ============================================================

def create_strategy(name: str, **kwargs) -> SplitStrategy:
    """Create a split strategy by name."""
    strategies = {
        "fixed_shallow": FixedShallow,
        "fixed_mid": FixedMid,
        "fixed_deep": FixedDeep,
        "channel_only": ChannelOnly,
        "oracle": Oracle,
        "round_robin": RoundRobin,
    }
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    return strategies[name](**kwargs)


def get_all_baseline_names() -> List[str]:
    return ["fixed_shallow", "fixed_mid", "fixed_deep",
            "channel_only", "oracle", "round_robin"]