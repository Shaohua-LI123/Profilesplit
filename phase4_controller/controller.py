"""
ProfileSplit Controller - Adaptive Split Layer Selection
=========================================================
Core algorithm implementing:

1. Profiling-guided cost estimation (offline lookup tables)
2. Channel-aware transmission cost (real-time uplink rate)
3. Split-switch penalty (penalizes frequent cut changes)
4. Stability threshold (hysteresis to prevent oscillation)
5. **Adaptive weighting** (weights shift based on current bottleneck)

Cost function:
    J_k(c, t) = α(t)·T̂_comp(c) + β(t)·T̂_tx(c, t) + γ(t)·Ê(c) + η·Δ(c, t)

Key insight: α(t), β(t), γ(t) are NOT fixed hyperparameters. They adapt
each round based on which component dominated the previous round's cost.
When the channel is poor, β rises automatically; when the device is weak,
α rises. This eliminates the need for manual per-deployment tuning.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# Data Structures
# ============================================================

@dataclass
class CutProfile:
    """Profiled costs for one cut point on one device (from Phase 1)."""
    cut_index: int
    comp_time_ms: float         # client forward + backward time
    activation_bytes: int       # raw float32 bytes to transmit
    energy_mj: float            # client energy per round
    server_comp_ms: float = 0.0 # server-side computation (optional)


@dataclass
class RoundDecision:
    """Record of one round's split selection decision (for paper figures)."""
    round_num: int
    device: str
    selected_cut: int
    uplink_rate_mbps: float
    # Per-cut costs evaluated
    costs_per_cut: Dict[int, float] = field(default_factory=dict)
    # Cost breakdown for selected cut
    comp_cost: float = 0.0
    tx_cost: float = 0.0
    energy_cost: float = 0.0
    switch_penalty: float = 0.0
    total_cost: float = 0.0
    # Adaptive weights at decision time
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    # Whether a switch actually occurred
    switched: bool = False


@dataclass
class ProfileSplitConfig:
    """All tunable parameters for ProfileSplit."""
    # --- Switch penalty ---
    eta: float = 15.0               # penalty for changing cut (ms-equivalent)
    
    # --- Stability threshold ---
    tau: float = 0.10               # only switch if new cost is >=tau lower
    
    # --- Adaptive weights ---
    use_adaptive_weights: bool = True
    weight_smoothing: float = 0.3   # EMA smoothing for weight updates
    alpha_init: float = 1.0         # initial computation weight
    beta_init: float = 1.0          # initial transmission weight
    gamma_init: float = 0.5         # initial energy weight
    
    # --- Rate estimation ---
    rate_ema_alpha: float = 0.3     # smoothing for uplink rate EMA
    min_rate_mbps: float = 0.5      # floor to prevent division by zero


# ============================================================
# ProfileSplit Controller
# ============================================================

class ProfileSplitController:
    """
    Lightweight split-layer controller.
    
    Offline: load profiling tables (Phase 1 output).
    Online:  each round, compute J_k(c,t) for each candidate cut,
             apply switch penalty + stability check, return best cut.
    
    The entire online decision is O(num_cuts) arithmetic — under 50 lines
    of core logic. This simplicity is intentional and is a selling point.
    """
    
    def __init__(self, config: ProfileSplitConfig = None):
        self.config = config or ProfileSplitConfig()
        self.name = "ProfileSplit"
        
        # Per-device profiling data: {device_type: {cut_index: CutProfile}}
        self.profiles: Dict[str, Dict[int, CutProfile]] = {}
        
        # Adaptive weights (shared across devices, evolve over time)
        self._alpha = self.config.alpha_init
        self._beta = self.config.beta_init
        self._gamma = self.config.gamma_init
        
        # Per-device state
        self._prev_cut: Dict[str, int] = {}
        self._prev_cost_breakdown: Dict[str, Dict[str, float]] = {}
        
        # Uplink rate EMA per device
        self._rate_ema: Dict[str, float] = {}
        
        # Full decision history (for paper figures)
        self.decision_history: List[RoundDecision] = []
    
    # --------------------------------------------------------
    # Profile Loading
    # --------------------------------------------------------
    
    def load_profiles_from_json(self, json_path: str, device_type: str):
        """Load Phase 1 profiling output."""
        with open(json_path, 'r') as f:
            raw = json.load(f)
        
        self.profiles[device_type] = {}
        for entry in raw:
            idx = entry["cut_index"]
            self.profiles[device_type][idx] = CutProfile(
                cut_index=idx,
                comp_time_ms=entry["client_forward_ms"] + entry["client_backward_ms"],
                activation_bytes=entry["activation_size_bytes"],
                energy_mj=entry.get("client_energy_per_round_mj") or 0.0,
                server_comp_ms=entry.get("server_forward_ms", 0) + entry.get("server_backward_ms", 0),
            )
        print(f"[ProfileSplit] Loaded {len(self.profiles[device_type])} cut profiles for {device_type}")
    
    def set_profiles_manual(self, device_type: str, profiles: Dict[int, CutProfile]):
        """Set profiles directly (for unit testing or synthetic experiments)."""
        self.profiles[device_type] = profiles
    
    # --------------------------------------------------------
    # Core: Adaptive Weight Update
    # --------------------------------------------------------
    
    def _update_adaptive_weights(self, device_type: str):
        """
        KEY NOVELTY: Adapt α, β, γ based on previous round's cost breakdown.
        
        Intuition:
        - If transmission dominated last round → increase β (pay more attention to channel)
        - If computation dominated → increase α (pay more attention to device speed)
        - If energy dominated → increase γ
        
        Implementation: Each weight is proportional to that component's fraction
        of total cost, smoothed by EMA to prevent erratic jumps.
        
        This eliminates the "who sets the weights?" criticism entirely.
        The system self-tunes based on which resource is currently the bottleneck.
        """
        if not self.config.use_adaptive_weights:
            return
        
        if device_type not in self._prev_cost_breakdown:
            return  # No history yet, keep initial weights
        
        prev = self._prev_cost_breakdown[device_type]
        comp = prev.get("comp", 0)
        tx = prev.get("tx", 0)
        energy = prev.get("energy", 0)
        
        total = comp + tx + energy
        if total <= 0:
            return
        
        # Target weights = fraction of total cost
        target_alpha = comp / total
        target_beta = tx / total
        target_gamma = energy / total
        
        # EMA smoothing
        s = self.config.weight_smoothing
        self._alpha = s * target_alpha + (1 - s) * self._alpha
        self._beta = s * target_beta + (1 - s) * self._beta
        self._gamma = s * target_gamma + (1 - s) * self._gamma
        
        # Normalize so they sum to a consistent scale (doesn't change argmin,
        # but makes costs interpretable across rounds)
        w_sum = self._alpha + self._beta + self._gamma
        if w_sum > 0:
            self._alpha /= w_sum
            self._beta /= w_sum
            self._gamma /= w_sum
    
    # --------------------------------------------------------
    # Core: Rate Estimation
    # --------------------------------------------------------
    
    def _update_rate_estimate(self, device_type: str, measured_rate_mbps: float) -> float:
        """
        Smooth the uplink rate with EMA.
        R̂_k(t) = ρ·R̂_k(t-1) + (1-ρ)·R_k(t)
        """
        measured = max(measured_rate_mbps, self.config.min_rate_mbps)
        
        if device_type not in self._rate_ema:
            self._rate_ema[device_type] = measured
        else:
            rho = self.config.rate_ema_alpha
            self._rate_ema[device_type] = (
                rho * measured + (1 - rho) * self._rate_ema[device_type]
            )
        
        return self._rate_ema[device_type]
    
    # --------------------------------------------------------
    # Core: Select Cut Point
    # --------------------------------------------------------
    
    def select_cut(self, device_type: str, round_num: int,
                   uplink_rate_mbps: float) -> int:
        """
        Main entry point. Select the optimal cut for this round.
        
        This is the function called every training round.
        Total computation: 4 multiplications + 4 additions per cut point.
        """
        
        if device_type not in self.profiles:
            raise ValueError(f"No profiles loaded for device '{device_type}'. "
                             f"Run Phase 1 profiler first.")
        
        profiles = self.profiles[device_type]
        prev_cut = self._prev_cut.get(device_type, None)
        
        # Step 1: Update adaptive weights from previous round
        self._update_adaptive_weights(device_type)
        
        # Step 2: Smooth the uplink rate
        rate_est = self._update_rate_estimate(device_type, uplink_rate_mbps)
        rate_bytes_per_ms = (rate_est * 1e6 / 8) / 1000  # convert to bytes/ms
        
        # Step 3: Evaluate cost for each candidate cut
        costs = {}
        breakdowns = {}
        
        for c, prof in profiles.items():
            # Computation cost (ms)
            comp = prof.comp_time_ms
            
            # Transmission cost (ms) — activation_bytes / rate
            tx = prof.activation_bytes / max(rate_bytes_per_ms, 1.0)
            
            # Energy cost (mJ, scaled to ms-equivalent for comparability)
            # Normalize: assume 1 mJ ≈ 1 ms in cost scale (adjustable)
            energy = prof.energy_mj
            
            # Switch penalty
            switch = self.config.eta if (prev_cut is not None and c != prev_cut) else 0.0
            
            # Weighted cost
            cost = (self._alpha * comp +
                    self._beta * tx +
                    self._gamma * energy +
                    switch)
            
            costs[c] = cost
            breakdowns[c] = {"comp": comp, "tx": tx, "energy": energy, "switch": switch}
        
        # Step 4: Find minimum-cost cut
        best_cut = min(costs, key=costs.get)
        best_cost = costs[best_cut]
        
        # Step 5: Stability check (hysteresis)
        switched = False
        if prev_cut is not None and best_cut != prev_cut:
            prev_cost = costs.get(prev_cut, float('inf'))
            improvement = (prev_cost - best_cost) / max(prev_cost, 1e-6)
            
            if improvement < self.config.tau:
                # Not enough improvement — stay with current cut
                best_cut = prev_cut
                best_cost = prev_cost
            else:
                switched = True
        
        # Step 6: Record state for next round
        self._prev_cut[device_type] = best_cut
        self._prev_cost_breakdown[device_type] = breakdowns[best_cut]
        
        # Step 7: Log decision
        decision = RoundDecision(
            round_num=round_num,
            device=device_type,
            selected_cut=best_cut,
            uplink_rate_mbps=uplink_rate_mbps,
            costs_per_cut=costs,
            comp_cost=breakdowns[best_cut]["comp"],
            tx_cost=breakdowns[best_cut]["tx"],
            energy_cost=breakdowns[best_cut]["energy"],
            switch_penalty=breakdowns[best_cut]["switch"],
            total_cost=best_cost,
            alpha=self._alpha,
            beta=self._beta,
            gamma=self._gamma,
            switched=switched,
        )
        self.decision_history.append(decision)
        
        return best_cut
    
    # --------------------------------------------------------
    # Analysis & Export
    # --------------------------------------------------------
    
    def get_switch_count(self, device_type: str = None) -> int:
        history = self.decision_history
        if device_type:
            history = [d for d in history if d.device == device_type]
        return sum(1 for d in history if d.switched)
    
    def get_weight_trajectory(self) -> List[Dict]:
        """For plotting adaptive weight evolution."""
        return [
            {"round": d.round_num, "alpha": d.alpha, "beta": d.beta, "gamma": d.gamma}
            for d in self.decision_history
        ]
    
    def get_cost_trajectory(self) -> List[Dict]:
        """For plotting per-round cost and its components."""
        return [
            {
                "round": d.round_num,
                "device": d.device,
                "cut": d.selected_cut,
                "total_cost": d.total_cost,
                "comp": d.comp_cost,
                "tx": d.tx_cost,
                "energy": d.energy_cost,
                "switch": d.switch_penalty,
                "rate_mbps": d.uplink_rate_mbps,
            }
            for d in self.decision_history
        ]
    
    def save_decision_history(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        records = []
        for d in self.decision_history:
            records.append({
                "round": d.round_num,
                "device": d.device,
                "cut": d.selected_cut,
                "rate_mbps": round(d.uplink_rate_mbps, 2),
                "cost": round(d.total_cost, 2),
                "comp": round(d.comp_cost, 2),
                "tx": round(d.tx_cost, 2),
                "energy": round(d.energy_cost, 2),
                "switch_pen": round(d.switch_penalty, 2),
                "alpha": round(d.alpha, 4),
                "beta": round(d.beta, 4),
                "gamma": round(d.gamma, 4),
                "switched": d.switched,
            })
        
        with open(path, 'w') as f:
            json.dump({
                "strategy": "ProfileSplit",
                "config": {
                    "eta": self.config.eta,
                    "tau": self.config.tau,
                    "adaptive": self.config.use_adaptive_weights,
                },
                "total_rounds": len(records),
                "total_switches": self.get_switch_count(),
                "decisions": records,
            }, f, indent=2)
        
        print(f"[ProfileSplit] Saved {len(records)} decisions to {path}")
    
    def summary(self, device_type: str = None) -> str:
        """Print a human-readable summary."""
        history = self.decision_history
        if device_type:
            history = [d for d in history if d.device == device_type]
        
        if not history:
            return "No decisions recorded."
        
        cuts_used = [d.selected_cut for d in history]
        from collections import Counter
        cut_dist = Counter(cuts_used)
        
        avg_cost = sum(d.total_cost for d in history) / len(history)
        switches = sum(1 for d in history if d.switched)
        
        lines = [
            f"ProfileSplit Summary ({device_type or 'all devices'}):",
            f"  Rounds: {len(history)}",
            f"  Avg cost: {avg_cost:.2f}",
            f"  Switches: {switches}",
            f"  Cut distribution: {dict(cut_dist)}",
            f"  Final weights: α={self._alpha:.3f}, β={self._beta:.3f}, γ={self._gamma:.3f}",
        ]
        return '\n'.join(lines)


# ============================================================
# Ablation Variants (for the paper)
# ============================================================

class ProfileSplitNoAdaptive(ProfileSplitController):
    """Ablation: ProfileSplit with FIXED weights (no adaptation)."""
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, **kwargs):
        config = ProfileSplitConfig(
            use_adaptive_weights=False,
            alpha_init=alpha,
            beta_init=beta,
            gamma_init=gamma,
            **kwargs,
        )
        super().__init__(config)
        self.name = "ProfileSplit-FixedWeights"


class ProfileSplitNoPenalty(ProfileSplitController):
    """Ablation: ProfileSplit without switch penalty (η=0)."""
    
    def __init__(self, **kwargs):
        config = ProfileSplitConfig(eta=0.0, tau=0.0, **kwargs)
        super().__init__(config)
        self.name = "ProfileSplit-NoPenalty"


class ProfileSplitNoStability(ProfileSplitController):
    """Ablation: ProfileSplit without stability threshold (τ=0)."""
    
    def __init__(self, **kwargs):
        config = ProfileSplitConfig(tau=0.0, **kwargs)
        super().__init__(config)
        self.name = "ProfileSplit-NoStability"