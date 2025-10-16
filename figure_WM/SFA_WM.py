from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# --- project imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import load_config
from src.model import ClusterModelNEST
import nest

default = load_config()


# =============================================================
# Helper dataclasses (UPDATED)
# =============================================================
@dataclass
class TrialSpec:
    trial: int
    cluster: int
    condition: str  # "ATTN", "CTRL", or "CTRL_PRE"
    # Absolute anchor for this trial (ms, relative to post-warmup zero)
    trial_start: float
    # All other times are RELATIVE TO trial_start (can be negative)
    t_stim_on: float
    t_stim_off: float
    t_attn_on: Optional[float] = None
    t_attn_off: Optional[float] = None


# =============================================================
# Utilities to (temporarily) set parameters on populations
# =============================================================

def _set_q_all_E(EI_Network, value: float):
    # EI_Network.Populations[0] is assumed to be a list of E cluster NodeCollections
    for pop in EI_Network.Populations[0]:
        pop.set({"q_stc": [float(value)]})


def _set_Ie_E_cluster(EI_Network, cluster_idx: int, value: float):
    EI_Network.Populations[0][int(cluster_idx)].set({"I_e": float(value)})


def _get_Ie_E_cluster(EI_Network, cluster_idx: int) -> float:
    """Read current baseline I_e for the given E cluster (assumes it's uniform)."""
    vals = EI_Network.Populations[0][int(cluster_idx)].get("I_e")
    try:
        # NEST NodeCollection.get typically returns a list of values
        return float(vals[0]) if hasattr(vals, "__len__") else float(vals)
    except Exception:
        return float(vals)


def _rheobase_from_baseline_Ie(base_Ie: float, I_th_E: float) -> float:
    """Estimate rheobase current from baseline I_e and the threshold factor I_th_E.
    Assumes baseline_Ie ≈ I_th_E * I_rheo.
    """
    if I_th_E == 0:
        raise ValueError("I_th_E must be non-zero to compute rheobase from baseline I_e.")
    return float(base_Ie) / float(I_th_E)


# =============================================================
# Simulation context (keeps absolute time + q schedule)
# =============================================================
class SimCtx:
    def __init__(self, EI, q_baseline: float):
        self.EI = EI
        self.q_baseline = float(q_baseline)
        self.q_current = float(q_baseline)
        self.t_abs = 0  # absolute simulated time (ms)
        self.t0 = 0     # post-warmup zero (ms)
        self.q_sched: List[Tuple[int, float]] = []

    def set_q(self, value: float):
        value = float(value)
        if float(self.q_current) != value:
            _set_q_all_E(self.EI, value)
            self.q_current = value
            self.q_sched.append((int(self.t_abs - self.t0), float(value)))

    def simulate(self, ms: int):
        ms = int(ms)
        if ms <= 0:
            return
        nest.Simulate(ms)
        self.t_abs += ms

    def warmup(self, warmup_ms: int, *, pre_first_ms: int = 0):
        # Baseline before warmup (optional)
        self.set_q(self.q_baseline)
        if pre_first_ms > 0:
            self.simulate(int(pre_first_ms))
        # Warmup
        self.simulate(int(warmup_ms))
        # Define relative zero after warmup
        self.t0 = int(self.t_abs)
        # Ensure schedule has a baseline marker at 0
        if not self.q_sched or self.q_sched[-1][0] != 0:
            self.q_sched.append((0, float(self.q_current)))


# =============================================================
# Per‑trial runners (one function per trial type)
# =============================================================

def _run_one_trial(
    sim: SimCtx,
    *,
    trial_idx: int,
    condition: str,
    cluster: int,
    iti_ms: int,
    q_attn: float,
    I_th_E: float,
    stim_Ie_pA: float,
    stim_fac_rheo: Optional[float],
    t_stim_on_rel: float,
    t_stim_off_rel: float,
    t_attn_on_rel: Optional[float],
    t_attn_off_rel: Optional[float],
) -> TrialSpec:
    """Generic trial executor.

    - Simulates the ITI *before* the trial. If any event has a negative time
      relative to trial_start, we allocate the last |min_negative| ms of the ITI
      to run those pre-on events. The baseline portion is (ITI - |min_negative|).
      If |min_negative| exceeds ITI, we truncate and warn.
    - Records `trial_start` at absolute time when relative t=0 is reached.
    - Steps through all event changes (attention/stimulus) in chronological order
      until the last event has been applied.
    """
    # Determine requested negative extent (ms)
    rel_times = [t_stim_on_rel]
    if t_attn_on_rel is not None:
        rel_times.append(t_attn_on_rel)
    min_rel_time = min(rel_times) if rel_times else 0.0
    neg_extent_req = max(0, int(round(-min_rel_time)))  # ms to simulate before t=0 with events

    # Effective pre-event slice cannot exceed ITI
    neg_extent_eff = min(neg_extent_req, int(iti_ms))
    if neg_extent_req > iti_ms:
        print(f"[WARN] Trial {trial_idx}: requested pre-on ({neg_extent_req} ms) exceeds ITI ({iti_ms} ms); truncating to {neg_extent_eff} ms.")

    # Adjust event times so nothing starts earlier than -neg_extent_eff
    if t_attn_on_rel is not None:
        t_attn_on_rel = max(t_attn_on_rel, -float(neg_extent_eff))
    t_stim_on_rel = max(t_stim_on_rel, -float(neg_extent_eff))

    # 1) ITI baseline part
    baseline_iti = int(iti_ms) - int(neg_extent_eff)
    if baseline_iti > 0:
        sim.set_q(sim.q_baseline)
        sim.simulate(baseline_iti)

    # 2) Build event schedule (relative to trial_start)
    events: List[Tuple[float, Tuple[str, str]]] = []  # (t_rel, (kind, action))
    # Stimulus events (always present)
    events.append((float(t_stim_on_rel), ("stim", "on")))
    events.append((float(t_stim_off_rel), ("stim", "off")))
    # Attention events (optional)
    if t_attn_on_rel is not None:
        events.append((float(t_attn_on_rel), ("attn", "on")))
    if t_attn_off_rel is not None:
        events.append((float(t_attn_off_rel), ("attn", "off")))

    # Enforce chronological order; ensure we will cross t=0 and record start
    events.sort(key=lambda x: (x[0], 0 if x[1][0] == "__mark" else 1))

    # 3) Pre-event window inside ITI (from -neg_extent_eff to first event or 0)
    # We'll walk the timeline from t_rel = -neg_extent_eff up to the last event.
    t_cursor_rel = -float(neg_extent_eff)

    # Helper to advance to a target relative time
    def _advance_to(target_rel: float):
        nonlocal t_cursor_rel
        dt = int(round(target_rel - t_cursor_rel))
        if dt > 0:
            sim.simulate(dt)
            t_cursor_rel += dt

    # Convenience: compute stimulus delta current
    base_Ie = _get_Ie_E_cluster(sim.EI, cluster)
    if stim_fac_rheo is not None:
        delta_I = float(stim_fac_rheo) * _rheobase_from_baseline_Ie(base_Ie, I_th_E)
    else:
        delta_I = float(stim_Ie_pA)

    # State flags
    stim_on = False
    attn_on = False

    # 4) Insert a synthetic marker at t=0 to capture trial_start *before* 0-time events
    events_with_mark = events + [(0.0, ("__mark", "start"))]
    events_with_mark.sort(key=lambda x: (x[0], 0 if x[1][0] == "__mark" else 1))

    trial_start_abs: Optional[float] = None

    # 5) Iterate through events in chronological order
    for t_rel, (kind, action) in events_with_mark:
        # Do not go earlier than our allowed pre-window
        if t_rel < -neg_extent_eff:
            continue
        # Advance time up to this change
        _advance_to(t_rel)

        if kind == "__mark" and action == "start" and trial_start_abs is None:
            trial_start_abs = float(sim.t_abs)  # record absolute time at t_rel == 0
            # continue without changing any state at t=0 (actual 0-time events will be processed next)
            continue

        if kind == "attn":
            if action == "on" and not attn_on:
                sim.set_q(q_attn)
                attn_on = True
            elif action == "off" and attn_on:
                sim.set_q(sim.q_baseline)
                attn_on = False
        elif kind == "stim":
            if action == "on" and not stim_on:
                _set_Ie_E_cluster(sim.EI, cluster, base_Ie + delta_I)
                stim_on = True
            elif action == "off" and stim_on:
                _set_Ie_E_cluster(sim.EI, cluster, base_Ie)
                stim_on = False

    # No more scheduled changes; ensure we finish in baseline/stim off state
    if stim_on:
        _set_Ie_E_cluster(sim.EI, cluster, base_Ie)
        stim_on = False
    if attn_on:
        sim.set_q(sim.q_baseline)
        attn_on = False

    # If trial_start_abs somehow wasn't set (e.g., all events < 0 and no marker processed),
    # we still need to advance to t=0 and mark it.
    if trial_start_abs is None:
        _advance_to(0.0)
        trial_start_abs = float(sim.t_abs)

    # Build TrialSpec (times are relative to trial_start)
    spec = TrialSpec(
        trial=trial_idx,
        cluster=int(cluster),
        condition=condition,
        trial_start=float(trial_start_abs - sim.t0),  # store relative to warmup zero
        t_stim_on=float(t_stim_on_rel),
        t_stim_off=float(t_stim_off_rel),
        t_attn_on=float(t_attn_on_rel) if t_attn_on_rel is not None else None,
        t_attn_off=float(t_attn_off_rel) if t_attn_off_rel is not None else None,
    )
    return spec


def run_trial_attn(
    sim: SimCtx,
    *,
    trial_idx: int,
    cluster: int,
    iti_ms: int,
    attn_dur_ms: int,
    stim_dur_ms: int,
    q_attn: float,
    I_th_E: float,
    stim_Ie_pA: float,
    stim_fac_rheo: Optional[float],
) -> TrialSpec:
    if attn_dur_ms < stim_dur_ms:
        print(f"[WARN] Trial {trial_idx} (ATTN): attention duration ({attn_dur_ms} ms) < stim duration ({stim_dur_ms} ms). Proceeding.")
    return _run_one_trial(
        sim,
        trial_idx=trial_idx,
        condition="ATTN",
        cluster=cluster,
        iti_ms=iti_ms,
        q_attn=q_attn,
        I_th_E=I_th_E,
        stim_Ie_pA=stim_Ie_pA,
        stim_fac_rheo=stim_fac_rheo,
        t_stim_on_rel=0.0,
        t_stim_off_rel=float(stim_dur_ms),
        t_attn_on_rel=80.0,
        t_attn_off_rel=float(attn_dur_ms),
    )


def run_trial_ctrl(
    sim: SimCtx,
    *,
    trial_idx: int,
    cluster: int,
    iti_ms: int,
    stim_dur_ms: int,
    I_th_E: float,
    stim_Ie_pA: float,
    stim_fac_rheo: Optional[float],
    q_attn: float,  # unused, kept for uniform signature
) -> TrialSpec:
    return _run_one_trial(
        sim,
        trial_idx=trial_idx,
        condition="CTRL",
        cluster=cluster,
        iti_ms=iti_ms,
        q_attn=q_attn,
        I_th_E=I_th_E,
        stim_Ie_pA=stim_Ie_pA,
        stim_fac_rheo=stim_fac_rheo,
        t_stim_on_rel=0.0,
        t_stim_off_rel=float(stim_dur_ms),
        t_attn_on_rel=None,
        t_attn_off_rel=None,
    )


def run_trial_ctrl_pre(
    sim: SimCtx,
    *,
    trial_idx: int,
    cluster: int,
    iti_ms: int,
    pre_attn_ms: int,
    post_attn_extra_ms: int,
    stim_dur_ms: int,
    q_attn: float,
    I_th_E: float,
    stim_Ie_pA: float,
    stim_fac_rheo: Optional[float],
) -> TrialSpec:
    # Attention ON before trial start, OFF inside following interval after stim
    t_attn_on_rel = -float(pre_attn_ms)
    t_attn_off_rel = float(stim_dur_ms + post_attn_extra_ms)
    return _run_one_trial(
        sim,
        trial_idx=trial_idx,
        condition="CTRL_PRE",
        cluster=cluster,
        iti_ms=iti_ms,
        q_attn=q_attn,
        I_th_E=I_th_E,
        stim_Ie_pA=stim_Ie_pA,
        stim_fac_rheo=stim_fac_rheo,
        t_stim_on_rel=0.0,
        t_stim_off_rel=float(stim_dur_ms),
        t_attn_on_rel=t_attn_on_rel,
        t_attn_off_rel=t_attn_off_rel,
    )


# =============================================================
# Trial protocol runner (UPDATED orchestrator)
# =============================================================

def run_trials_protocol_mixed(
    EI_Network,
    *,
    warmup_ms: int,
    n_trials: int,
    iti_ms: int = 1000,
    iti_range: Optional[Tuple[int, int]] = None,
    q_baseline: float,
    attn_factor: float,
    stim_Ie_pA: float,
    I_th_E: float,
    stim_fac_rheo: Optional[float] = None,
    stim_dur_ms: int = 100,
    attn_dur_ms: int = 500,
    pre_first_ms: int = 0,
    p_ctrl: float = 0.5,
    p_ctrl_pre: float = 0.0,
    pre_attn_ms: int = 500,
    post_attn_extra_ms: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[int, float]], List[TrialSpec]]:
    """
    Interleaved, pseudo-randomized sequence of ATTN, CTRL, and CTRL_PRE trials.

    Changes vs. previous version:
    - Adds `trial_start` (absolute) to TrialSpec and makes all other times relative to it.
    - No more "virtual" attention window for CTRL.
    - Allows attention window shorter than stimulation (emits a warning for ATTN).
    - Each trial is simulated by a dedicated function handling: ITI baseline, any
      pre-on events (negative times inside ITI), the trial itself, and any post
      changes belonging to that trial. CTRL_PRE trials can thus be longer than
      others; analysis windows remain relative to `trial_start`.
    """
    config = EI_Network.get_parameter()
    Q = int(config['network']['clusters']['count'])  # number of clusters
    rng = rng or np.random.default_rng()

    # Build pseudo-random order
    n_ctrl = int(round(p_ctrl * n_trials))
    n_ctrl_pre = int(round(p_ctrl_pre * n_trials))
    n_attn = int(max(0, n_trials - n_ctrl - n_ctrl_pre))
    total = n_ctrl + n_ctrl_pre + n_attn
    if total != n_trials:
        n_attn = max(0, n_trials - n_ctrl - n_ctrl_pre)
    order = np.array(["CTRL"] * n_ctrl + ["CTRL_PRE"] * n_ctrl_pre + ["ATTN"] * n_attn, dtype=object)
    rng.shuffle(order)

    # Simulation context (keeps absolute time and q schedule)
    sim = SimCtx(EI_Network, q_baseline)
    sim.warmup(warmup_ms, pre_first_ms=pre_first_ms)

    q_attn = float(q_baseline) * float(attn_factor)

    trials: List[TrialSpec] = []

    for i, cond in enumerate(order.tolist()):
        # Decide ITI for this trial
        if iti_range is not None:
            lo, hi = int(iti_range[0]), int(iti_range[1])
            if hi < lo:
                lo, hi = hi, lo
            iti_this = int(rng.integers(lo, hi + 1))
        else:
            iti_this = int(iti_ms)

        cluster = int(rng.integers(0, Q))

        if cond == "ATTN":
            spec = run_trial_attn(
                sim,
                trial_idx=i,
                cluster=cluster,
                iti_ms=iti_this,
                attn_dur_ms=int(attn_dur_ms),
                stim_dur_ms=int(stim_dur_ms),
                q_attn=q_attn,
                I_th_E=I_th_E,
                stim_Ie_pA=stim_Ie_pA,
                stim_fac_rheo=stim_fac_rheo,
            )
        elif cond == "CTRL":
            spec = run_trial_ctrl(
                sim,
                trial_idx=i,
                cluster=cluster,
                iti_ms=iti_this,
                stim_dur_ms=int(stim_dur_ms),
                I_th_E=I_th_E,
                stim_Ie_pA=stim_Ie_pA,
                stim_fac_rheo=stim_fac_rheo,
                q_attn=q_attn,
            )
        else:  # CTRL_PRE
            spec = run_trial_ctrl_pre(
                sim,
                trial_idx=i,
                cluster=cluster,
                iti_ms=iti_this,
                pre_attn_ms=int(pre_attn_ms),
                post_attn_extra_ms=int(post_attn_extra_ms),
                stim_dur_ms=int(stim_dur_ms),
                q_attn=q_attn,
                I_th_E=I_th_E,
                stim_Ie_pA=stim_Ie_pA,
                stim_fac_rheo=stim_fac_rheo,
            )

        trials.append(spec)

    # Ensure final q-schedule marker at end
    t_end_rel = int(sim.t_abs - sim.t0)
    if not sim.q_sched or sim.q_sched[-1][0] != t_end_rel:
        sim.q_sched.append((t_end_rel, float(sim.q_current)))

    return sim.q_sched, trials


# =============================================================
# Analysis helpers (UPDATED to use trial_start)
# =============================================================

def bin_cluster_rates(
    spike_times_ms: np.ndarray,
    spike_ids: np.ndarray,
    N_E: int,
    Q: int,
    bin_ms: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    t_ms     : (nbins,) bin centers (absolute, ms since post-warmup)
    rates_hz : (nbins, Q) cluster-wise E rates
    dom      : (nbins,) dominant cluster per bin
    """
    E_PER_CLUST = N_E // Q

    # E-only
    exc_mask = spike_ids < N_E
    tE = spike_times_ms[exc_mask]
    idE = spike_ids[exc_mask]

    T = float(tE.max() if tE.size else 0.0)
    nbins = int(np.ceil(T / bin_ms))
    edges = np.arange(nbins + 1) * bin_ms
    t_bins = (edges[:-1] + edges[1:]) / 2.0

    counts = np.zeros((nbins, Q), dtype=np.int32)
    if tE.size:
        b = np.clip((tE // bin_ms).astype(int), 0, nbins - 1)
        cl = (idE // E_PER_CLUST).astype(int)
        np.add.at(counts, (b, cl), 1)

    bin_s = bin_ms / 1000.0
    rates = counts / (bin_s * E_PER_CLUST)
    dom = np.argmax(rates, axis=1)
    return t_bins, rates, dom


def dominant_and_success_over_window(
    t_bins: np.ndarray,
    rates: np.ndarray,
    trials: List[TrialSpec],
    win_start_ms: int = -50,
    win_end_ms: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each trial, compute the dominant cluster by averaging cluster rates over
    a configurable window RELATIVE TO trial_start (t=0):
    [trial_start + win_start_ms, trial_start + win_end_ms].

    Returns
    -------
    pred : (n_trials,) predicted dominant cluster index per trial (=-1 if no bins)
    succ : (n_trials,) bool array whether pred == stimulated cluster
    """
    preds: List[int] = []
    succs: List[bool] = []

    # Ensure proper ordering of window bounds
    w0, w1 = (win_start_ms, win_end_ms)
    if w1 < w0:
        w0, w1 = w1, w0

    for tr in trials:
        t0 = tr.trial_start + w0
        t1 = tr.trial_start + w1 + 1e-6
        idx = np.where((t_bins >= t0) & (t_bins <= t1))[0]
        if idx.size == 0:
            preds.append(-1)
            succs.append(False)
            continue
        mean_rates = rates[idx].mean(axis=0)  # average over window
        pred = int(np.argmax(mean_rates))
        preds.append(pred)
        succs.append(pred == tr.cluster)

    return np.array(preds, dtype=int), np.array(succs, dtype=bool)

    # =============================================================
    # Moving-window retention curve (NEW)
    # =============================================================

def moving_retention_success(
        t_bins: np.ndarray,
        rates: np.ndarray,
        trials: List[TrialSpec],
        *,
        win_ms: int = 100,
        t_min_ms: int = -200,
        t_max_ms: int = 1200,
        step_ms: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute condition-wise retention success curves using a moving window.

    For each time *t* (relative to trial start), we average cluster rates over
    [trial_start + t, trial_start + t + win_ms] for each trial, predict the
    dominant cluster, and mark success if it equals the stimulated cluster.
    We then aggregate per condition (ATTN/CTRL/CTRL_PRE) to get success rates.

    Returns
    -------
    t_axis     : (K,) array of window *start* times (ms, rel. trial start)
    curves     : dict(cond -> (K,) success rate in [0,1] or NaN if no trials)
    counts     : dict(cond -> (K,) number of trials contributing at each t)
    """
    if step_ms is None or step_ms <= 0:
        step_ms = max(1, int(win_ms // 5))

    t_axis = np.arange(int(t_min_ms), int(t_max_ms) - int(win_ms) + 1, int(step_ms))
    conds = np.array([tr.condition for tr in trials])
    clusters = np.array([tr.cluster for tr in trials], dtype=int)
    tstarts = np.array([tr.trial_start for tr in trials], dtype=float)

    uniq_conds = ["ATTN", "CTRL", "CTRL_PRE"]
    curves: Dict[str, np.ndarray] = {c: np.full(t_axis.shape, np.nan, dtype=float) for c in uniq_conds}
    counts: Dict[str, np.ndarray] = {c: np.zeros_like(t_axis, dtype=int) for c in uniq_conds}

    for k, t0 in enumerate(t_axis):
        # Per-trial outcome at this window
        succ_k: List[Optional[bool]] = [None] * len(trials)
        for i, tr in enumerate(trials):
            a = tstarts[i] + t0
            b = a + win_ms + 1e-6
            idx = np.where((t_bins >= a) & (t_bins <= b))[0]
            if idx.size == 0:
                continue  # no data for this trial at this window
            mean_rates = rates[idx].mean(axis=0)
            pred = int(np.argmax(mean_rates))
            succ_k[i] = bool(pred == clusters[i])

        # Aggregate per condition
        for c in uniq_conds:
            mask_c = (conds == c)
            vals = [succ for i, succ in enumerate(succ_k) if mask_c[i] and succ is not None]
            if len(vals) > 0:
                curves[c][k] = float(np.mean(vals))
                counts[c][k] = int(len(vals))
            # else: remain NaN / 0

    return t_axis, curves, counts


# =============================================================
# Raster plotting (UPDATED to use trial_start & optional attention)
# =============================================================

def _raster_panel(
    ax,
    spike_times_ms: np.ndarray,
    spike_ids: np.ndarray,
    tr: TrialSpec,
    N_E: int,
    N_I: int,
    Q: int,
    check_win_start_ms: int,
    check_win_end_ms: int,
    raster_stride: int,
    pad_pre_ms: int = 200,
    pad_post_ms: int = 1200,
) -> Tuple[float, float]:
    """
    Plot a *subset* of neurons (respecting raster_stride), aligned at **trial start** (t=0).
    E spikes in black; I spikes (ids >= N_E) in dark red at the top.

    X-range is fixed to [-pad_pre_ms, +pad_post_ms] relative to trial start.
    Overlays: trial start (green), stimulus (red), attention span (blue, if present),
    and the retention check window (dashed).
    """
    E_PER_CLUST = N_E // Q

    # Fixed window relative to trial start
    xmin_rel = -float(pad_pre_ms)
    xmax_rel = float(pad_post_ms)

    # Subset spikes and align to trial start
    t_rel = spike_times_ms - tr.trial_start
    m = (t_rel >= xmin_rel) & (t_rel <= xmax_rel)
    tt = t_rel[m]
    ii = spike_ids[m]

    # Keep only every Nth neuron id
    if int(raster_stride) > 1:
        keep = (ii % int(raster_stride)) == 0
        tt = tt[keep]
        ii = ii[keep]

    # Plot: E black, I dark red (I at top due to higher ids)
    mE = ii < N_E
    mI = ~mE
    if np.any(mE):
        ax.scatter(tt[mE], ii[mE], s=1, c='black')
    if np.any(mI):
        ax.scatter(tt[mI], ii[mI], s=1, c='#8B0000')

    # Shade chosen E-cluster band (neuron-id coordinates)
    y0 = tr.cluster * E_PER_CLUST
    y1 = (tr.cluster + 1) * E_PER_CLUST
    ax.axhspan(y0, y1, alpha=0.10)

    ax.set_ylim([0-0.1*E_PER_CLUST, N_E+N_I+0.75*E_PER_CLUST])

    # Attention overlay in BLUE (if present)
    if tr.t_attn_on is not None and tr.t_attn_off is not None:
        # Clip overlay to panel x-lims
        attn_start_rel = float(tr.t_attn_on)
        attn_end_rel = float(tr.t_attn_off)
        ymin, ymax = ax.get_ylim()
        width = max(0.0, attn_end_rel - attn_start_rel)
        ax.add_patch(Rectangle((attn_start_rel, ymin),
                               width,
                               ymax - ymin,
                               linewidth=0, facecolor='tab:blue', alpha=0.12))

    # Stimulus window (red rectangle over chosen cluster band)
    stim_rel_start = float(tr.t_stim_on)
    stim_rel_end = float(tr.t_stim_off)
    if stim_rel_end > stim_rel_start:
        rect = Rectangle((stim_rel_start, y0), stim_rel_end - stim_rel_start, y1 - y0,
                         linewidth=0, facecolor='red', alpha=0.2)
        ax.add_patch(rect)
        ymin, ymax = ax.get_ylim()
        ax.hlines(ymax - 0.02 * (ymax - ymin), stim_rel_start, stim_rel_end, colors='red', linewidth=2)

    # Retention window (relative to trial start)
    w0_rel = float(check_win_start_ms)
    w1_rel = float(check_win_end_ms)
    if w1_rel < w0_rel:
        w0_rel, w1_rel = w1_rel, w0_rel
    ax.axvline(w0_rel, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(w1_rel, color='black', linestyle='--', linewidth=1.5)

    # Trial start marker (green)
    ax.axvline(0.0, color='tab:green', linewidth=1.2)

    # Axes cosmetics
    ax.set_ylabel("Neuron id")
    ax.set_xlim(xmin_rel, xmax_rel)

    return float(xmin_rel), float(xmax_rel)



def plot_three_rasters(
    spike_times_ms: np.ndarray,
    spike_ids: np.ndarray,
    N_E: int,
    N_I: int,
    trials: List[TrialSpec],
    idx_attn: int,
    idx_ctrl: int,
    idx_ctrl_pre: Optional[int],
    Q: int,
    check_win_start_ms: int = -50,
    check_win_end_ms: int = 0,
    raster_stride: int = 10,
    pad_pre_ms: int = 200,
    pad_post_ms: int = 200,
    outpath: Optional[str] = None,
):
    """Plot ATTN / CTRL / CTRL_PRE (if available), aligned at trial start (t=0)."""
    assert 0 <= idx_attn < len(trials)
    assert 0 <= idx_ctrl < len(trials)

    trA = trials[idx_attn]
    trC = trials[idx_ctrl]
    trP = trials[idx_ctrl_pre] if (idx_ctrl_pre is not None and 0 <= idx_ctrl_pre < len(trials)) else None

    nrows = 3 if trP is not None else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 8 if nrows==3 else 6), sharex=True, sharey=True)
    axes_list = list(axes) if isinstance(axes, np.ndarray) else [axes]

    xmins, xmaxs = [], []

    xmin, xmax = _raster_panel(
        axes_list[0], spike_times_ms, spike_ids, trA, N_E, N_I, Q,
        check_win_start_ms, check_win_end_ms, raster_stride,
        pad_pre_ms=pad_pre_ms, pad_post_ms=pad_post_ms,
    )
    xmins.append(xmin); xmaxs.append(xmax)
    axes_list[0].set_title(f"Trial {trA.trial} (ATTN), cluster {trA.cluster}  (stride={raster_stride})")

    xmin, xmax = _raster_panel(
        axes_list[1], spike_times_ms, spike_ids, trC, N_E, N_I, Q,
        check_win_start_ms, check_win_end_ms, raster_stride,
        pad_pre_ms=pad_pre_ms, pad_post_ms=pad_post_ms,
    )
    xmins.append(xmin); xmaxs.append(xmax)
    axes_list[1].set_title(f"Trial {trC.trial} (CTRL), cluster {trC.cluster}  (stride={raster_stride})")

    if trP is not None:
        xmin, xmax = _raster_panel(
            axes_list[2], spike_times_ms, spike_ids, trP, N_E, N_I, Q,
            check_win_start_ms, check_win_end_ms, raster_stride,
            pad_pre_ms=pad_pre_ms, pad_post_ms=pad_post_ms,
        )
        xmins.append(xmin); xmaxs.append(xmax)
        axes_list[2].set_title(f"Trial {trP.trial} (CTRL_PRE), cluster {trP.cluster}  (stride={raster_stride})")

    # Harmonize x-lims across rows
    xmin_all = min(xmins)
    xmax_all = max(xmaxs)
    for ax in axes_list:
        ax.set_xlim(xmin_all, xmax_all)

    # Re-enable x tick labels on first and second subplot even with sharex=True
    axes_list[0].tick_params(axis='x', labelbottom=True)
    if len(axes_list) > 1:
        axes_list[1].tick_params(axis='x', labelbottom=True)

    axes_list[-1].set_xlabel("Time (ms, rel. trial start)")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    return fig


# =============================================================
# All-trials debug plot (UPDATED to use trial_start)
# =============================================================

def plot_all_trials_debug(
    t_bins: np.ndarray,
    dom: np.ndarray,
    trials: List[TrialSpec],
    pred: np.ndarray,
    succ: np.ndarray,
    Q: int,
    check_win_start_ms: int = -50,
    check_win_end_ms: int = 0,
    bin_ms: int = 5,
    pad_ms: int = 200,
    outpath: Optional[str] = None,
):
    """Compact debug overview: one row per trial showing the dominant-cluster time course
    around each trial. Overlays stim window (red), attention span (light), and check window (dashed lines).
    Row labels show condition, chosen cluster, predicted cluster, and success."""
    n_trials = len(trials)
    if n_trials == 0:
        return None

    # Determine per-trial attention durations (for sizing); ignore CTRL (no attention)
    attn_durs = []
    for tr in trials:
        if tr.t_attn_on is not None and tr.t_attn_off is not None:
            attn_durs.append(float(tr.t_attn_off - tr.t_attn_on))
    max_attn = float(max(attn_durs)) if attn_durs else 0.0
    L_ms = int(pad_ms + max_attn + pad_ms)
    nbins_row = int(np.ceil(L_ms / bin_ms))

    # Build matrix of dominant clusters per trial window
    M = -1 * np.ones((n_trials, nbins_row), dtype=int)
    for i, tr in enumerate(trials):
        t0 = tr.trial_start - pad_ms
        idx = np.where((t_bins >= t0) & (t_bins <= t0 + L_ms))[0]
        if idx.size == 0:
            continue
        j = np.clip(np.round((t_bins[idx] - t0) / bin_ms).astype(int), 0, nbins_row - 1)
        M[i, j] = dom[idx]

    fig_h = min(0.35 * n_trials + 2.0, 14.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    ax.imshow(M, aspect='auto', interpolation='nearest', origin='lower',
              extent=(0, nbins_row * bin_ms, -0.5, n_trials - 0.5))

    # Overlays per trial
    for i, tr in enumerate(trials):
        t0 = tr.trial_start - pad_ms
        # attention span
        if tr.t_attn_on is not None and tr.t_attn_off is not None:
            attn_dur_i = float(tr.t_attn_off - tr.t_attn_on)
            ax.add_patch(Rectangle((tr.t_attn_on - 0 + pad_ms, i - 0.5), attn_dur_i, 1,
                                   linewidth=0, facecolor='grey', alpha=0.08))
        # stim span
        x0 = tr.t_stim_on + pad_ms
        x1 = tr.t_stim_off + pad_ms
        if x1 > x0:
            ax.add_patch(Rectangle((x0, i - 0.5), x1 - x0, 1,
                                   linewidth=0, facecolor='red', alpha=0.2))
        # check window lines
        w0 = check_win_start_ms + pad_ms
        w1 = check_win_end_ms + pad_ms
        if w1 < w0:
            w0, w1 = w1, w0
        ax.vlines([w0, w1], i - 0.5, i + 0.5, colors='black', linestyles='--', linewidth=1.2)

    # Y labels with condition + result
    labels = []
    for i, tr in enumerate(trials):
        ok = bool(succ[i]) if i < len(succ) else False
        pd = int(pred[i]) if i < len(pred) else -1
        mark = '✓' if ok else '✗'
        abbr = {"ATTN": "A", "CTRL": "C", "CTRL_PRE": "CP"}.get(tr.condition, tr.condition[:1])
        labels.append(f"{i:02d} {abbr}  cl{tr.cluster}→{pd if pd>=0 else '-'}  {mark}")
    ax.set_yticks(np.arange(n_trials))
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlim(0, nbins_row * bin_ms)
    ax.set_xlabel("Time around trial (ms, rel. trial start)")
    ax.set_ylabel("Trials")
    ax.set_title("All trials debug: dominant cluster per bin (colors) with overlays")
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150)
    return fig

def plot_moving_retention(
    t_axis: np.ndarray,
    curves: Dict[str, np.ndarray],
    *,
    win_ms: int,
    counts: Optional[Dict[str, np.ndarray]] = None,
    outpath: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot success-rate curves
    order = ["ATTN", "CTRL", "CTRL_PRE"]
    labels = {"ATTN": "ATTN", "CTRL": "CTRL", "CTRL_PRE": "CTRL_PRE"}
    for c in order:
        if c in curves:
            ax.plot(t_axis, curves[c], label=labels[c])

    ax.set_xlabel("Time relative to trial start (ms)")
    ax.set_ylabel("Retention success (moving avg)")
    ax.set_title(f"Retention vs time (window = {win_ms} ms)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    # Optional counts as dashed lines on a secondary y-axis (normalized)
    if counts is not None and len(counts) > 0:
        ax2 = ax.twinx()
        # Normalize by max across all to map to [0,1]
        all_counts = np.concatenate([v for v in counts.values()]) if counts else np.array([1])
        denom = max(1, int(all_counts.max()))
        for c in order:
            if c in counts and denom > 0:
                ax2.plot(t_axis, counts[c] / denom, linestyle='--', alpha=0.4)
        ax2.set_ylabel("Trials used (norm.)")
        ax2.set_ylim(0, 1)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    return fig


# =============================================================
# Main (wired to updated API)
# =============================================================
if __name__ == "__main__":
    # --- SLURM + env ---
    CPUcount = int(os.environ.get("SLURM_CPUS_PER_TASK", "10"))
    JobID = os.environ.get("SLURM_JOB_ID", "0")
    ArrayID = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

    gitHash = os.popen("git rev-parse HEAD").read().strip()

    randseed = int(os.environ.get("randseed", ArrayID)) + 1
    np.random.seed(randseed)

    # --- base model params ---
    jep = float(os.environ.get("jep", 6.8))
    jip_ratio = float(os.environ.get("jip_ratio", "0.75"))
    I_th_E = float(os.environ.get("I_th_E", 2.0))
    I_th_I = float(os.environ.get("I_th_I", 1.4))
    N_E = int(os.environ.get("N_E", 4000))
    N_I = int(os.environ.get("N_I", 1000))
    tau_stc = float(os.environ.get("tau_stc", 50.0))
    q_stc = float(os.environ.get("q_stc", 0.3))

    # Trials protocol params (env-overridable)
    N_TRIALS = int(os.environ.get("N_trials", 50))
    ITI_MS = int(os.environ.get("ITI_ms", 1000))
    ITI_MS_MIN = int(os.environ.get("ITI_ms_min", ITI_MS))
    ITI_MS_MAX = int(os.environ.get("ITI_ms_max", ITI_MS))
    P_CTRL = float(os.environ.get("p_ctrl", 0.5))
    P_CTRL_PRE = float(os.environ.get("p_ctrl_pre", 0.0))
    STIM_DUR_MS = int(os.environ.get("stim_dur_ms", 350))
    ATTN_DUR_MS = int(os.environ.get("attn_dur_ms", 800))
    ATTN_FACTOR = float(os.environ.get("attn_factor", 0.1))  # q := q * ATTN_FACTOR during attention
    I_STIM_E = float(os.environ.get("I_stim_E", 150.0))       # pA
    # Optional: stimulus as a multiple of rheobase (e.g., 0.2 means +0.2 * I_rheo)
    _fac_env = os.environ.get("I_stim_E_fac", os.environ.get("I_stim_E_rheo", ""))
    I_STIM_E_FAC = float(_fac_env) if str(_fac_env).strip() != "" else None  # None -> use absolute pA
    # Retention check window (relative to trial start). Defaults to last 50 ms before t=0
    CHECK_WIN_START_MS = int(os.environ.get("check_win_start_ms", 700))
    CHECK_WIN_END_MS = int(os.environ.get("check_win_end_ms", 800))
    # Binning and raster sparsification
    BIN_MS = int(os.environ.get("bin_ms", 10))
    RASTER_STRIDE = int(os.environ.get("raster_stride", 10))
    WARMUP_MS = int(os.environ.get("warmup_ms", 5000))
    PRE_FIRST_MS = int(os.environ.get("pre_first_ms", 0))
    PRE_ATTN_MS = int(os.environ.get("pre_attn_ms", 500))
    POST_ATTN_EXTRA_MS = int(os.environ.get("post_attn_extra_ms", 1200))

    # Plot window controls (ms relative to trial start)
    PLOT_WINDOW_PRESTIM = int(os.environ.get("plot_window_prestim", -200))
    PLOT_WINDOW_POSTSTART = int(os.environ.get("plot_window_poststart", os.environ.get("plot_window_past_att", 1000)))
    PAD_PRE_MS = abs(PLOT_WINDOW_PRESTIM)
    PAD_POST_MS = int(PLOT_WINDOW_POSTSTART)

    # output
    output_path = os.environ.get("output_path", "../output/")
    if output_path.endswith(os.path.sep):
        output_file = f"job_{JobID}_array_{ArrayID}.pkl"
        output_path = os.path.join(output_path, output_file)
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    # compute inhibitory weight
    jip = 1.0 + (jep - 1.0) * jip_ratio
    overrides = {
        'simulation': {
            'n_jobs': CPUcount,
            'dt': 0.1,
            'warmup': float(WARMUP_MS),
            'duration': float(WARMUP_MS),
            'randseed': randseed,
        },
        'network': {
            'population': {'excitatory': N_E, 'inhibitory': N_I},
            'clusters': {'count': 10, 'jep': float(jep), 'jip_ratio': float(jip_ratio)},
        },
        'neuron': {
            'model': 'gif_psc_exp',
            'I_th': {'excitatory': float(I_th_E), 'inhibitory': float(I_th_I)},
            'tau_stc': float(tau_stc),
            'q_stc': float(q_stc),
        },
        'stimulation': {'background': 'DC'},
    }

    # ------------------- RUN MIXED (ATTN + CTRL + CTRL_PRE) -------------------
    EI = ClusterModelNEST.ClusteredNetworkNEST(default, overrides)
    EI.setup_network()
    cfg = EI.get_parameter()
    N_E = int(cfg['network']['population']['excitatory'])
    N_I = int(cfg['network']['population']['inhibitory'])
    Q = cfg['network']['clusters']['count']

    rng = np.random.default_rng(randseed)
    q_sched, trials = run_trials_protocol_mixed(
        EI,
        warmup_ms=WARMUP_MS,
        n_trials=N_TRIALS,
        iti_ms=ITI_MS,
        iti_range=(ITI_MS_MIN, ITI_MS_MAX),
        q_baseline=q_stc,
        attn_factor=ATTN_FACTOR,
        stim_Ie_pA=I_STIM_E,
        I_th_E=I_th_E,
        stim_fac_rheo=I_STIM_E_FAC,
        stim_dur_ms=STIM_DUR_MS,
        attn_dur_ms=ATTN_DUR_MS,
        pre_first_ms=PRE_FIRST_MS,
        p_ctrl=P_CTRL,
        p_ctrl_pre=P_CTRL_PRE,
        pre_attn_ms=PRE_ATTN_MS,
        post_attn_extra_ms=POST_ATTN_EXTRA_MS,
        rng=rng,
    )

    spk_times, spk_ids = EI.get_recordings()
    t_bins, rates, dom = bin_cluster_rates(
        np.array(spk_times), np.array(spk_ids), N_E=N_E, Q=Q, bin_ms=BIN_MS
    )
    pred, succ = dominant_and_success_over_window(
        t_bins, rates, trials,
        win_start_ms=CHECK_WIN_START_MS,
        win_end_ms=CHECK_WIN_END_MS,
    )

    # Per-condition summary
    cond_arr = np.array([t.condition for t in trials])
    succ_arr = np.array(succ, dtype=bool)
    mask_attn = cond_arr == "ATTN"
    mask_ctrl = cond_arr == "CTRL"
    mask_ctrl_pre = cond_arr == "CTRL_PRE"

    n_attn = int(mask_attn.sum())
    n_ctrl = int(mask_ctrl.sum())
    n_ctrl_pre = int(mask_ctrl_pre.sum())

    s_attn = int((succ_arr & mask_attn).sum())
    s_ctrl = int((succ_arr & mask_ctrl).sum())
    s_ctrl_pre = int((succ_arr & mask_ctrl_pre).sum())

    rate_attn = (s_attn / n_attn) if n_attn > 0 else float("nan")
    rate_ctrl = (s_ctrl / n_ctrl) if n_ctrl > 0 else float("nan")
    rate_ctrl_pre = (s_ctrl_pre / n_ctrl_pre) if n_ctrl_pre > 0 else float("nan")

    print(f"""Summary:
  ATTN     {s_attn}/{n_attn} = {rate_attn:.3f}
  CTRL     {s_ctrl}/{n_ctrl} = {rate_ctrl:.3f}
  CTRL_PRE {s_ctrl_pre}/{n_ctrl_pre} = {rate_ctrl_pre:.3f}""")

    results: Dict[str, Any] = {
        "recordings": (spk_times, spk_ids),
        "params": EI.get_parameter(),
        "q_schedule": q_sched,
        "trials": [t.__dict__ for t in trials],
        "binning": {"t": t_bins, "rates": rates, "dom": dom},
        "pred_cluster": pred,
        "success": succ,
        "summary": {
            "ATTN": {"n": n_attn, "n_success": s_attn, "rate": rate_attn},
            "CTRL": {"n": n_ctrl, "n_success": s_ctrl, "rate": rate_ctrl},
            "CTRL_PRE": {"n": n_ctrl_pre, "n_success": s_ctrl_pre, "rate": rate_ctrl_pre},
        },
        "check_window": (CHECK_WIN_START_MS, CHECK_WIN_END_MS),
    }

    # ------------------- SAVE -------------------
    meta = {"gitHash": gitHash, "JobID": JobID, "ArrayID": ArrayID}
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
        pickle.dump(meta, f)

    # Example (optional): create a quick raster for one ATTN, one CTRL, and CTRL_PRE

    try:
        plot_failed = False
        # choose one ATTN, CTRL, CTRL_PRE (prefer successful, else first occurrence)
        cond_list = [t["condition"] for t in results["trials"]]
        succ_list = results["success"]
        if plot_failed:
            succ_list = np.invert(succ_list)
        try:
            iA = next(i for i, (c, s) in enumerate(zip(cond_list, succ_list)) if c == "ATTN" and s)
        except StopIteration:
            iA = next((i for i, c in enumerate(cond_list) if c == "ATTN"), 0)
        try:
            iC = next(i for i, (c, s) in enumerate(zip(cond_list, succ_list)) if c == "CTRL" and s)
        except StopIteration:
            iC = next((i for i, c in enumerate(cond_list) if c == "CTRL"), 0)
        try:
            iP = next(i for i, (c, s) in enumerate(zip(cond_list, succ_list)) if c == "CTRL_PRE" and s)
        except StopIteration:
            iP = next((i for i, c in enumerate(cond_list) if c == "CTRL_PRE"), None)

        figpath = os.path.splitext(output_path)[0] + "_rasters3.png"
        plot_three_rasters(
            np.array(spk_times),
            np.array(spk_ids),
            N_E,
            N_I,
            [TrialSpec(**d) for d in results["trials"]],
            iA,
            iC,
            iP,
            Q=Q,
            check_win_start_ms=CHECK_WIN_START_MS,
            check_win_end_ms=CHECK_WIN_END_MS,
            raster_stride=RASTER_STRIDE,
            pad_pre_ms=PAD_PRE_MS,
            pad_post_ms=PAD_POST_MS,
            outpath=figpath,
        )
    except Exception as e:
        print("Raster plotting failed:", e)

    # Debug: all trials overview
    try:
        figpath_all = os.path.splitext(output_path)[0] + "_alltrials_debug.png"
        plot_all_trials_debug(
            t_bins,
            dom,
            [TrialSpec(**d) for d in results["trials"]],
            pred,
            succ,
            Q=Q,
            check_win_start_ms=CHECK_WIN_START_MS,
            check_win_end_ms=CHECK_WIN_END_MS,
            bin_ms=BIN_MS,
            pad_ms=200,
            outpath=figpath_all,
        )
    except Exception as e:
        print("All-trials debug plotting failed:", e)

    # ------------------- Moving-window retention figure (NEW) -------------------
    try:
        MOV_STEP_MS=5
        MOV_WIN_MS=100
        t_axis, curves, counts = moving_retention_success(
            t_bins,
            rates,
            [TrialSpec(**d) for d in results["trials"]],
            win_ms=MOV_WIN_MS,
            t_min_ms=PLOT_WINDOW_PRESTIM,
            t_max_ms=PLOT_WINDOW_POSTSTART,
            step_ms=MOV_STEP_MS,
        )
        figpath_mov = os.path.splitext(output_path)[0] + f"_retention_moving_{MOV_WIN_MS}ms.png"
        plot_moving_retention(t_axis, curves, win_ms=MOV_WIN_MS, outpath=figpath_mov)
    except Exception as e:
        print("Moving-window retention plotting failed:", e)
