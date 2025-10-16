#!/usr/bin/env python3
import os, glob, pickle, itertools, argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- KM utilities ----------------
def _runs_to_episode_lengths_per_bin(labels_1d):
    n = len(labels_1d)
    change = np.flatnonzero(np.diff(labels_1d) != 0) + 1
    starts = np.r_[0, change]
    ends   = np.r_[change - 1, n - 1]
    ep_len_bins = np.empty(n, dtype=int)
    event_obs   = np.empty(n, dtype=np.int8)
    for s, e in zip(starts, ends):
        L = e - s + 1
        cens = (s == 0) or (e == n - 1)   # touches start or end → censored
        ep_len_bins[s:e+1] = L
        event_obs[s:e+1]   = 0 if cens else 1
    return ep_len_bins, event_obs

def _km_median(durs_ms, events):
    if len(durs_ms) == 0: return np.nan
    order = np.argsort(durs_ms)
    t = durs_ms[order].astype(float); e = events[order].astype(int)
    uniq = np.unique(t)
    at_risk = len(t); S = 1.0
    for ut in uniq:
        d = np.sum((t == ut) & (e == 1))
        c = np.sum((t == ut) & (e == 0))
        if at_risk > 0 and d > 0:
            S *= (1.0 - d / at_risk)
            if S <= 0.5:
                return float(ut)
        at_risk -= (d + c)
    return np.nan

def _km_mean_rmst(durs_ms, events, tau=None):
    if len(durs_ms) == 0: return np.nan
    order = np.argsort(durs_ms)
    t = durs_ms[order].astype(float); e = events[order].astype(int)
    ev_times = np.unique(t[e == 1])
    if ev_times.size == 0:
        return np.nan
    Tmax = ev_times.max()
    if tau is None: tau = Tmax
    tau = min(tau, Tmax)
    at_risk = len(t); S = 1.0
    surv_times = [0.0]; surv_vals = [1.0]
    for ut in np.unique(t):
        d = np.sum((t == ut) & (e == 1))
        c = np.sum((t == ut) & (e == 0))
        if d > 0 and at_risk > 0:
            S *= (1.0 - d / at_risk)
            surv_times.append(ut); surv_vals.append(S)
        at_risk -= (d + c)
    times = np.array(surv_times, float)
    vals  = np.array(surv_vals,  float)
    seg_times = np.r_[times, tau]
    seg_vals  = np.r_[vals,  vals[-1]]
    area = 0.0
    for i in range(len(times)):
        t0 = seg_times[i]
        t1 = min(seg_times[i+1], tau)
        if t1 <= t0: break
        area += seg_vals[i] * (t1 - t0)
        if seg_times[i+1] >= tau: break
    return float(area)

# ------------- loading -------------
def load_label_time_from_pkls(data_dir):
    """
    Returns:
      t_ms_ref: [nbins]
      labels_list: list of [nbins] arrays (dom per seed)
      adapt_list:  list of [nbins] arrays (adaptation value per bin)
      kept_files:  list of filenames loaded
      recordings:  list of (times, ids) arrays per file  (times: float64 [M], ids: int64 [M])
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .pkl files in {data_dir}")

    t_ref = None
    labels_list, adapt_list, kept_files, recordings = [], [], [], []

    for f in files:
        try:
            with open(f, "rb") as fh:
                rec   = pickle.load(fh)  # 2xN np.array: [times; ids]
                _param = pickle.load(fh)
                trio   = pickle.load(fh)  # {"adapt": adapt_q, "time": t_ms, "label": dom}
                _meta  = pickle.load(fh)

            # --- parse t_ms / dom / adapt ---
            t_ms  = np.asarray(trio["time"], dtype=float)
            dom   = np.asarray(trio["label"])
            adapt = np.asarray(trio.get("adapt", np.full_like(dom, np.nan, dtype=float)), dtype=float)

            if t_ref is None:
                t_ref = t_ms
            else:
                if len(t_ms) != len(t_ref) or np.max(np.abs(t_ms - t_ref)) > 1e-6:
                    print(f"Skip (mismatched t_ms): {os.path.basename(f)}")
                    continue

            labels_list.append(dom)
            adapt_list.append(adapt)
            kept_files.append(f)

            # --- parse recordings (2×N array) ---
            times_ids = (np.asarray(rec) if not isinstance(rec, np.ndarray) else rec)
            if isinstance(times_ids, np.ndarray) and times_ids.ndim == 2 and times_ids.shape[0] >= 2:
                times = np.asarray(times_ids[0], dtype=float)
                ids   = np.asarray(times_ids[1], dtype=int)
                recordings.append((times, ids))
            else:
                # fallback empty if format unexpected
                recordings.append((np.array([], dtype=float), np.array([], dtype=int)))

        except Exception as e:
            print(f"Skip (read error): {os.path.basename(f)} -> {e}")

    if not labels_list:
        raise RuntimeError("No usable .pkl files after consistency checks.")
    print(f"Loaded {len(labels_list)} simulations.")
    return t_ref, labels_list, adapt_list, kept_files, recordings


# ------------- main per-bin KM over window -------------
def km_dwell_over_window_from_labels(label_sims, t_ms, bin_ms, lo_ms, hi_ms, ignore_label=None):
    """
    Returns:
      t_valid, km_med_ms, km_mean_ms, mean_events_ms, median_events_ms,
      ep_len_bins_all [S,nbins], event_obs_all [S,nbins], idx_v
    """
    S, nbins = label_sims.shape
    valid = (t_ms >= lo_ms) & (t_ms <= hi_ms)
    idx_v = np.where(valid)[0]; t_valid = t_ms[valid]
    ep_len_bins_all = np.empty_like(label_sims, dtype=int)
    event_obs_all   = np.empty_like(label_sims, dtype=np.int8)
    for s in range(S):
        ep_len_bins_all[s], event_obs_all[s] = _runs_to_episode_lengths_per_bin(label_sims[s])

    km_med   = np.full(idx_v.size, np.nan, float)
    km_mean  = np.full(idx_v.size, np.nan, float)
    mean_e   = np.full(idx_v.size, np.nan, float)
    median_e = np.full(idx_v.size, np.nan, float)

    for k, i in enumerate(idx_v):
        durs = []; evts = []
        for s in range(S):
            lab = label_sims[s, i]
            if (ignore_label is not None) and (lab == ignore_label): continue
            durs.append(ep_len_bins_all[s, i] * bin_ms)
            evts.append(event_obs_all[s, i])
        if not durs: continue
        durs = np.asarray(durs, float); evts = np.asarray(evts, np.int8)
        km_med[k]  = _km_median(durs, evts)
        km_mean[k] = _km_mean_rmst(durs, evts)
        if np.any(evts == 1):
            eo = durs[evts == 1]
            mean_e[k]   = float(np.mean(eo))
            median_e[k] = float(np.median(eo))
    return t_valid, km_med, km_mean, mean_e, median_e, ep_len_bins_all, event_obs_all, idx_v

# --------- bootstrap CIs per bin (over seeds) ----------
def _bootstrap_bin_worker(args):
    i, durs_ms_all, events_all, B, q_lo, q_hi, seed = args
    rng = np.random.default_rng(None if seed is None else seed + i)
    S = durs_ms_all.shape[0]
    durs = durs_ms_all[:, i]
    evts = events_all[:, i]
    boots_med  = np.empty(B, float)
    boots_mean = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, S, size=S)
        boots_med[b]  = _km_median(durs[idx], evts[idx])
        boots_mean[b] = _km_mean_rmst(durs[idx], evts[idx])
    med_lo, med_hi   = np.nanpercentile(boots_med,  q=[q_lo, q_hi])
    mean_lo, mean_hi = np.nanpercentile(boots_mean, q=[q_lo, q_hi])
    return i, med_lo, med_hi, mean_lo, mean_hi

def bootstrap_km_ci_per_bin_parallel(durs_ms_all, events_all, idx_bins, B=1000, alpha=0.05, rng_seed=None, n_jobs=1):
    if B <= 0:
        return None, None, None, None
    q_lo, q_hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    med_lo = np.full(len(idx_bins), np.nan)
    med_hi = np.full(len(idx_bins), np.nan)
    mean_lo = np.full(len(idx_bins), np.nan)
    mean_hi = np.full(len(idx_bins), np.nan)
    if n_jobs == 1:
        for j, i in enumerate(idx_bins):
            _, lo_m, hi_m, lo_mean, hi_mean = _bootstrap_bin_worker((i, durs_ms_all, events_all, B, q_lo, q_hi, rng_seed))
            med_lo[j], med_hi[j], mean_lo[j], mean_hi[j] = lo_m, hi_m, lo_mean, hi_mean
        return med_lo, med_hi, mean_lo, mean_hi
    tasks = [(i, durs_ms_all, events_all, B, q_lo, q_hi, rng_seed) for i in idx_bins]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_bootstrap_bin_worker, t) for t in tasks]
        for fut in as_completed(futures):
            i, lo_m, hi_m, lo_mean, hi_mean = fut.result()
            j = np.where(idx_bins == i)[0][0]
            med_lo[j], med_hi[j], mean_lo[j], mean_hi[j] = lo_m, hi_m, lo_mean, hi_mean
    return med_lo, med_hi, mean_lo, mean_hi

# --------- tests at specific timepoints (console only) ----------
def nearest_bin_indices(t_ms, times_ms):
    return np.array([int(np.argmin(np.abs(t_ms - t))) for t in times_ms], int)

def _bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p); ranked = p[order]
    q = np.empty(n, float)
    min_val = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        min_val = min(min_val, ranked[i] * n / rank)
        q[i] = min_val
    qvals = np.empty(n, float); qvals[order] = np.clip(q, 0, 1)
    return qvals

def _bootstrap_diff_worker(args):
    iA, iB, durs_ms_all, events_all, B, alpha, seed = args
    rng = np.random.default_rng(None if seed is None else seed + (iA * 1315423911 ^ iB))
    dA, eA = durs_ms_all[:, iA], events_all[:, iA]
    dB, eB = durs_ms_all[:, iB], events_all[:, iB]
    S = len(dA)
    boots = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, S, size=S)
        boots[b] = _km_median(dA[idx], eA[idx]) - _km_median(dB[idx], eB[idx])
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return np.nan, np.nan, np.nan
    p_two = 2 * min(np.mean(boots >= 0), np.mean(boots <= 0))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return p_two, lo, hi

def report_timepoint_tests_parallel(times_ms, t_ms, durs_ms_all, events_all,
                                    alpha=0.05, rng=None, B=5000, n_jobs=1):
    if not times_ms:
        return
    idxs = nearest_bin_indices(t_ms, times_ms)
    print("\n=== Timepoint KM medians (with bootstrap CI) ===")
    for t, i in zip(times_ms, idxs):
        durs = durs_ms_all[:, i]; evts = events_all[:, i]
        hat = _km_median(durs, evts)
        rng_local = np.random.default_rng(rng)
        S = len(durs)
        boots = np.empty(B, float)
        for b in range(B):
            idx = rng_local.integers(0, S, size=S)
            boots[b] = _km_median(durs[idx], evts[idx])
        lo, hi = np.nanpercentile(boots, [100*alpha/2, 100*(1-alpha/2)])
        print(f"t={t:.1f} ms (bin ~{t_ms[i]:.1f}): KM median={hat:.1f} ms  [{lo:.1f}, {hi:.1f}]")
    pairs = list(itertools.combinations(zip(times_ms, idxs), 2))
    if not pairs:
        return
    tasks = [(iA, iB, durs_ms_all, events_all, B, alpha, rng)
             for (_, iA), (_, iB) in pairs]
    if n_jobs == 1:
        results = [_bootstrap_diff_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_bootstrap_diff_worker, tasks))
    pvals = [p for (p, _, _) in results]
    qvals = _bh_fdr(np.array(pvals)) if pvals else np.array([])
    print("\n=== Pairwise differences (KM median): two-sided p (BH-FDR q) with CI ===")
    for ((tA, iA), (tB, iB)), (p, lo, hi), q in zip(pairs, results, qvals):
        print(f"Δ(t={tA:.0f}−{tB:.0f})  p={p:.3g},  q={q:.3g},  CI[{lo:.1f}, {hi:.1f}] ms")

# ------------------------- analysis driver -------------------------
def analyze_dir(
    data_dir, pre_ms, post_ms, bin_ms, ignore_label=None,
    make_plots=True, bootstrap_B=0, test_bootstrap_B=2000, alpha=0.05,
    test_times=None, rng=None, n_jobs=1,
    with_raster=False, raster_mod=5, N_E=1200, raster_seed_index=0, no_ci=False
):
    t_ms, labels_list, adapt_list, files, recordings = load_label_time_from_pkls(data_dir)
    S = len(labels_list)
    label_sims = np.vstack(labels_list)  # [S, nbins]
    adapt_arr  = np.vstack(adapt_list)
    adapt_ref  = np.nanmean(adapt_arr, axis=0)

    # central window = [pre_ms, t_end - post_ms]
    t_end = t_ms[-1]
    lo_ms, hi_ms = float(pre_ms), float(t_end - post_ms)
    if lo_ms >= hi_ms:
        raise ValueError("pre_ms + post_ms exceeds total simulation time.")

    # Compute KM over window
    (t_valid, km_med_ms, km_mean_ms, mean_ms, median_ms,
     ep_len_bins_all, event_obs_all, idx_v) = km_dwell_over_window_from_labels(
        label_sims, t_ms, bin_ms, lo_ms, hi_ms, ignore_label=ignore_label
    )
    durs_ms_all = ep_len_bins_all * float(bin_ms)

    # Bootstrap CIs per bin (median & mean)
    med_lo = med_hi = mean_lo = mean_hi = None
    if (bootstrap_B and bootstrap_B > 0) and (not no_ci):
        med_lo, med_hi, mean_lo, mean_hi = bootstrap_km_ci_per_bin_parallel(
            durs_ms_all, event_obs_all, idx_v, B=bootstrap_B, alpha=alpha,
            rng_seed=rng, n_jobs=n_jobs
        )

    # ---- time shift: set 0 at pre_ms ----
    # Full timeline shifted (for adapt & raster)
    t_ms0 = t_ms - lo_ms
    # Valid (window) timeline shifted (for dwell)
    t_valid0 = t_valid - lo_ms
    x_max = hi_ms - lo_ms

    # ---- Raster data (optional) ----
    raster_t, raster_id = None, None
    if with_raster:
        r_idx = int(np.clip(raster_seed_index, 0, len(recordings) - 1))
        stimes, sids = recordings[r_idx]
        if stimes.size:
            # crop to central window, then shift by pre_ms (lo_ms)
            mask = (stimes >= lo_ms) & (stimes <= hi_ms)
            st = stimes[mask] - lo_ms
            si = np.asarray(sids[mask], dtype=int)

            # every k-th neuron
            everyk = (si % raster_mod == 0)
            st = st[everyk];
            si = si[everyk]
            raster_t, raster_id = st, si
        else:
            raster_t = np.array([], float);
            raster_id = np.array([], int)

    # ---- Plots (stacked; shared x; start at 0) ----
    if make_plots:
        # Decide layout: 2 or 3 rows
        if with_raster:
            # height: top adapt small, raster middle, dwell big
            fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
                3, 1, figsize=(10, 5.2), sharex=True,
                gridspec_kw={"height_ratios": [1, 4, 7]}
            )
            # top: adaptation (shifted)
            ax_top.plot(t_ms0, adapt_ref, lw=1.2)
            ax_top.set_ylabel("adapt"); ax_top.grid(alpha=0.2)

            # mid: raster
            if raster_t is not None and raster_t.size:
                exc_mask = (raster_id < N_E)
                inh_mask = ~exc_mask
                ax_mid.plot(raster_t[exc_mask], raster_id[exc_mask], '|', ms=0.5, color='black', rasterized=True)
                ax_mid.plot(raster_t[inh_mask], raster_id[inh_mask], '|', ms=0.5, color='darkred', rasterized=True)

            ax_mid.set_yticks([]);
            ax_mid.set_ylabel('')

            # Place “exc.” / “inh.” tags at the actual midpoints
            try:
                if raster_id.size:
                    y_exc = np.median(raster_id[raster_id < N_E]) if np.any(raster_id < N_E) else N_E * 0.5
                    y_inh = np.median(raster_id[raster_id >= N_E]) if np.any(raster_id >= N_E) else N_E + 100
                    ax_mid.text(-350, y_exc, 'exc.', va='center', ha='left', color='black', fontsize=12, rotation=90)
                    ax_mid.text(-350, y_inh, 'inh.', va='center', ha='left', color='darkred', fontsize=12, rotation=90)
            except Exception:
                pass

            # bot: dwell
            if not no_ci and (med_lo is not None):
                ax_bot.fill_between(t_valid0, med_lo, med_hi, color='C0', alpha=0.18, linewidth=0, label=f"KM median {(1-alpha):.0%} CI")
            if not no_ci and (mean_lo is not None):
                ax_bot.fill_between(t_valid0, mean_lo, mean_hi, color='C1', alpha=0.15, linewidth=0, label=f"KM mean {(1-alpha):.0%} CI")
            ax_bot.plot(t_valid0, km_med_ms,  label="KM median", lw=1.8)
            ax_bot.plot(t_valid0, km_mean_ms, label="KM mean (RMST)", lw=1.5, alpha=0.9)
            ax_bot.set_xlim([0, x_max]); ax_bot.set_xlabel("Time (ms)"); ax_bot.set_ylabel("Dwell (ms)")
            ax_bot.set_title(f"Time-resolved dwell — S={S}")
            ax_bot.legend(frameon=False, ncol=2); ax_bot.grid(alpha=0.2)
            plt.tight_layout(); fig.savefig("Time_res_dwell_linear.png", dpi=150)

            # Log version
            fig2, (ax2_top, ax2_mid, ax2_bot) = plt.subplots(
                3, 1, figsize=(10, 5.2), sharex=True,
                gridspec_kw={"height_ratios": [1, 4, 7]}
            )
            ax2_top.plot(t_ms0, adapt_ref, lw=1.2); ax2_top.set_ylabel("adapt"); ax2_top.grid(alpha=0.2)
            if raster_t is not None and raster_t.size:
                exc_mask = (raster_id < N_E)
                inh_mask = ~exc_mask
                ax2_mid.plot(raster_t[exc_mask], raster_id[exc_mask], '|', ms=0.5, color='black', rasterized=True)
                ax2_mid.plot(raster_t[inh_mask], raster_id[inh_mask], '|', ms=0.5, color='darkred', rasterized=True)
            ax2_mid.set_yticks([]); ax2_mid.set_ylabel('')
            def _plot_logsafe(ax, x, y, **kw):
                y = np.asarray(y, float).copy(); y[y <= 0] = np.nan
                ax.plot(x, y, **kw)
            _plot_logsafe(ax2_bot, t_valid0, km_med_ms,  label="KM median", lw=1.8)
            _plot_logsafe(ax2_bot, t_valid0, km_mean_ms, label="KM mean (RMST)", lw=1.5, alpha=0.9)
            if not no_ci and (med_lo is not None):
                ylo, yhi = med_lo.copy(), med_hi.copy()
                m = (ylo > 0) & (yhi > 0)
                ax2_bot.fill_between(t_valid0[m], ylo[m], yhi[m], color='C0', alpha=0.18, linewidth=0, label=f"KM median {(1-alpha):.0%} CI")
            if not no_ci and (mean_lo is not None):
                ylo, yhi = mean_lo.copy(), mean_hi.copy()
                m = (ylo > 0) & (yhi > 0)
                ax2_bot.fill_between(t_valid0[m], ylo[m], yhi[m], color='C1', alpha=0.15, linewidth=0, label=f"KM mean {(1-alpha):.0%} CI")
            ax2_bot.set_yscale('log'); ax2_bot.set_xlim([0, x_max]); ax2_bot.set_xlabel("Time (ms)"); ax2_bot.set_ylabel("Dwell (ms)")
            ymax = np.nanmax([np.nanmax(km_med_ms), np.nanmax(km_mean_ms), np.nanmax(mean_ms), np.nanmax(median_ms)])
            if np.isfinite(ymax): ax2_bot.set_ylim(1.0, max(10.0, ymax * 1.2))
            ax2_bot.set_title(f"Time-resolved dwell (log scale) — S={S}")
            ax2_bot.legend(frameon=False, ncol=2); ax2_bot.grid(alpha=0.2)
            plt.tight_layout(); fig2.savefig("Time_res_dwell_log.png", dpi=150)

        else:
            # Two rows (no raster)
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True,
                                                 gridspec_kw={"height_ratios": [1, 9]})
            ax_top.plot(t_ms0, adapt_ref, lw=1.2)
            ax_top.set_ylabel("adapt"); ax_top.grid(alpha=0.2)
            if not no_ci and (med_lo is not None):
                ax_bot.fill_between(t_valid0, med_lo, med_hi, color='C0', alpha=0.18, linewidth=0, label=f"KM median {(1-alpha):.0%} CI")
            if not no_ci and (mean_lo is not None):
                ax_bot.fill_between(t_valid0, mean_lo, mean_hi, color='C1', alpha=0.15, linewidth=0, label=f"KM mean {(1-alpha):.0%} CI")
            ax_bot.plot(t_valid0, km_med_ms,  label="KM median", lw=1.8)
            ax_bot.plot(t_valid0, km_mean_ms, label="KM mean (RMST)", lw=1.5, alpha=0.9)
            ax_bot.set_xlim([0, x_max]); ax_bot.set_xlabel("Time (ms)"); ax_bot.set_ylabel("Dwell (ms)")
            ax_bot.set_title(f"Time-resolved dwell — S={S}")
            ax_bot.legend(frameon=False, ncol=2); ax_bot.grid(alpha=0.2)
            plt.tight_layout(); fig.savefig("Time_res_dwell_linear.png", dpi=150)

            fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True,
                                                    gridspec_kw={"height_ratios": [1, 9]})
            ax2_top.plot(t_ms0, adapt_ref, lw=1.2)
            ax2_top.set_ylabel("adapt"); ax2_top.grid(alpha=0.2)
            def _plot_logsafe(ax, x, y, **kw):
                y = np.asarray(y, float).copy(); y[y <= 0] = np.nan
                ax.plot(x, y, **kw)
            _plot_logsafe(ax2_bot, t_valid0, km_med_ms,  label="KM median", lw=1.8)
            _plot_logsafe(ax2_bot, t_valid0, km_mean_ms, label="KM mean (RMST)", lw=1.5, alpha=0.9)
            if not no_ci and (med_lo is not None):
                ylo, yhi = med_lo.copy(), med_hi.copy()
                m = (ylo > 0) & (yhi > 0)
                ax2_bot.fill_between(t_valid0[m], ylo[m], yhi[m], color='C0', alpha=0.18, linewidth=0, label=f"KM median {(1-alpha):.0%} CI")
            if not no_ci and (mean_lo is not None):
                ylo, yhi = mean_lo.copy(), mean_hi.copy()
                m = (ylo > 0) & (yhi > 0)
                ax2_bot.fill_between(t_valid0[m], ylo[m], yhi[m], color='C1', alpha=0.15, linewidth=0, label=f"KM mean {(1-alpha):.0%} CI")
            ax2_bot.set_yscale('log'); ax2_bot.set_xlim([0, x_max]); ax2_bot.set_xlabel("Time (ms)"); ax2_bot.set_ylabel("Dwell (ms)")
            ymax = np.nanmax([np.nanmax(km_med_ms), np.nanmax(km_mean_ms), np.nanmax(mean_ms), np.nanmax(median_ms)])
            if np.isfinite(ymax): ax2_bot.set_ylim(1.0, max(10.0, ymax * 1.2))
            ax2_bot.set_title(f"Time-resolved dwell (log scale) — S={S}")
            ax2_bot.legend(frameon=False, ncol=2); ax2_bot.grid(alpha=0.2)
            plt.tight_layout(); fig2.savefig("Time_res_dwell_log.png", dpi=150)

    # Console tests (not plotted)
    if test_times:
        report_timepoint_tests_parallel(
            test_times, t_ms, durs_ms_all, event_obs_all,
            alpha=alpha, rng=rng, B=test_bootstrap_B, n_jobs=n_jobs
        )

    return {
        "t_valid": t_valid0,
        "km_median_ms": km_med_ms,
        "km_mean_ms": km_mean_ms,
        "mean_events_ms": mean_ms,
        "median_events_ms": median_ms,
        "window": (0.0, x_max),
        "seeds": S,
    }

# ----------------------- CLI -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="KM dwell (median & mean) over central window from pkl sims.")
    ap.add_argument("data_dir", type=str, help="Directory with .pkl files")
    ap.add_argument("--pre_ms",  type=float, required=True, help="Padding before central window (ms)")
    ap.add_argument("--post_ms", type=float, required=True, help="Padding after central window (ms)")
    ap.add_argument("--bin_ms",  type=float, required=True, help="Bin size (ms) used in sims")
    ap.add_argument("--ignore_label", type=int, default=None, help="Label to ignore (e.g., -1 for noise)")
    ap.add_argument("--bootstrap_B", type=int, default=0, help="Bootstrap draws per bin for CIs (0 disables)")
    ap.add_argument("--test_bootstrap_B", type=int, default=None, help="Bootstrap draws for timepoint tests (override)")
    ap.add_argument("--alpha", type=float, default=0.05, help="CI level (e.g., 0.05 for 95%)")
    ap.add_argument("--test_times", type=str, default="", help="Comma-separated times in ms for tests (e.g., '50000,80000,110000,140000')")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel workers for bootstraps (processes)")
    ap.add_argument("--no_plots", action="store_true", help="Disable plots")
    ap.add_argument("--no_ci", action="store_true", help="Disable CI ribbons in plots")
    # Raster options
    ap.add_argument("--with_raster", action="store_true", help="Add raster subplot in the middle")
    ap.add_argument("--raster_mod", type=int, default=5, help="Plot every k-th neuron id in raster")
    ap.add_argument("--N_E", type=int, default=1200, help="Number of excitatory neurons (split for raster)")
    ap.add_argument("--raster_seed_index", type=int, default=0, help="Which simulation file to take raster from (index)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for bootstrap")
    args = ap.parse_args()

    times = [float(x) for x in args.test_times.split(",")] if args.test_times.strip() else None
    test_B = args.test_bootstrap_B if args.test_bootstrap_B is not None else max(2000, args.bootstrap_B)

    analyze_dir(
        args.data_dir,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        bin_ms=args.bin_ms,
        ignore_label=args.ignore_label,
        make_plots=not args.no_plots,
        bootstrap_B=args.bootstrap_B,
        test_bootstrap_B=test_B,
        alpha=args.alpha,
        test_times=times,
        rng=args.seed,
        n_jobs=args.n_jobs,
        with_raster=args.with_raster,
        raster_mod=args.raster_mod,
        N_E=args.N_E,
        raster_seed_index=args.raster_seed_index,
        no_ci=args.no_ci
    )
