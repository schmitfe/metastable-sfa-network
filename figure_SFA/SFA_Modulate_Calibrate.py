from __future__ import annotations

import numpy as np
import pickle
import sys
import os
from pathlib import Path

from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import load_config
from src.model import ClusterModelNEST
import nest

default = load_config()

def simulate_with_trajectory(EI_Network, nest, q_stc_initial, *,
                             warmup: int,
                             simtime: int,
                             trajectory,                   # list[(t,q)] OR (times, values)
                             return_as: str = "tuple_of_lists"):
    """
    Simulate with a piecewise-constant trajectory of q_stc.

    Parameters
    ----------
    EI_Network : your network
    nest       : nest module
    q_stc_initial : float
        Initial q_stc used for warmup and as default at t=0 if not given in trajectory.
    warmup : int (ms)
    simtime : int (ms)
        Duration of the main simulation.
    trajectory :
        Either:
          - list of (time_ms, value) tuples (relative to start of main sim), or
          - a tuple of two lists: (times_ms_list, values_list)
        Times are relative to the **start of the main simulation**.
        At each given time, q_stc is set to the given value and stays until the next change.
    return_as : "list_of_tuples" or "tuple_of_lists"

    Returns
    -------
    If "list_of_tuples": [(time0, q0), (time1, q1), ..., (simtime, q_last)]
    If "tuple_of_lists": ([times...], [values...]) aligned with the above.
    """

    def _set_q(value):
        # original code uses list-wrapped parameter
        for pop in EI_Network.Populations[0]:
            pop.set({"q_stc": [value]})

    # --- Normalize trajectory input to list[(t,q)] ---
    if isinstance(trajectory, tuple) and len(trajectory) == 2:
        times, values = trajectory
        if len(times) != len(values):
            raise ValueError("trajectory tuple must have same-length times and values")
        traj = list(zip(times, values))
    else:
        traj = list(trajectory)  # assume iterable of (t,q)

    # Clamp times to [0, simtime], drop anything outside
    clamped = []
    for t, q in traj:
        if t < 0 or t > simtime:
            # clamp rather than drop: move to boundary
            t = 0 if t < 0 else simtime
        clamped.append((int(t), float(q)))

    # Sort by time, keep the last value for duplicate times
    clamped.sort(key=lambda x: x[0])
    dedup = {}
    for t, q in clamped:
        dedup[t] = q
    traj = sorted(dedup.items(), key=lambda x: x[0])  # [(t,q), ...]

    # Ensure a value at t=0
    if not traj or traj[0][0] > 0:
        traj.insert(0, (0, float(q_stc_initial)))

    # --- Warmup with initial value (pre-main sim) ---
    _set_q(q_stc_initial)
    print(f'Simulate warmup: {int(warmup)}')
    nest.Simulate(int(warmup))

    # --- Piecewise-constant simulation over main window ---
    records = []  # [(time, value)] checkpoints (relative to main sim start)
    for i, (t_curr, q_curr) in enumerate(traj):
        _set_q(q_curr)  # set value at this time
        records.append((t_curr, q_curr))

        # Determine how long to simulate until next change (or end)
        t_next = traj[i + 1][0] if i + 1 < len(traj) else simtime
        dt = int(t_next - t_curr)
        if dt > 0:
            print(f'Simulate dt: {int(dt)}')
            nest.Simulate(dt)

    # Ensure final marker at simtime
    if records[-1][0] != simtime:
        records.append((simtime, records[-1][1]))

    if return_as == "tuple_of_lists":
        times, values = zip(*records)
        return list(times), list(values)
    return records




if __name__ == '__main__':
    #get SLURM environment variables
    CPUcount = int(os.environ.get('SLURM_CPUS_PER_TASK', '10'))
    JobID = os.environ.get('SLURM_JOB_ID', '0')
    ArrayID = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    #get git hash
    gitHash = os.popen('git rev-parse HEAD').read().strip()

    #get enviroment variables for simulation parmaeters
    randseed = int(os.environ.get('randseed',ArrayID))+1
    print(randseed)
    # Jep, jip_ratio, I_th_E, I_th_I, N_E, tau_stc, q_stc
    jep = 5.3
    jip_ratio = float(os.environ.get('jip_ratio', '0.75'))
    I_th_E = 1.9
    I_th_I = 1.5
    N_E = 4000
    N_I = 1000


    tau_stc = 100.0
    q_stc = 0.025

    # Warm up 500 ms, then run 5000 ms, changing q_stc to 0.5 at t=2000 ms
    #times_q = [0, 4000+prerun, 8000+prerun, 12000+prerun, 16000+prerun]
    #values_q = [q_stc, 0.3, 0.2, 0.1, 0.0]


    #get enviroment variables for output_path
    output_path = os.environ.get('output_path', '../output/')
    #get enviroment variables for simulation protocol
    #Trials per direction, ITI, Preperatory duration, Stimulus duration, Stimulus intensity, Stimulus type
    

    #calculate inhibitory weight
    jip = 1. + (jep - 1) * jip_ratio

    #test if output path ends with a directory seperator
    if output_path.endswith(os.path.sep):
        #create a short name for the output file based on the parameters hash
        output_file = 'job_{}_array_{}'.format(JobID, ArrayID)
        #hash the output file name and encode it to utf-8 string
        #output_file = hashlib.sha1(output_file.encode('utf-8')).hexdigest()
        #add the file extension
        output_file = output_file + '.pkl'
        #join the output path and the output file
        output_path = os.path.join(output_path, output_file)

    #create output directory if it does not exist
    try:
        output_dir = os.path.dirname(output_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except:
        pass


    overrides = {
        'simulation': {
            'n_jobs': CPUcount,
            'dt': 0.1,
            'warmup': 5000.0,
            'duration': 1_000_000.0,
            'randseed': randseed,
        },
        'network': {
            'population': {'excitatory': N_E, 'inhibitory': N_I},
            'clusters': {'count': 10, 'jplus': [[float(jep), float(jip)], [float(jip), float(jip)]]},
        },
        'neuron': {
            'model': 'gif_psc_exp',
            'I_th': {'excitatory': float(I_th_E), 'inhibitory': float(I_th_I)},
            'tau_stc': float(tau_stc),
            'q_stc': float(q_stc),
        },
        'stimulation': {'background': 'DC'},
        'only_E_SFA': True,
    }

    EI_Network = ClusterModelNEST.ClusteredNetworkNEST(default, overrides)
    EI_Network.setup_network()
    EI_Network.simulate()
    cfg = EI_Network.get_parameter()

    #values_q = [q_stc * (1 - i / 5) for i in range(4)]
    #times_q = [i * (params['simtime'] - 2*prerun) // 5 + (prerun if i > 0 else 0) for i in range(4)]

    #time_stim, q_stc_hist = simulate_with_trajectory(EI_Network, nest, q_stc_initial=q_stc,
    #                                          warmup=params['warmup'], simtime=params['simtime'],
    #                                          trajectory=(times_q, values_q),
    #                                          return_as="tuple_of_lists")

    # save data

    spiketimes=EI_Network.get_recordings()
    exc_mask = (spiketimes[1] < N_E)
    inh_mask = (spiketimes[1] >= N_E)

    # ----- config -----
    BIN_MS = 5
    Q = cfg['network']['clusters']['count']
    E_PER_CLUST = N_E // Q

    # ----- inputs (from your env) -----
    spike_times_ms = np.array(spiketimes[0])[exc_mask]  # excitatory-only times
    spike_neuron_ids = np.array(spiketimes[1])[exc_mask]  # excitatory-only ids

    # ----- time axis (bin centers) -----
    T = float(cfg['simulation']['duration'])
    nbins = int(np.ceil(T / BIN_MS))
    edges = np.arange(nbins + 1) * BIN_MS
    t_ms = (edges[:-1] + edges[1:]) / 2.0

    # ----- bin spikes per excitatory cluster -----
    counts = np.zeros((nbins, Q), dtype=np.int32)
    if spike_times_ms.size:
        bin_idx = np.clip((spike_times_ms // BIN_MS).astype(int), 0, nbins - 1)
        clus = (spike_neuron_ids // E_PER_CLUST).astype(int)
        np.add.at(counts, (bin_idx, clus), 1)

    bin_s = BIN_MS / 1000.0
    rates_hz = counts / (bin_s * E_PER_CLUST)  # [nbins, Q]
    dom = np.argmax(rates_hz, axis=1)  # dominant cluster per bin


    with open(output_path, 'wb') as outfile:
        pickle.dump(EI_Network.get_recordings(), outfile)
        pickle.dump(EI_Network.get_parameter(), outfile)
        pickle.dump({"time": t_ms, "label": dom}, outfile)
        pickle.dump({'gitHash': gitHash, 'JobID': JobID, 'ArrayID': ArrayID}, outfile)
