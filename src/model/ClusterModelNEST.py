import nest
import numpy as np

try:
    from .. import GeneralHelper
    from . import ClusterHelper
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    import importlib

    GeneralHelper = importlib.import_module("src.GeneralHelper")  # type: ignore
    ClusterHelper = importlib.import_module("src.model.ClusterHelper")  # type: ignore


class ClusteredNetworkNEST:
    """
    EI-clustered network helper for NEST simulations driven by the hierarchical YAML configuration.
    """

    def __init__(self, default_config, overrides):
        self.config = GeneralHelper.mergeParams(dict(overrides or {}), default_config)
        self.clean_network()

    # --------------------------------------------------------------------- #
    # Configuration helpers
    # --------------------------------------------------------------------- #

    def _neuron_value(self, key, cell_type=None, default=None):
        neuron_cfg = self.config.get('neuron', {})
        value = neuron_cfg.get(key, default)
        if value is None:
            return default
        if isinstance(value, dict):
            if cell_type is None:
                return value
            if cell_type in value:
                return value[cell_type]
            if 'default' in value:
                return value['default']
            return next(iter(value.values()), default)
        return value

    def _neuron_model(self, cell_type):
        model_cfg = self.config.get('neuron', {}).get('model', 'iaf_psc_exp')
        if isinstance(model_cfg, dict):
            if cell_type in model_cfg:
                return model_cfg[cell_type]
            if 'default' in model_cfg:
                return model_cfg['default']
            return next(iter(model_cfg.values()), 'iaf_psc_exp')
        return model_cfg

    def _baseline_current(self, cell_type):
        stim_cfg = self.config.get('stimulation', {})
        if stim_cfg.get('background', 'DC') != 'DC':
            return 0.0

        I_th = self._neuron_value('I_th', cell_type, None)
        base_current = self._neuron_value('I_e', cell_type, 0.0)
        if I_th is None:
            return base_current

        tau_m = self._neuron_value('tau_m', cell_type)
        V_th = self._neuron_value('V_th', cell_type)
        neuron_cfg = self.config['neuron']
        E_L = neuron_cfg['E_L']
        C_m = neuron_cfg['C_m']

        if tau_m is None or V_th is None:
            return base_current

        return I_th * (V_th - E_L) / tau_m * C_m

    def _build_neuron_params(self, cell_type, model, baseline_current):
        neuron_cfg = self.config['neuron']
        params = {
            'E_L': neuron_cfg['E_L'],
            'C_m': neuron_cfg['C_m'],
            't_ref': neuron_cfg['t_ref'],
            'V_reset': neuron_cfg['V_reset'],
            'I_e': baseline_current,
        }

        tau_m = self._neuron_value('tau_m', cell_type)
        if tau_m is not None:
            params['tau_m'] = tau_m

        V_th = self._neuron_value('V_th', cell_type)
        if V_th is not None:
            params['V_th'] = V_th

        tau_syn_ex = self._neuron_value('tau_syn', 'excitatory')
        tau_syn_in = self._neuron_value('tau_syn', 'inhibitory')
        if tau_syn_ex is not None:
            params['tau_syn_ex'] = tau_syn_ex
        if tau_syn_in is not None:
            params['tau_syn_in'] = tau_syn_in

        delta_noise = self._neuron_value('delta', cell_type, self._neuron_value('delta'))
        if delta_noise is not None:
            params['delta'] = delta_noise

        rho_noise = self._neuron_value('rho', cell_type, self._neuron_value('rho'))
        if rho_noise is not None:
            params['rho'] = rho_noise

        if 'gif_psc_exp' in model:
            # Adaptation parameters are only meaningful for gif_psc_exp neurons
            lambda_0 = self._neuron_value('lambda_0', cell_type, self._neuron_value('lambda_0'))
            q_sfa = self._neuron_value('q_sfa', cell_type, self._neuron_value('q_sfa'))
            tau_sfa = self._neuron_value('tau_sfa', cell_type, self._neuron_value('tau_sfa'))
            q_stc = self._neuron_value('q_stc', cell_type, self._neuron_value('q_stc'))
            tau_stc = self._neuron_value('tau_stc', cell_type, self._neuron_value('tau_stc'))
            delta_v = self._neuron_value('Delta_V', cell_type, self._neuron_value('Delta_V'))

            v_th_value = params.pop('V_th', None)
            tau_m_value = params.pop('tau_m', None)
            if v_th_value is None or tau_m_value is None:
                raise ValueError("gif_psc_exp neurons require both V_th and tau_m to be configured.")

            params['V_T_star'] = v_th_value
            params['g_L'] = params['C_m'] / tau_m_value

            if lambda_0 is not None:
                params['lambda_0'] = lambda_0
            if q_sfa is not None:
                params['q_sfa'] = [q_sfa]
            if tau_sfa is not None:
                params['tau_sfa'] = [tau_sfa]
            if q_stc is not None:
                params['q_stc'] = [q_stc]
            if tau_stc is not None:
                params['tau_stc'] = [tau_stc]
            if delta_v is not None:
                params['Delta_V'] = delta_v

        return params

    def _apply_current_variation(self, population, cell_type):
        delta = self._neuron_value('I_e_delta', cell_type, 0.0)
        if not delta or delta <= 0:
            return

        currents = nest.GetStatus(population, 'I_e')
        varied = [
            (1 - 0.5 * delta + np.random.rand() * delta) * current for current in currents
        ]
        nest.SetStatus(population, [{'I_e': val} for val in varied])

    def _resolve_initial_vm(self, cell_type):
        vm_cfg = self.config['neuron'].get('V_m', 'rand')
        if isinstance(vm_cfg, dict):
            if cell_type in vm_cfg:
                return vm_cfg[cell_type]
            if 'default' in vm_cfg:
                return vm_cfg['default']
            return next(iter(vm_cfg.values()), 'rand')
        return vm_cfg

    def _set_initial_state(self, population, cell_type, baseline_current):
        vm_value = self._resolve_initial_vm(cell_type)
        if vm_value != 'rand':
            nest.SetStatus(population, [{'V_m': vm_value}] * len(population))
            return

        neuron_cfg = self.config['neuron']
        tau_m = self._neuron_value('tau_m', cell_type)
        V_th = self._neuron_value('V_th', cell_type)
        if tau_m is None or V_th is None:
            nest.SetStatus(population, [{'V_m': neuron_cfg['V_reset']}] * len(population))
            return

        t_ref = neuron_cfg['t_ref']
        E_L = neuron_cfg['E_L']
        C_m = neuron_cfg['C_m']
        V_reset = neuron_cfg['V_reset']

        T_0 = t_ref + ClusterHelper.FPT(tau_m, E_L, baseline_current, C_m, V_th, V_reset)
        if np.isnan(T_0):
            T_0 = 10.0

        values = [
            ClusterHelper.V_FPT(
                tau_m,
                E_L,
                baseline_current,
                C_m,
                T_0 * np.random.rand(),
                V_th,
                t_ref,
            )
            for _ in range(len(population))
        ]
        nest.SetStatus(population, [{'V_m': val} for val in values])

    def _create_population(self, cell_type, count, baseline_current):
        model = self._neuron_model(cell_type)
        params = self._build_neuron_params(cell_type, model, baseline_current)
        population = nest.Create(model, count, params=dict(params))
        self._apply_current_variation(population, cell_type)
        self._set_initial_state(population, cell_type, baseline_current)
        return population

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def clean_network(self):
        self.Populations = []
        self.RecordingDevices = []
        self.Currentsources = []

    def setup_nest(self):
        nest.ResetKernel()
        nest.set_verbosity('M_WARNING')
        sim_cfg = self.config['simulation']
        nest.local_num_threads = sim_cfg.get('n_jobs', 1)
        nest.resolution = sim_cfg.get('dt')
        if sim_cfg.get('randseed') is None:
            sim_cfg['randseed'] = int(np.random.randint(1_000_000))
        nest.rng_seed = sim_cfg['randseed']

    def create_populations(self):
        network_cfg = self.config['network']
        populations = network_cfg['population']
        clusters_cfg = network_cfg['clusters']

        n_exc = populations['excitatory']
        n_inh = populations['inhibitory']
        Q = clusters_cfg['count']

        assert n_exc % Q == 0, 'N_E needs to be evenly divisible by Q'
        assert n_inh % Q == 0, 'N_I needs to be evenly divisible by Q'

        baseline_exc = self._baseline_current('excitatory')
        baseline_inh = self._baseline_current('inhibitory')

        excitatory = [
            self._create_population('excitatory', n_exc // Q, baseline_exc)
            for _ in range(Q)
        ]
        inhibitory = [
            self._create_population('inhibitory', n_inh // Q, baseline_inh)
            for _ in range(Q)
        ]

        self.Populations = [excitatory, inhibitory]

    def create_connect_Poisson(self):
        stim_cfg = self.config.get('stimulation', {})
        if stim_cfg.get('background', 'DC') == "DC":
            return

        baseline_exc = self._baseline_current('excitatory')
        baseline_inh = self._baseline_current('inhibitory')

        tau_syn_ex = self._neuron_value('tau_syn', 'excitatory')
        if tau_syn_ex is None:
            raise ValueError("Excitatory synapse time constant is required for Poisson background.")

        p_rate = 150
        self.Populations.append([nest.Create("poisson_generator", params={"rate": p_rate})])
        self.Populations.append([nest.Create("poisson_generator", params={"rate": p_rate})])

        J_Poisson_E = 1000.0 * baseline_exc / (tau_syn_ex * p_rate) if tau_syn_ex else 0.0
        J_Poisson_I = 1000.0 * baseline_inh / (tau_syn_ex * p_rate) if tau_syn_ex else 0.0

        delay_cfg = self.config['network']['connectivity']['delay']
        if isinstance(delay_cfg, (list, tuple)):
            delay = nest.random.uniform(min=min(delay_cfg), max=max(delay_cfg))
        else:
            delay = delay_cfg

        nest.CopyModel("static_synapse", "noise_E", {"weight": J_Poisson_E, "delay": delay})
        nest.CopyModel("static_synapse", "noise_I", {"weight": J_Poisson_I, "delay": delay})

        for post in self.Populations[0]:
            nest.Connect(self.Populations[2][0], post, syn_spec="noise_E")
        for post in self.Populations[1]:
            nest.Connect(self.Populations[3][0], post, syn_spec="noise_I")

    def connect(self):
        network_cfg = self.config['network']
        pop_cfg = network_cfg['population']
        conn_cfg = network_cfg['connectivity']
        weight_cfg = network_cfg['weights']
        cluster_cfg = network_cfg['clusters']
        neuron_cfg = self.config['neuron']

        n_exc = pop_cfg['excitatory']
        n_inh = pop_cfg['inhibitory']
        total_units = n_exc + n_inh

        ps = np.asarray(conn_cfg['probabilities'], dtype=float)
        js = np.asarray(weight_cfg['matrix'], dtype=float)
        scaling = weight_cfg.get('scale', 1.0)
        modifiers = weight_cfg.get('modifiers', {})
        ge = modifiers.get('ge', 1.0)
        gi = modifiers.get('gi', 1.0)
        gie = modifiers.get('gie', 1.0)

        Q = cluster_cfg['count']
        jplus_cfg = cluster_cfg.get('jplus')
        if jplus_cfg is None:
            jplus_cfg = [[float('nan'), float('nan')], [float('nan'), float('nan')]]
        jplus = np.asarray(jplus_cfg, dtype=float)

        if np.isnan(jplus).any():
            jep = cluster_cfg.get('jep')
            jip_ratio = cluster_cfg.get('jip_ratio', cluster_cfg.get('rj'))
            if jep is None or jip_ratio is None:
                raise ValueError("Cluster configuration requires either a full 'jplus' matrix or both 'jep' and 'jip_ratio'/'rj'.")
            jip = 1.0 + (float(jep) - 1.0) * float(jip_ratio)
            jplus = np.asarray([[float(jep), jip], [jip, jip]], dtype=float)

        delay_cfg = conn_cfg['delay']
        if isinstance(delay_cfg, (list, tuple)):
            delay = nest.random.uniform(min=min(delay_cfg), max=max(delay_cfg))
        else:
            delay = delay_cfg

        if np.isnan(js).any():
            calc_params = {
                'N_E': n_exc,
                'N_I': n_inh,
                'ps': ps,
                'ge': ge,
                'gi': gi,
                'gie': gie,
                'V_th_E': self._neuron_value('V_th', 'excitatory'),
                'V_th_I': self._neuron_value('V_th', 'inhibitory'),
                'tau_E': self._neuron_value('tau_m', 'excitatory'),
                'tau_I': self._neuron_value('tau_m', 'inhibitory'),
                'E_L': neuron_cfg['E_L'],
                'neuron_type': self._neuron_model('excitatory'),
                'tau_syn_ex': self._neuron_value('tau_syn', 'excitatory'),
                'tau_syn_in': self._neuron_value('tau_syn', 'inhibitory'),
            }
            js = ClusterHelper.calc_js(calc_params)
        js = np.asarray(js, dtype=float) * scaling

        if Q > 1:
            jminus = (Q - jplus) / float(Q - 1)
        else:
            jplus = np.ones((2, 2))
            jminus = np.ones((2, 2))

        def connect_block(pre_pops, post_pops, idx_target, idx_source, conn_probability, fixed_indegree):
            j_base = js[idx_target, idx_source] / np.sqrt(total_units)
            if fixed_indegree:
                target_size = n_exc if idx_target == 0 else n_inh
                indegree = int(conn_probability * target_size / Q)
                conn_params = {
                    'rule': 'fixed_indegree',
                    'indegree': indegree,
                    'allow_autapses': False,
                    'allow_multapses': False,
                }
            else:
                conn_params = {
                    'rule': 'pairwise_bernoulli',
                    'p': conn_probability,
                    'allow_autapses': False,
                    'allow_multapses': False,
                }

            for i, pre in enumerate(pre_pops):
                for j, post in enumerate(post_pops):
                    weight = jplus[idx_target, idx_source] if i == j else jminus[idx_target, idx_source]
                    nest.Connect(pre, post, conn_params, syn_spec={"weight": weight * j_base, "delay": delay})

        fixed_indegree = conn_cfg.get('fixed_indegree', False)
        connect_block(self.Populations[0], self.Populations[0], 0, 0, ps[0, 0], fixed_indegree)
        connect_block(self.Populations[1], self.Populations[0], 0, 1, ps[0, 1], fixed_indegree)
        connect_block(self.Populations[0], self.Populations[1], 1, 0, ps[1, 0], fixed_indegree)
        connect_block(self.Populations[1], self.Populations[1], 1, 1, ps[1, 1], fixed_indegree)

    def create_stimulation(self):
        stim_cfg = self.config.get('stimulation', {})
        warmup = self.config['simulation']['warmup']
        self.Currentsources = []

        clusters = stim_cfg.get('clusters')
        if clusters:
            stim_amp = stim_cfg.get('amplitude', 0.0)
            stim_starts = stim_cfg.get('starts', [])
            stim_ends = stim_cfg.get('ends', [])
            amplitude_values = []
            amplitude_times = []
            for start, end in zip(stim_starts, stim_ends):
                amplitude_times.append(start + warmup)
                amplitude_values.append(stim_amp)
                amplitude_times.append(end + warmup)
                amplitude_values.append(0.0)
            self.Currentsources = [nest.Create('step_current_generator')]
            for stim_cluster in clusters:
                nest.Connect(self.Currentsources[0], self.Populations[0][stim_cluster])
            nest.SetStatus(
                self.Currentsources[0],
                {'amplitude_times': amplitude_times, 'amplitude_values': amplitude_values},
            )
            return

        multi_cfg = stim_cfg.get('multi')
        if not multi_cfg or multi_cfg.get('clusters') is None:
            return

        clusters_list = multi_cfg.get('clusters', [])
        amps_list = multi_cfg.get('amps', [])
        times_list = multi_cfg.get('times', [])
        delays = multi_cfg.get('delay')
        if delays is not None and not isinstance(delays, (list, tuple)):
            delays = [delays]

        for stim_clusters, amplitudes, times in zip(clusters_list, amps_list, times_list):
            generator = nest.Create('step_current_generator')
            self.Currentsources.append(generator)
            warm_times = [t + warmup for t in times[1:]]
            nest.SetStatus(generator, {'amplitude_times': warm_times, 'amplitude_values': amplitudes[1:]})
            if delays:
                if len(delays) == 1:
                    syn_dict = {"delay": nest.random.uniform(min=0.1, max=delays[0])}
                else:
                    syn_dict = {"delay": nest.random.uniform(min=delays[0], max=delays[1])}
            else:
                syn_dict = {}
            for stim_cluster in stim_clusters:
                nest.Connect(generator, self.Populations[0][stim_cluster], syn_spec=syn_dict)

    def create_recording_devices(self):
        self.RecordingDevices = [nest.Create("spike_recorder")]
        self.RecordingDevices[0].record_to = "memory"

        all_units = self.Populations[0][0]
        for E_pop in self.Populations[0][1:]:
            all_units += E_pop
        for I_pop in self.Populations[1]:
            all_units += I_pop
        nest.Connect(all_units, self.RecordingDevices[0], "all_to_all")

    def setup_network(self):
        self.clean_network()
        self.setup_nest()
        self.create_populations()
        self.connect()
        self.create_connect_Poisson()
        self.create_recording_devices()
        self.create_stimulation()

    def simulate(self):
        sim_cfg = self.config['simulation']
        nest.Simulate(sim_cfg['warmup'] + sim_cfg['duration'])

    def get_recordings(self):
        events = nest.GetStatus(self.RecordingDevices[0], 'events')[0]
        spiketimes = np.append(events['times'][None, :], events['senders'][None, :], axis=0)
        spiketimes[1] -= 1
        warmup = self.config['simulation']['warmup']
        spiketimes = spiketimes[:, spiketimes[0] >= warmup]
        spiketimes[0] -= warmup
        return spiketimes

    def get_parameter(self):
        return self.config

    def get_firing_rates(self, spiketimes=None):
        if spiketimes is None:
            spiketimes = self.get_recordings()
        populations = self.config['network']['population']
        simtime = float(self.config['simulation']['duration'])
        n_exc = populations['excitatory']
        n_inh = populations['inhibitory']
        e_count = spiketimes[:, spiketimes[1] < n_exc].shape[1]
        i_count = spiketimes[:, spiketimes[1] >= n_exc].shape[1]
        e_rate = e_count / float(n_exc) / simtime * 1000.0
        i_rate = i_count / float(n_inh) / simtime * 1000.0
        return e_rate, i_rate

    def create_and_simulate(self):
        self.setup_network()
        self.simulate()
        return self.get_recordings()


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from config import add_override_arguments, load_from_args

    parser = argparse.ArgumentParser(description="Run a clustered NEST network demo.")
    add_override_arguments(parser)
    parser.add_argument(
        "--no-demo",
        dest="demo",
        action="store_false",
        help="Skip built-in demo overrides (defaults to enabled unless overrides are provided).",
    )
    parser.set_defaults(demo=True)
    args = parser.parse_args()

    base_config = load_from_args(args)

    demo_overrides = {}
    if args.demo and not getattr(args, "overwrite", []):
        demo_overrides = {
            'simulation': {'n_jobs': 4, 'warmup': 500.0, 'duration': 1200.0},
            'stimulation': {
                'clusters': [3],
                'amplitude': 2.0,
                'starts': [600.0],
                'ends': [1000.0],
            },
        }

    EI_cluster = ClusteredNetworkNEST(base_config, demo_overrides)
    spikes = EI_cluster.create_and_simulate()
    print(EI_cluster.get_parameter())
    plt.figure()
    plt.plot(spikes[0, :], spikes[1, :], '.')
    plt.savefig('NEST.png')
