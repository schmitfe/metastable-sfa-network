import nest
import numpy as np
import time
import pickle
import signal

from .. import GeneralHelper
from . import ClusterHelper, ClusterModelBase


class ClusteredNetworkNEST(ClusterModelBase.ClusteredNetworkBase):
    """ 
    Creates an object with functions to create neuron populations, 
    stimulation devices and recording devices for an EI-clustered network.
    Provides also function to initialize NEST (v3.x), simulate the network and
    to grab the spike data.
    """

    def __init__(self, defaultValues, parameters):
        """
        Creates an object with functions to create neuron populations,
        stimulation devices and recording devices for an EI-clustered network.
        Initializes the object. Creates the attributes Populations, RecordingDevices and 
        Currentsources to be filled during network construction.
        Attribute params contains all parameters used to construct network.

        Parameters:
            defaultValues (module): A Module which contains the default configuration
            parameters (dict):      Dictionary with parameters which should be modified from their default values
        """
        super().__init__(defaultValues, parameters)
        self.Populations = []
        self.RecordingDevices = []
        self.Currentsources = []

    def _neuron_value(self, key, cell_type=None, default=None):
        value = self.config['neuron'].get(key, default)
        if isinstance(value, dict):
            if cell_type is None:
                return value
            return value.get(cell_type, default)
        return value

    def clean_network(self):
        """
        Creates empty attributes of a network.
        """
        self.Populations = []
        self.RecordingDevices = []
        self.Currentsources = []

    def setup_nest(self):
        """ Initializes the NEST kernel.
        Reset the NEST kernel and pass parameters to it.
        Updates randseed of parameters to the actual used one if none is supplied.
        """
        nest.ResetKernel()
        nest.set_verbosity('M_WARNING')
        sim_cfg = self.config['simulation']
        nest.local_num_threads = sim_cfg.get('n_jobs', 1)
        nest.resolution = sim_cfg.get('dt')
        if sim_cfg.get('randseed') is None:
            sim_cfg['randseed'] = int(np.random.randint(1_000_000))
        nest.rng_seed = sim_cfg['randseed']
        #nest.print_time = True

    def create_populations(self):
        """
        Creates Q excitatory and inhibitory neuron populations with the parameters of the network.
        """
        network = self.config['network']
        neuron_cfg = self.config['neuron']
        stim_cfg = self.config.get('stimulation', {})

        n_exc = network['population']['excitatory']
        n_inh = network['population']['inhibitory']
        Q = network['clusters']['count']
        assert n_exc % Q == 0, 'N_E needs to be evenly divisible by Q'
        assert n_inh % Q == 0, 'N_I needs to be evenly divisible by Q'

        total_units = n_exc + n_inh
        background_mode = stim_cfg.get('background', 'DC')

        E_L = neuron_cfg['E_L']
        C_m = neuron_cfg['C_m']
        V_reset = neuron_cfg['V_reset']
        t_ref = neuron_cfg['t_ref']

        tau_m_e = self._neuron_value('tau_m', 'excitatory')
        tau_m_i = self._neuron_value('tau_m', 'inhibitory')
        tau_syn_ex = self._neuron_value('tau_syn', 'excitatory')
        tau_syn_in = self._neuron_value('tau_syn', 'inhibitory')

        V_th_e = self._neuron_value('V_th', 'excitatory')
        V_th_i = self._neuron_value('V_th', 'inhibitory')
        I_e_e = self._neuron_value('I_e', 'excitatory', 0.0)
        I_e_i = self._neuron_value('I_e', 'inhibitory', 0.0)
        I_th_e = self._neuron_value('I_th', 'excitatory', None)
        I_th_i = self._neuron_value('I_th', 'inhibitory', None)

        if background_mode == "DC":
            I_xE = (
                I_e_e
                if I_th_e is None
                else I_th_e * (V_th_e - E_L) / tau_m_e * C_m
            )
            I_xI = (
                I_e_i
                if I_th_i is None
                else I_th_i * (V_th_i - E_L) / tau_m_i * C_m
            )
        else:
            I_xE = 0.0
            I_xI = 0.0

        neuron_model = neuron_cfg['model']
        is_gif = 'gif_psc_exp' in neuron_model
        is_iaf = 'iaf_psc_exp' in neuron_model
        delta_noise = neuron_cfg.get('delta')
        rho_noise = neuron_cfg.get('rho')
        def per_type(key, default=None):
            base = self._neuron_value(key, default=default)
            exc = self._neuron_value(key, 'excitatory', base)
            inh = self._neuron_value(key, 'inhibitory', base)
            return exc, inh

        lambda_0_e, lambda_0_i = per_type('lambda_0', neuron_cfg.get('lambda_0'))
        q_sfa_e, q_sfa_i = per_type('q_sfa', neuron_cfg.get('q_sfa'))
        tau_sfa_e, tau_sfa_i = per_type('tau_sfa', neuron_cfg.get('tau_sfa'))
        q_stc_e, q_stc_i = per_type('q_stc', neuron_cfg.get('q_stc'))
        tau_stc_e, tau_stc_i = per_type('tau_stc', neuron_cfg.get('tau_stc'))
        delta_v_e, delta_v_i = per_type('Delta_V', neuron_cfg.get('Delta_V'))
        V_m_init = neuron_cfg.get('V_m', 'rand')
        I_e_delta = self._neuron_value('I_e_delta', cell_type=None, default={'excitatory': 0.0, 'inhibitory': 0.0})
        if not isinstance(I_e_delta, dict):
            I_e_delta = {'excitatory': I_e_delta, 'inhibitory': I_e_delta}

        only_E_SFA = self.config.get('only_E_SFA', False)

        E_neuron_params = {
            'E_L': E_L,
            'C_m': C_m,
            'tau_m': tau_m_e,
            't_ref': t_ref,
            'V_th': V_th_e,
            'V_reset': V_reset,
            'I_e': I_xE,
        }
        I_neuron_params = {
            'E_L': E_L,
            'C_m': C_m,
            'tau_m': tau_m_i,
            't_ref': t_ref,
            'V_th': V_th_i,
            'V_reset': V_reset,
            'I_e': I_xI,
        }

        if only_E_SFA:
            I_neuron_type = 'iaf_psc_exp'
            I_neuron_params['tau_syn_in'] = tau_syn_in
            I_neuron_params['tau_syn_ex'] = tau_syn_ex
            if delta_noise is not None:
                I_neuron_params['delta'] = delta_noise
            if rho_noise is not None:
                I_neuron_params['rho'] = rho_noise

            if is_iaf or is_gif:
                E_neuron_params['tau_syn_ex'] = tau_syn_ex
                E_neuron_params['tau_syn_in'] = tau_syn_in
                if delta_noise is not None:
                    E_neuron_params['delta'] = delta_noise
                if rho_noise is not None:
                    E_neuron_params['rho'] = rho_noise

            if is_gif:
                E_neuron_params['V_T_star'] = E_neuron_params.pop('V_th')
                E_neuron_params['g_L'] = E_neuron_params['C_m'] / E_neuron_params.pop('tau_m')
                if lambda_0_e is not None:
                    E_neuron_params['lambda_0'] = lambda_0_e
                if q_sfa_e is not None:
                    E_neuron_params['q_sfa'] = [q_sfa_e]
                if tau_sfa_e is not None:
                    E_neuron_params['tau_sfa'] = [tau_sfa_e]
                if q_stc_e is not None:
                    E_neuron_params['q_stc'] = [q_stc_e]
                if tau_stc_e is not None:
                    E_neuron_params['tau_stc'] = [tau_stc_e]
                if delta_v_e is not None:
                    E_neuron_params['Delta_V'] = delta_v_e
        else:
            I_neuron_type = neuron_model
            if is_iaf or is_gif:
                E_neuron_params['tau_syn_ex'] = tau_syn_ex
                E_neuron_params['tau_syn_in'] = tau_syn_in
                I_neuron_params['tau_syn_in'] = tau_syn_in
                I_neuron_params['tau_syn_ex'] = tau_syn_ex
                if delta_noise is not None:
                    E_neuron_params['delta'] = delta_noise
                    I_neuron_params['delta'] = delta_noise
                if rho_noise is not None:
                    E_neuron_params['rho'] = rho_noise
                    I_neuron_params['rho'] = rho_noise
            if is_gif:
                E_neuron_params['V_T_star'] = E_neuron_params.pop('V_th')
                I_neuron_params['V_T_star'] = I_neuron_params.pop('V_th')
                E_neuron_params['g_L'] = E_neuron_params['C_m'] / E_neuron_params.pop('tau_m')
                I_neuron_params['g_L'] = I_neuron_params['C_m'] / I_neuron_params.pop('tau_m')
                if lambda_0_e is not None:
                    E_neuron_params['lambda_0'] = lambda_0_e
                if lambda_0_i is not None:
                    I_neuron_params['lambda_0'] = lambda_0_i
                if q_sfa_e is not None:
                    E_neuron_params['q_sfa'] = [q_sfa_e]
                if q_sfa_i is not None:
                    I_neuron_params['q_sfa'] = [q_sfa_i]
                if tau_sfa_e is not None:
                    E_neuron_params['tau_sfa'] = [tau_sfa_e]
                if tau_sfa_i is not None:
                    I_neuron_params['tau_sfa'] = [tau_sfa_i]
                if q_stc_e is not None:
                    E_neuron_params['q_stc'] = [q_stc_e]
                if q_stc_i is not None:
                    I_neuron_params['q_stc'] = [q_stc_i]
                if tau_stc_e is not None:
                    E_neuron_params['tau_stc'] = [tau_stc_e]
                if tau_stc_i is not None:
                    I_neuron_params['tau_stc'] = [tau_stc_i]
                if delta_v_e is not None:
                    E_neuron_params['Delta_V'] = delta_v_e
                if delta_v_i is not None:
                    I_neuron_params['Delta_V'] = delta_v_i
        if not is_gif:
            # restore tau_m values if no conversion happened
            E_neuron_params.setdefault('tau_m', tau_m_e)
            I_neuron_params.setdefault('tau_m', tau_m_i)

        # create the neuron populations
        E_pops = []
        I_pops = []
        for _ in range(Q):
            E_pop = nest.Create(neuron_model, int(n_exc / Q))
            nest.SetStatus(E_pop, E_neuron_params)
            E_pops.append(E_pop)
        for _ in range(Q):
            I_pop = nest.Create(I_neuron_type, int(n_inh / Q))
            nest.SetStatus(I_pop, I_neuron_params)
            I_pops.append(I_pop)

        if I_e_delta.get('excitatory', 0.0) > 0:
            delta_val = I_e_delta['excitatory']
            for E_pop in E_pops:
                currents = nest.GetStatus(E_pop, 'I_e')
                nest.SetStatus(E_pop, [
                    {'I_e': (1 - 0.5 * delta_val + np.random.rand() * delta_val) * current}
                    for current in currents
                ])

        if I_e_delta.get('inhibitory', 0.0) > 0:
            delta_val = I_e_delta['inhibitory']
            for I_pop in I_pops:
                currents = nest.GetStatus(I_pop, 'I_e')
                nest.SetStatus(I_pop, [
                    {'I_e': (1 - 0.5 * delta_val + np.random.rand() * delta_val) * current}
                    for current in currents
                ])

        if V_m_init == 'rand':
            T_0_E = t_ref + ClusterHelper.FPT(tau_m_e, E_L, I_xE, C_m, V_th_e, V_reset)
            if np.isnan(T_0_E):
                T_0_E = 10.0
            for E_pop in E_pops:
                nest.SetStatus(E_pop, [
                    {'V_m': ClusterHelper.V_FPT(tau_m_e, E_L, I_xE, C_m, T_0_E * np.random.rand(), V_th_e, t_ref)}
                    for _ in range(len(E_pop))
                ])

            T_0_I = t_ref + ClusterHelper.FPT(tau_m_i, E_L, I_xI, C_m, V_th_i, V_reset)
            if np.isnan(T_0_I):
                T_0_I = 10.0
            for I_pop in I_pops:
                nest.SetStatus(I_pop, [
                    {'V_m': ClusterHelper.V_FPT(tau_m_i, E_L, I_xI, C_m, T_0_I * np.random.rand(), V_th_i, t_ref)}
                    for _ in range(len(I_pop))
                ])
        else:
            nest.SetStatus(
                nest.NodeCollection(range(1, total_units + 1)),
                [{'V_m': V_m_init} for _ in range(total_units)],
            )

        self.Populations = [E_pops, I_pops]

    def create_connect_Poisson(self):
        stim_cfg = self.config.get('stimulation', {})
        if stim_cfg.get('background', 'DC') == "DC":
            print("Use DC background stimulation.")
        else:
            neuron_cfg = self.config['neuron']
            E_L = neuron_cfg['E_L']
            C_m = neuron_cfg['C_m']
            tau_m_e = self._neuron_value('tau_m', 'excitatory')
            tau_m_i = self._neuron_value('tau_m', 'inhibitory')
            V_th_e = self._neuron_value('V_th', 'excitatory')
            V_th_i = self._neuron_value('V_th', 'inhibitory')
            I_th_e = self._neuron_value('I_th', 'excitatory', None)
            I_th_i = self._neuron_value('I_th', 'inhibitory', None)
            I_e_e = self._neuron_value('I_e', 'excitatory', 0.0)
            I_e_i = self._neuron_value('I_e', 'inhibitory', 0.0)

            I_xE = (
                I_e_e
                if I_th_e is None
                else I_th_e * (V_th_e - E_L) / tau_m_e * C_m
            )
            I_xI = (
                I_e_i
                if I_th_i is None
                else I_th_i * (V_th_i - E_L) / tau_m_i * C_m
            )

            p_rate = 150
            # we target for an incoming rate of 100 spikes/s
            self.Populations.append([nest.Create("poisson_generator", params={"rate": p_rate})])
            self.Populations.append([nest.Create("poisson_generator", params={"rate": p_rate})])

            #calculate average current for PSC with 1 pA
            tau_syn_ex = self._neuron_value('tau_syn', 'excitatory')
            J_Poisson_E = 1000 * I_xE / (tau_syn_ex * p_rate)
            J_Poisson_I = 1000 * I_xI / (tau_syn_ex * p_rate)

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
        """ Connects the excitatory and inhibitory populations with each other in the EI-clustered scheme
        """
        network = self.config['network']
        neuron_cfg = self.config['neuron']
        pop_cfg = network['population']
        conn_cfg = network['connectivity']
        weight_cfg = network['weights']
        cluster_cfg = network['clusters']

        n_exc = pop_cfg['excitatory']
        n_inh = pop_cfg['inhibitory']
        total_units = n_exc + n_inh

        ps = np.asarray(conn_cfg['probabilities'], dtype=float)
        js = np.asarray(weight_cfg['matrix'], dtype=float)
        scaling = weight_cfg.get('scale', 1.0)
        modifiers = weight_cfg['modifiers']
        ge = modifiers['ge']
        gi = modifiers['gi']
        gie = modifiers['gie']

        q_count = cluster_cfg['count']
        jplus = np.asarray(cluster_cfg['jplus'], dtype=float)

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
                'neuron_type': neuron_cfg['model'],
                'tau_syn_ex': self._neuron_value('tau_syn', 'excitatory'),
                'tau_syn_in': self._neuron_value('tau_syn', 'inhibitory'),
            }
            js = ClusterHelper.calc_js(calc_params)
        js = np.asarray(js, dtype=float) * scaling

        if q_count > 1:
            jminus = (q_count - jplus) / float(q_count - 1)
        else:
            jplus = np.ones((2, 2))
            jminus = np.ones((2, 2))

        j_ee = js[0, 0] / np.sqrt(total_units)
        if conn_cfg.get('fixed_indegree'):
            K_EE = int(ps[0, 0] * n_exc / q_count)
            print('K_EE: ', K_EE)
            conn_params_EE = {'rule': 'fixed_indegree', 'indegree': K_EE, 'allow_autapses': False,
                              'allow_multapses': False}
        else:
            conn_params_EE = {'rule': 'pairwise_bernoulli', 'p': ps[0, 0], 'allow_autapses': False,
                              'allow_multapses': False}
        for i, pre in enumerate(self.Populations[0]):
            for j, post in enumerate(self.Populations[0]):
                weight = jplus[0, 0] if i == j else jminus[0, 0]
                nest.Connect(pre, post, conn_params_EE,
                             syn_spec={"weight": weight * j_ee, "delay": delay})

        j_ei = js[0, 1] / np.sqrt(total_units)
        if conn_cfg.get('fixed_indegree'):
            K_EI = int(ps[0, 1] * n_inh / q_count)
            print('K_EI: ', K_EI)
            conn_params_EI = {'rule': 'fixed_indegree', 'indegree': K_EI, 'allow_autapses': False,
                              'allow_multapses': False}
        else:
            conn_params_EI = {'rule': 'pairwise_bernoulli', 'p': ps[0, 1], 'allow_autapses': False,
                              'allow_multapses': False}
        for i, pre in enumerate(self.Populations[1]):
            for j, post in enumerate(self.Populations[0]):
                weight = jplus[0, 1] if i == j else jminus[0, 1]
                nest.Connect(pre, post, conn_params_EI,
                             syn_spec={"weight": weight * j_ei, "delay": delay})

        j_ie = js[1, 0] / np.sqrt(total_units)
        if conn_cfg.get('fixed_indegree'):
            K_IE = int(ps[1, 0] * n_exc / q_count)
            print('K_IE: ', K_IE)
            conn_params_IE = {'rule': 'fixed_indegree', 'indegree': K_IE, 'allow_autapses': False,
                              'allow_multapses': False}
        else:
            conn_params_IE = {'rule': 'pairwise_bernoulli', 'p': ps[1, 0], 'allow_autapses': False,
                              'allow_multapses': False}
        for i, pre in enumerate(self.Populations[0]):
            for j, post in enumerate(self.Populations[1]):
                weight = jplus[1, 0] if i == j else jminus[1, 0]
                nest.Connect(pre, post, conn_params_IE,
                             syn_spec={"weight": weight * j_ie, "delay": delay})

        j_ii = js[1, 1] / np.sqrt(total_units)
        if conn_cfg.get('fixed_indegree'):
            K_II = int(ps[1, 1] * n_inh / q_count)
            print('K_II: ', K_II)
            conn_params_II = {'rule': 'fixed_indegree', 'indegree': K_II, 'allow_autapses': False,
                              'allow_multapses': False}
        else:
            conn_params_II = {'rule': 'pairwise_bernoulli', 'p': ps[1, 1], 'allow_autapses': False,
                              'allow_multapses': False}
        for i, pre in enumerate(self.Populations[1]):
            for j, post in enumerate(self.Populations[1]):
                weight = jplus[1, 1] if i == j else jminus[1, 1]
                nest.Connect(pre, post, conn_params_II,
                             syn_spec={"weight": weight * j_ii, "delay": delay})

    def create_stimulation(self):
        """
        Creates a current source as stimulation of the specified cluster/s.
        """
        stim_cfg = self.config.get('stimulation', {})
        warmup = self.config['simulation']['warmup']
        self.Currentsources = []
        clusters = stim_cfg.get('clusters')
        multi_cfg = stim_cfg.get('multi', {})

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
                amplitude_values.append(0.)
            self.Currentsources = [nest.Create('step_current_generator')]
            for stim_cluster in clusters:
                nest.Connect(self.Currentsources[0], self.Populations[0][stim_cluster])
            nest.SetStatus(self.Currentsources[0],
                           {'amplitude_times': amplitude_times, 'amplitude_values': amplitude_values})

        elif multi_cfg and multi_cfg.get('clusters') is not None:
            print('stimulating multi stim ...')
            clusters_list = multi_cfg.get('clusters', [])
            amps_list = multi_cfg.get('amps', [])
            times_list = multi_cfg.get('times', [])
            delays = multi_cfg.get('delay')
            if delays is not None and not isinstance(delays, (list, tuple)):
                delays = [delays]

            for stim_clusters, amplitudes, times in zip(clusters_list, amps_list, times_list):
                self.Currentsources.append(nest.Create('step_current_generator'))
                warm_times = [t + warmup for t in times[1:]]
                nest.SetStatus(self.Currentsources[-1], {'amplitude_times': warm_times,
                                                         'amplitude_values': amplitudes[1:]})
                stim_units = []
                if delays:
                    if len(delays) == 1:
                        syn_dict = {"delay": nest.random.uniform(min=0.1, max=delays[0])}
                    else:
                        syn_dict = {"delay": nest.random.uniform(min=delays[0], max=delays[1])}
                else:
                    syn_dict = {}
                for stim_cluster in stim_clusters:
                    nest.Connect(self.Currentsources[-1],
                                 self.Populations[0][stim_cluster],
                                 syn_spec=syn_dict)


    def create_recording_devices(self):
        """
        Creates a spike recorder connected to all neuron populations created by create_populations
        """
        self.RecordingDevices = [nest.Create("spike_recorder")]
        self.RecordingDevices[0].record_to = "memory"

        all_units = self.Populations[0][0]
        for E_pop in self.Populations[0][1:]:
            all_units += E_pop
        for I_pop in self.Populations[1]:
            all_units += I_pop
        nest.Connect(all_units, self.RecordingDevices[0], "all_to_all")  # Spikerecorder

    def setup_network(self):
        """
        Initializes NEST and creates the network in NEST, ready to be simulated.
        nest.Prepare is executed in this function.
        """
        self.setup_nest()
        self.create_populations()
        self.connect()
        self.create_connect_Poisson()
        self.create_recording_devices()
        self.create_stimulation()

    def simulate(self):
        """
        Simulates network for a period of warmup+simtime
        """
        sim_cfg = self.config['simulation']
        nest.Simulate(sim_cfg['warmup'] + sim_cfg['duration'])

    def get_recordings(self):
        """
        Extracts spikes form the Spikerecorder connected to all populations created in create_populations.
        Cuts the warmup period away and sets time relative to end of warmup.
        Ids 1:N_E correspond to excitatory neurons, N_E+1:N_E+N_I correspond to inhibitory neurons.

        Returns:
            spiketimes (np.array): Row 0: spiketimes, Row 1: neuron ID.
        """
        events = nest.GetStatus(self.RecordingDevices[0], 'events')[0]
        # convert them to the format accepted by spiketools
        spiketimes = np.append(events['times'][None, :], events['senders'][None, :], axis=0)
        spiketimes[1] -= 1
        # remove the pre warmup spikes
        warmup = self.config['simulation']['warmup']
        spiketimes = spiketimes[:, spiketimes[0] >= warmup]
        spiketimes[0] -= warmup
        return spiketimes

    def get_parameter(self):
        """
        Return:
            parameters (dict): Dictionary with all parameters for the simulation / network creation.
        """
        return self.config

    def create_and_simulate(self):
        """
        Creates the EI-clustered network and simulates it with the parameters supplied in the object creation.

        Returns:
            spiketimes (np.array):  Row 0: spiketimes, Row 1: neuron ID.
                                    Ids 1:N_E correspond to excitatory neurons,
                                    N_E+1:N_E+N_I correspond to inhibitory neurons.
        """
        self.setup_network()
        self.simulate()
        return self.get_recordings()


if __name__ == "__main__":
    from config import load_config
    import matplotlib.pyplot as plt

    default = load_config()
    overrides = {
        'simulation': {'n_jobs': 4, 'warmup': 500.0, 'duration': 1200.0},
        'stimulation': {
            'clusters': [3],
            'amplitude': 2.0,
            'starts': [600.0],
            'ends': [1000.0],
        },
    }

    EI_cluster = ClusteredNetworkNEST(default, overrides)
    spikes = EI_cluster.create_and_simulate()
    print(EI_cluster.get_parameter())
    plt.figure()
    plt.plot(spikes[0, :], spikes[1, :], '.')
    plt.savefig('NEST.png')
