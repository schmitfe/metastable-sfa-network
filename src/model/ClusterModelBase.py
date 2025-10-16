from __future__ import annotations

import copy
from collections.abc import MutableMapping
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import GeneralHelper


PARAM_KEY_MAP: dict[str, Sequence[str]] = {
    'C_m': ('neurons', 'capacitance'),
    'Delta_V': ('neurons', 'adaptation', 'Delta_V'),
    'DistParams': ('synapse', 'distribution'),
    'E_L': ('neurons', 'rest_potential'),
    'I_th_E': ('neurons', 'target_current', 'excitatory'),
    'I_th_I': ('neurons', 'target_current', 'inhibitory'),
    'I_xE': ('neurons', 'background_current', 'excitatory'),
    'I_xI': ('neurons', 'background_current', 'inhibitory'),
    'N_E': ('network', 'population', 'excitatory'),
    'N_I': ('network', 'population', 'inhibitory'),
    'Q': ('network', 'clusters', 'count'),
    'V_m': ('neurons', 'initial_membrane_state'),
    'V_r': ('neurons', 'reset_potential'),
    'V_th_E': ('neurons', 'threshold', 'excitatory'),
    'V_th_I': ('neurons', 'threshold', 'inhibitory'),
    'background_stim': ('stimulation', 'background'),
    'delay': ('network', 'connectivity', 'delay'),
    'delta_': ('neurons', 'noise', 'delta'),
    'delta_I_xE': ('neurons', 'background_current', 'delta', 'excitatory'),
    'delta_I_xI': ('neurons', 'background_current', 'delta', 'inhibitory'),
    'eps': ('constants', 'eps'),
    'fixed_indegree': ('network', 'connectivity', 'fixed_indegree'),
    'ge': ('network', 'weights', 'modifiers', 'ge'),
    'gi': ('network', 'weights', 'modifiers', 'gi'),
    'gie': ('network', 'weights', 'modifiers', 'gie'),
    'jplus': ('network', 'clusters', 'jplus'),
    'js': ('network', 'weights', 'matrix'),
    'lambda_0': ('neurons', 'adaptation', 'lambda_0'),
    'multi_stim_amps': ('stimulation', 'multi', 'amps'),
    'multi_stim_clusters': ('stimulation', 'multi', 'clusters'),
    'multi_stim_times': ('stimulation', 'multi', 'times'),
    'neuron_type': ('neurons', 'type'),
    'n_jobs': ('simulation', 'n_jobs'),
    'ps': ('network', 'connectivity', 'probabilities'),
    'q_sfa': ('neurons', 'adaptation', 'q_sfa'),
    'q_stc': ('neurons', 'adaptation', 'q_stc'),
    'randseed': ('simulation', 'randseed'),
    'record_from': ('recording', 'target'),
    'record_voltage': ('recording', 'voltage'),
    'recording_interval': ('recording', 'interval'),
    'return_weights': ('simulation', 'return_weights'),
    'rho': ('neurons', 'noise', 'rho'),
    's': ('network', 'weights', 'scale'),
    'simtime': ('simulation', 'duration'),
    'stim_amp': ('stimulation', 'amplitude'),
    'stim_clusters': ('stimulation', 'clusters'),
    'stim_clusters_delay': ('stimulation', 'multi', 'delay'),
    'stim_ends': ('stimulation', 'ends'),
    'stim_starts': ('stimulation', 'starts'),
    'syn_params': ('synapse', 'plasticity'),
    't_ref': ('neurons', 'refractory_period'),
    'tau_E': ('neurons', 'membrane_time_constant', 'excitatory'),
    'tau_I': ('neurons', 'membrane_time_constant', 'inhibitory'),
    'tau_sfa': ('neurons', 'adaptation', 'tau_sfa'),
    'tau_stc': ('neurons', 'adaptation', 'tau_stc'),
    'tau_syn_ex': ('neurons', 'synapse_time_constant', 'excitatory'),
    'tau_syn_in': ('neurons', 'synapse_time_constant', 'inhibitory'),
    'dt': ('simulation', 'dt'),
    'warmup': ('simulation', 'warmup'),
}

ARRAY_PATHS: tuple[Sequence[str], ...] = (
    PARAM_KEY_MAP['ps'],
    PARAM_KEY_MAP['js'],
    PARAM_KEY_MAP['jplus'],
)


class ParamAccessor(MutableMapping[str, Any]):
    """
    Provides backward-compatible access to flattened parameter names while
    storing values inside the hierarchical configuration dictionary.
    """

    def __init__(
        self,
        config: dict[str, Any],
        mapping: Mapping[str, Sequence[str]],
        extras: dict[str, Any],
    ) -> None:
        self._config = config
        self._mapping = mapping
        self._extras = extras

    def __getitem__(self, key: str) -> Any:
        if key in self._mapping:
            try:
                return _get_path(self._config, self._mapping[key])
            except KeyError as err:
                raise KeyError(key) from err
        if key in self._extras:
            return self._extras[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._mapping:
            _set_path(self._config, self._mapping[key], value)
        else:
            self._extras[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._mapping:
            path = self._mapping[key]
            parent = path[:-1]
            leaf = path[-1]
            try:
                container = _get_path(self._config, parent) if parent else self._config
            except KeyError as err:
                raise KeyError(key) from err
            if leaf in container:
                del container[leaf]
            else:
                raise KeyError(key)
        elif key in self._extras:
            del self._extras[key]
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterable[str]:
        for key, path in self._mapping.items():
            try:
                _get_path(self._config, path)
            except KeyError:
                continue
            yield key
        yield from self._extras.keys()

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


class ClusteredNetworkBase:
    """
    Base object with basic initialisation and method for firing rate estimation.
    """

    def __init__(
        self,
        defaultValues: Mapping[str, Any],
        parameters: Mapping[str, Any] | None,
    ) -> None:
        """
        Parameters
        ----------
        defaultValues:
            Hierarchical configuration dictionary (e.g. loaded from YAML).
        parameters:
            Optional overrides for configuration values. Accepts either the new
            hierarchical structure or the previous flat key/value pairs.
        """
        params_copy = copy.deepcopy(parameters or {})
        overrides, extras = _normalize_parameters(params_copy)
        self.config = GeneralHelper.mergeParams(overrides, defaultValues)
        _coerce_config_types(self.config)
        self._extra_params = extras
        self.params: MutableMapping[str, Any] = ParamAccessor(
            self.config, PARAM_KEY_MAP, self._extra_params
        )

    def get_firing_rates(self, spiketimes=None):
        """
        Calculates the firing rates of all excitatory neurons and the firing rates of all inhibitory neurons.
        """
        if spiketimes is None:
            spiketimes = self.get_recordings()
        n_exc = self.params['N_E']
        n_inh = self.params['N_I']
        simtime = float(self.params['simtime'])
        e_count = spiketimes[:, spiketimes[1] < n_exc].shape[1]
        i_count = spiketimes[:, spiketimes[1] >= n_exc].shape[1]
        e_rate = e_count / float(n_exc) / simtime * 1000.0
        i_rate = i_count / float(n_inh) / simtime * 1000.0
        return e_rate, i_rate

    def get_recordings(self):
        """
        Placeholder to be filled in children.
        """
        return []


def _normalize_parameters(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    nested: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, value in parameters.items():
        if key in PARAM_KEY_MAP:
            _set_path(nested, PARAM_KEY_MAP[key], value)
        elif isinstance(value, Mapping):
            existing = nested.get(key, {})
            merged = GeneralHelper.nested_update(existing, value)
            nested[key] = merged
        else:
            extras[key] = copy.deepcopy(value)
    return nested, extras


def _get_path(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = config
    for segment in path:
        if not isinstance(node, Mapping) or segment not in node:
            raise KeyError('.'.join(path))
        node = node[segment]
    return node


def _set_path(config: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    node: MutableMapping[str, Any] = config
    for segment in path[:-1]:
        next_node = node.get(segment)
        if not isinstance(next_node, MutableMapping):
            next_node = {}
            node[segment] = next_node
        node = next_node
    node[path[-1]] = copy.deepcopy(value)


def _coerce_config_types(config: dict[str, Any]) -> None:
    for path in ARRAY_PATHS:
        try:
            value = _get_path(config, path)
        except KeyError:
            continue
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            continue
        _set_path(config, path, np.asarray(value, dtype=float))
