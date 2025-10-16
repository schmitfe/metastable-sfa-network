from __future__ import annotations

from typing import Any, Mapping

from .. import GeneralHelper


class ClusteredNetworkBase:
    """
    Base object with basic initialisation and method for firing rate estimation.
    """

    def __init__(
        self,
        default_config: Mapping[str, Any],
        overrides: Mapping[str, Any] | None,
    ) -> None:
        self.config = GeneralHelper.mergeParams(dict(overrides or {}), default_config)

    def get_firing_rates(self, spiketimes=None):
        """
        Calculates the firing rates of all excitatory neurons and the firing rates of all inhibitory neurons.
        """
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

    def get_recordings(self):
        """
        Placeholder to be filled in children.
        """
        return []
