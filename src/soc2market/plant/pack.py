"""Scaling between a grid-scale pack and the representative cell simulated by the plant."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackSpec:
    """A pack of identical cells; cell-to-cell imbalance is neglected (stated as a limitation)."""

    power_mw: float
    energy_mwh: float
    cell_capacity_ah: float = 280.0
    cell_nominal_v: float = 3.2

    @property
    def n_cells(self):
        return self.energy_mwh * 1e6 / (self.cell_capacity_ah * self.cell_nominal_v)

    def cell_power_w(self, pack_power_mw):
        """Pack power (MW, positive when charging) to the power of one cell (W)."""
        return pack_power_mw * 1e6 / self.n_cells

    def pack_energy_mwh(self, cell_energy_wh):
        return cell_energy_wh * self.n_cells / 1e6
