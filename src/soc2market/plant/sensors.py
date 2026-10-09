"""
Measurement chain between the plant and the estimators.

Offset and gain errors on the current are the dominant source of coulomb-counting drift;
voltage and temperature are affected by noise and quantisation. Systematic errors are fixed
for a run, random noise is drawn from the generator passed by the caller.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorSpec:
    current_offset_a: float = 0.2
    current_gain_error: float = 0.002
    current_noise_a: float = 0.5
    voltage_noise_v: float = 0.002
    voltage_resolution_v: float = 0.001
    temperature_noise_c: float = 0.1
    temperature_resolution_c: float = 0.1
    current_resolution_a: float = 0.0

    @classmethod
    def ideal(cls):
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def matched_to(cls, reference, resolution_v=1e-4, resolution_a=1e-4):
        """
        Measurement chain matched to the acquisition of the estimators' training data: the
        resolution of the laboratory records (0.1 mV, 0.1 mA) and random noise estimated from a
        reference record, without offset or gain errors (invisible in those records, because their
        reference SOC is counted from the same measured current).
        """
        return cls(current_offset_a=0.0, current_gain_error=0.0,
                   current_noise_a=robust_noise(reference["Current_inv"].to_numpy(float)),
                   voltage_noise_v=robust_noise(reference["Voltage"].to_numpy(float)),
                   voltage_resolution_v=resolution_v, current_resolution_a=resolution_a,
                   temperature_noise_c=0.0, temperature_resolution_c=0.1)


def robust_noise(x):
    """
    White-noise standard deviation from first differences, robust to the steps of the signal:
    sigma = 1.4826 * MAD(dx) / sqrt(2).
    """
    dx = np.diff(np.asarray(x, float))
    dx = dx[np.isfinite(dx)]
    mad = np.median(np.abs(dx - np.median(dx)))
    return float(1.4826 * mad / np.sqrt(2.0))


def _quantise(x, resolution):
    return np.round(x / resolution) * resolution if resolution > 0 else x


def measure(result, spec, rng):
    """
    Turn a plant SimulationResult into the table the estimators read, with the column names
    of the Huawei data: Current_inv (A, positive charging), Voltage (V), Temp (C) and the
    true SOC in percent (SOC_true), which the estimators never see as an input.
    """
    n = result.current_a.size

    def noise(sigma):
        return rng.normal(0.0, sigma, n) if sigma > 0 else 0.0

    current = (result.current_a * (1.0 + spec.current_gain_error) + spec.current_offset_a
               + noise(spec.current_noise_a))
    voltage = result.voltage_v + noise(spec.voltage_noise_v)
    temp = result.temp_c + noise(spec.temperature_noise_c)
    return pd.DataFrame({
        "Current_inv": _quantise(current, spec.current_resolution_a),
        "Voltage": _quantise(voltage, spec.voltage_resolution_v),
        "Temp": _quantise(temp, spec.temperature_resolution_c),
        "SOC_true": result.soc * 100.0,
    })
