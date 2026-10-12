"""
Convert the 2023 Continental Europe frequency csv into the binary daily matrix used by the
signal builder, and print a short quality report.
"""

import numpy as np

from soc2market.signals.frequency import FREQ_DIR, FrequencyArchive, convert_frequency_csv

YEAR = 2023
CSV = FREQ_DIR / "continental_europe_2023.csv"
OUT = FREQ_DIR / "2023.npz"


def main():
    dev, rows = convert_frequency_csv(CSV, OUT, YEAR)
    arch = FrequencyArchive.load(OUT)
    miss = arch.missing_share()
    valid = dev[~np.isnan(dev)]
    print(f"rows read: {rows}, days: {arch.n_days}, written to {OUT}")
    print(f"missing seconds over the year: {100 * np.isnan(dev).mean():.3f}%")
    print(f"days with more than 1% missing: {int((miss > 0.01).sum())}")
    for d in np.flatnonzero(miss > 0.01)[:15]:
        print(f"   day {d:3d}: {100 * miss[d]:.2f}% missing")
    print(f"deviation: mean {1e3 * valid.mean():.2f} mHz, std {1e3 * valid.std():.2f} mHz, "
          f"min {1e3 * valid.min():.0f} mHz, max {1e3 * valid.max():.0f} mHz")
    print(f"share of seconds outside the 10 mHz deadband: {100 * np.mean(np.abs(valid) > 0.010):.1f}%")
    print(f"share of seconds beyond full activation (200 mHz): {100 * np.mean(np.abs(valid) > 0.200):.4f}%")


if __name__ == "__main__":
    main()
