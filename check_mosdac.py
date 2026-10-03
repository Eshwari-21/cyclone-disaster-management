import h5py
import numpy as np
import glob
import os

files = sorted(glob.glob("data/raw/mosdac/*.h5"))

print("File | max mm/hr | nonzero pixels")
print("-" * 70)

for path in files:
    with h5py.File(path, "r") as f:
        x = np.squeeze(f["IMR"][:])
        lat = f["latitude"][:]
        lon = f["longitude"][:]

        ii = np.where(
            (lat >= 16.672423) &
            (lat <= 19.166914)
        )[0]

        jj = np.where(
            (lon >= 81.903123) &
            (lon <= 84.765077)
        )[0]

        sub = x[np.ix_(ii, jj)]

        print(
            os.path.basename(path),
            "| max =", round(float(np.nanmax(sub)), 3),
            "| nonzero =", int(np.count_nonzero(sub > 0))
        )