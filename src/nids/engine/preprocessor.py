"""Online preprocessor (M4 inference engine, plan §3.3; FR5, FR6, Fig. 3.7 process 2.3).

Applies the state fitted offline on the training partition (artefacts/preprocessor.json)
to one record using NumPy only: missing -> training median, log(1+x) on the log
columns, then standardisation with the training mean and scale. The arithmetic is the
same as nids.prep.prepare_data.transform_offline; a unit test checks that they agree
(plan §3.3 "Done when", test UT1).

The state is plain JSON, never a pickle (NFR6).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Preprocessor:
    def __init__(self, path: str | Path):
        p = json.loads(Path(path).read_text())
        self.features: list[str] = p["features"]
        self.median = np.array(p["median"], dtype=np.float64)
        self.log_mask = np.array(p["log_mask"], dtype=bool)
        self.mean = np.array(p["mean"], dtype=np.float64)
        self.scale = np.array(p["scale"], dtype=np.float64)
        n = len(self.features)
        if not all(a.shape == (n,) for a in (self.median, self.log_mask, self.mean, self.scale)):
            raise ValueError("preprocessor.json arrays do not match the feature list")

    def transform(self, values) -> np.ndarray:
        """Scale one record (values in feature_list order). Raises ValueError on bad input (FR5)."""
        try:
            x = np.array(values, dtype=np.float64)  # copy: never mutate the caller's data
        except (TypeError, ValueError) as e:
            raise ValueError(f"non-numeric feature value: {e}") from None
        if x.shape != (len(self.features),):
            raise ValueError(f"expected {len(self.features)} features, got shape {x.shape}")
        x[~np.isfinite(x)] = np.nan                      # inf -> missing (step 4)
        x = np.where(np.isnan(x), self.median, x)        # training medians (step 10c)
        x[self.log_mask] = np.log1p(np.clip(x[self.log_mask], 0, None))  # step 10d
        return ((x - self.mean) / self.scale).astype(np.float32)        # step 10e
