"""Training-only statistics; raw returns remain untouched for accounting."""

from dataclasses import dataclass
import numpy as np
from .settings import write_json


@dataclass
class FeatureScaler:
    return_mean: np.ndarray
    return_scale: np.ndarray
    macd_mean: np.ndarray
    macd_scale: np.ndarray
    macro_mean: np.ndarray
    macro_scale: np.ndarray
    fit_end: str

    @classmethod
    def fit(cls, data, end):
        if not 1 <= end < len(data.dates):
            raise ValueError("Scaler fit end outside data")
        args = []
        for values in (data.returns, data.macd, data.macro):
            x = values[: end + 1]
            args.extend([x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)])
        return cls(*args, str(data.dates[end])[:10])

    def transform(self, name, values):
        return np.clip(
            (values - getattr(self, name + "_mean")) / getattr(self, name + "_scale"),
            -10,
            10,
        )

    def save(self, path):
        write_json(
            path,
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in vars(self).items()
            },
        )

    @classmethod
    def load(cls, path):
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            **{
                key: value if key == "fit_end" else np.array(value)
                for key, value in raw.items()
            }
        )
