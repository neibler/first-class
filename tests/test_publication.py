"""Publication metadata must not leak the machine used for collection."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


def test_macro_collection_publishes_filename_without_personal_path(
    monkeypatch, tmp_path
):
    source = (
        Path(__file__).resolve().parents[1]
        / "data_process/src/preprocessing/02_collect_macro.py"
    )
    spec = importlib.util.spec_from_file_location("macro_publication_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_dotenv", lambda *args: None)
    monkeypatch.setenv("FRED_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr(module, "version", lambda name: "test")
    session = MagicMock()
    monkeypatch.setattr(module.requests, "Session", lambda: session)

    def rows(series, variable):
        return pd.DataFrame(
            {
                "series_id": [series],
                "variable_name": [variable],
                "observation_date": [pd.Timestamp("2025-04-01")],
                "realtime_start": [pd.Timestamp("2025-04-02")],
                "release_date": [pd.Timestamp("2025-04-02")],
                "value": [1.0],
            }
        )

    monkeypatch.setattr(
        module,
        "collect_fred",
        lambda session, series, variable, *args: rows(series, variable),
    )
    monkeypatch.setattr(module, "read_pmi", lambda *args: rows("ISM_PMI", "ism_pmi"))
    private_input = tmp_path / "personal-owner" / "ISM_PMI_cleaned.xlsx"
    monkeypatch.setattr("sys.argv", [str(source), "--pmi-file", str(private_input)])
    module.main()
    output = pd.read_parquet(tmp_path / "raw/macro_observations.parquet")
    assert output.attrs["pmi_file"] == "ISM_PMI_cleaned.xlsx"
    assert "personal-owner" not in str(output.attrs)
    assert (
        b"personal-owner"
        not in (tmp_path / "raw/macro_observations.parquet").read_bytes()
    )
    assert len(output) == len(module.SERIES) + 1
    session.__enter__.return_value.get.assert_not_called()
