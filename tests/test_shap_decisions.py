import json
from copy import deepcopy

import pandas as pd
import pytest
from backend.performance.shap_decisions import attach_decisions


@pytest.mark.parametrize(
    "previous,current,direction",
    [(0.1, 0.2, "확대"), (0.3, 0.2, "축소"), (0.2, 0.2, "유지")],
)
def test_trade_direction_independent_of_shap(tmp_path, previous, current, direction):
    payload = {
        "model_type": "td3",
        "data_hash": "same",
        "run_id": "test",
        "assets": {
            "GLD": {
                "decisions": [
                    {
                        "date": "2025-01-02",
                        "execution_date": "2025-01-03",
                        "final_value": 0.2,
                        "base_value": 0.05,
                        "shap_values": [0.2, -0.05],
                    }
                ]
            }
        },
    }
    original = deepcopy(payload)
    (tmp_path / "config.json").write_text(json.dumps({"min_weight_threshold": 0.03}))
    (tmp_path / "evaluation.json").write_text(
        json.dumps({"completed": True, "model_type": "td3", "data_hash": "same"})
    )
    (tmp_path / "evaluation").mkdir()
    frame = pd.DataFrame(
        [
            {
                "decision_date": "2025-01-02",
                "execution_date": "2025-01-03",
                "run_id": "test",
                "target_weight_gld": 0.2,
                "previous_weight_gld": previous,
                "current_weight_gld": current,
            }
        ]
    )
    path = tmp_path / "evaluation/ppo_history.csv"
    frame.to_csv(path, index=False)
    result = attach_decisions(payload, tmp_path)
    execution = result["assets"]["GLD"]["decisions"][0]["execution_context"]
    assert execution["direction"] == direction
    assert execution["traded"] == (previous != current)
    assert payload == original
    frame.loc[0, "target_weight_gld"] = 0.9
    frame.to_csv(path, index=False)
    assert (
        "execution_context"
        not in attach_decisions(payload, tmp_path)["assets"]["GLD"]["decisions"][0]
    )
    frame.loc[0, "target_weight_gld"] = 0.2
    frame.loc[0, "decision_date"] = "2025-01-01"
    frame.to_csv(path, index=False)
    assert (
        "execution_context"
        not in attach_decisions(payload, tmp_path)["assets"]["GLD"]["decisions"][0]
    )
