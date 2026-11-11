"""Read-only links between saved explanations and their evaluation records."""

import json
import math
from copy import deepcopy

import pandas as pd


def attach_decisions(payload, bundle):
    result = deepcopy(payload)
    result["execution_context_status"] = "unavailable"
    result["weight_stage"] = "필터 적용 여부 미확인"
    try:
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        evaluation = json.loads(
            (bundle / "evaluation.json").read_text(encoding="utf-8")
        )
        if (
            not evaluation.get("completed")
            or evaluation["data_hash"] != payload["data_hash"]
        ):
            return result
        if evaluation["model_type"] != payload["model_type"]:
            return result
        history = pd.read_csv(bundle / "evaluation/ppo_history.csv")
        result["min_weight_threshold"] = config.get("min_weight_threshold")
        result["weight_stage"] = (
            "최소 비중 필터 설정이 없는 이전 학습본"
            if "min_weight_threshold" not in config
            else "필터 적용 여부는 아래 기록 일치 결과로 확인"
        )
        count = 0
        for ticker, asset in result["assets"].items():
            suffix = ticker.lower()
            for row in asset["decisions"]:
                matched = history[
                    (history.decision_date == row["date"])
                    & (history.execution_date == row["execution_date"])
                    & (history.run_id == payload["run_id"])
                ]
                if len(matched) != 1:
                    continue
                record = matched.iloc[0]
                # Only join an output that agrees with this saved evaluation.
                target = float(record[f"target_weight_{suffix}"])
                if not math.isclose(target, row["final_value"], abs_tol=2e-6):
                    continue
                previous = float(record[f"previous_weight_{suffix}"])
                current = float(record[f"current_weight_{suffix}"])
                if not all(
                    math.isfinite(v) and 0 <= v <= 1
                    for v in [previous, current, target]
                ):
                    continue
                change = current - previous
                traded = abs(change) > 1e-8
                row["execution_context"] = {
                    "previous_weight": previous,
                    "target_weight": target,
                    "current_weight": current,
                    "weight_change": change,
                    "traded": traded,
                    "direction": (
                        "확대"
                        if change > 1e-8
                        else "축소" if change < -1e-8 else "유지"
                    ),
                    "execution_date": row["execution_date"],
                }
                count += 1
        if count:
            result["execution_context_status"] = "matched"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return result
