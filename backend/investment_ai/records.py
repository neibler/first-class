"""Long-form asset records with explicit execution-time weight semantics."""

import pandas as pd


def asset_rows(row, assets):
    for asset in assets:
        item = {key: row[key] for key in ["date", "run_id", "window_id"]}
        item["ticker"] = asset
        for key in [
            "previous_weight",
            "current_weight",
            "target_weight",
            "weight_change",
        ]:
            item[key] = row[f"{key}_{asset.lower()}"]
        if f"model_target_weight_{asset.lower()}" in row:
            item["model_target_weight"] = row[f"model_target_weight_{asset.lower()}"]
        for key in [
            "raw_target_weight",
            "filtered_target_weight",
            "decision_current_weight",
        ]:
            if f"{key}_{asset.lower()}" in row:
                item[key] = row[f"{key}_{asset.lower()}"]
        for key in [
            "training_step",
            "episode",
            "rebalance_reason",
            "portfolio_weight_distance",
            "max_risk_score",
        ]:
            if key in row:
                item[key] = row[key]
        yield item


def save_asset_history(history, assets, path):
    pd.DataFrame(
        [item for row in history.to_dict("records") for item in asset_rows(row, assets)]
    ).to_csv(path, index=False)
