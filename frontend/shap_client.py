"""Read saved real model explanations over HTTP; never use fixtures."""

from collections.abc import Mapping

import httpx
import numpy as np
from frontend.api_client import get_api_base_url, APIError


def load_model_shap(model):
    try:
        response = httpx.get(
            get_api_base_url() + "/performance/model-shap",
            params={"model": model},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if (
            not isinstance(data, Mapping)
            or data["model_type"] != model
            or not isinstance(data["assets"], Mapping)
            or not data["assets"]
            or not data["features"]
        ):
            raise ValueError("Invalid explanation")
        for asset in data["assets"].values():
            if not asset["decisions"]:
                raise ValueError("No decisions")
            for row in asset["decisions"]:
                if len(row["shap_values"]) != len(data["features"]) or len(
                    row["feature_values_raw"]
                ) != len(data["features"]):
                    raise ValueError("Invalid features")
                if not np.isfinite(
                    [
                        *row["shap_values"],
                        *row["feature_values_raw"],
                        row["base_value"],
                        row["final_value"],
                    ]
                ).all():
                    raise ValueError("Nonfinite values")
                if (
                    abs(
                        row["base_value"] + sum(row["shap_values"]) - row["final_value"]
                    )
                    > 2e-5
                ):
                    raise ValueError("SHAP additivity mismatch")
        return data
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise APIError(
            "실제 SHAP 결과를 불러오지 못했습니다. API 실행 및 해당 모델의 해석 결과 설치 여부를 확인하세요."
        ) from None
