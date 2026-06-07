from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.features import FeatureConfig, add_financial_kinematics_features
from src.utils.config import load_config


def load_asset_csv(path: str | Path, asset: str | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    if asset is None:
        asset = csv_path.stem
    frame["asset"] = asset
    return frame


def process_asset_frame(
    raw: pd.DataFrame,
    asset: str,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    frame = raw.copy()
    frame["asset"] = asset
    return add_financial_kinematics_features(frame, feature_config)


def process_raw_directory(
    raw_dir: str | Path,
    output_dir: str | Path,
    assets: list[str] | None = None,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_path.glob("*.csv"))
    if assets:
        wanted = set(assets)
        csv_files = [path for path in csv_files if path.stem in wanted]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}.")

    processed = []
    for csv_file in csv_files:
        asset = csv_file.stem
        raw = load_asset_csv(csv_file, asset=asset)
        features = process_asset_frame(raw, asset=asset, feature_config=feature_config)
        features.to_csv(out_path / f"{asset}_features.csv", index=False)
        processed.append(features)

    combined = pd.concat(processed, ignore_index=True)
    combined.to_csv(out_path / "features_all.csv", index=False)
    return combined


def feature_config_from_dict(config: dict) -> FeatureConfig:
    features = config.get("features", {})
    return FeatureConfig(
        horizon=int(config.get("target", {}).get("horizon", features.get("horizon", 30))),
        rolling_vol_windows=tuple(features.get("rolling_vol_windows", (7, 14, 30))),
        momentum_windows=tuple(features.get("momentum_windows", (7, 30, 90))),
        drawdown_window=int(features.get("drawdown_window", 90)),
        true_range_windows=tuple(features.get("true_range_windows", (7, 30))),
        volume_z_window=int(features.get("volume_z_window", 30)),
        include_jerk=bool(features.get("include_jerk", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe processed feature CSVs.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    paths = config.get("paths", {})
    assets = data_cfg.get("assets")
    process_raw_directory(
        raw_dir=paths.get("raw_data_dir", "data/raw"),
        output_dir=paths.get("processed_data_dir", "data/processed"),
        assets=assets,
        feature_config=feature_config_from_dict(config),
    )


if __name__ == "__main__":
    main()
