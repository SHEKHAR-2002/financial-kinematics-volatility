from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config


def download_yahoo_assets(
    assets: list[str],
    start: str,
    end: str | None,
    output_dir: str | Path,
) -> None:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install the download extra first: pip install '.[download]'") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        data = yf.download(asset, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            raise RuntimeError(f"No data returned for {asset}.")
        data.reset_index().to_csv(out / f"{asset}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily Indian equity data from Yahoo.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    paths = config.get("paths", {})
    download_yahoo_assets(
        assets=data_cfg.get("assets", ["RELIANCE.NS"]),
        start=data_cfg.get("start_date", "2008-01-01"),
        end=data_cfg.get("end_date"),
        output_dir=paths.get("raw_data_dir", "data/raw"),
    )


if __name__ == "__main__":
    main()
