"""_find_run_files must pick the per-day series, not a per-seed summary.

A walk-forward year directory holds several CSVs whose headers contain the
substring "log_return": the daily series, the per-seed validation summary,
and one training curve per seed. Matching by substring picked whichever
sorted last -- validation_seeds.csv -- and evaluation died on its missing
date column. These pin the exact-column contract.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.report import _find_run_files


def _year_dir(tmp_path):
    d = tmp_path / "2021"; d.mkdir()
    (d / "test_daily.csv").write_text("date,log_return,turnover,cash_weight\n2021-01-04,0.1,0.5,0.01\n")
    (d / "test_weights.csv").write_text("date,CASH,AAPL\n2021-01-04,0.01,0.99\n")
    (d / "validation_seeds.csv").write_text("seed,val_log_return,val_days\n0,0.15,252\n")
    for s in range(5):
        (d / f"curve_seed{s}.csv").write_text("steps,val_log_return\n100000,0.1\n")
    (d / "sb3_log_seed0.csv").write_text("train/n_updates,time/fps\n1,100\n")
    (d / "summary.json").write_text("{}")
    return d


def test_picks_the_daily_series_not_the_seed_summary(tmp_path):
    files = _find_run_files(_year_dir(tmp_path))
    assert files["daily"].name == "test_daily.csv"
    assert files["weights"].name == "test_weights.csv"
    assert files["summary"].name == "summary.json"


def test_daily_file_is_readable_as_a_date_indexed_frame(tmp_path):
    import pandas as pd
    files = _find_run_files(_year_dir(tmp_path))
    df = pd.read_csv(files["daily"], parse_dates=["date"], index_col="date")
    assert "log_return" in df.columns


def test_returns_none_when_no_daily_series_exists(tmp_path):
    d = tmp_path / "2021"; d.mkdir()
    (d / "validation_seeds.csv").write_text("seed,val_log_return\n0,0.1\n")
    assert _find_run_files(d) is None
