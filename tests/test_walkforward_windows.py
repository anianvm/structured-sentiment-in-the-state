"""The walk-forward window layout (§ Walk-Forward Design and Model Selection).

Pure date arithmetic — no market data, no network. ISO date strings compare
lexicographically in chronological order, so plain string comparisons below
are date comparisons.
"""

from src import config
from src.experiments.walkforward import make_windows


def test_six_windows_with_test_years_2021_to_2026():
    windows = make_windows()
    assert len(windows) == 6
    assert [w.test_year for w in windows] == [2021, 2022, 2023, 2024, 2025, 2026]


def test_train_span_is_five_calendar_years_ending_dec_31_before_val():
    for w in make_windows():
        first_train_year = int(w.train_start[:4])
        last_train_year = int(w.train_end[:4])
        assert last_train_year - first_train_year + 1 == 5
        assert w.train_start.endswith("-01-01")
        assert w.train_end.endswith("-12-31")
        # training ends Dec 31 of the year before the validation year
        assert last_train_year == int(w.val_start[:4]) - 1


def test_val_year_is_test_year_minus_one():
    for w in make_windows():
        assert w.val_start == f"{w.test_year - 1}-01-01"
        assert w.val_end == f"{w.test_year - 1}-12-31"
        assert w.test_start == f"{w.test_year}-01-01"


def test_test_period_never_overlaps_train_or_val():
    for w in make_windows():
        assert w.train_start <= w.train_end < w.val_start
        assert w.val_start <= w.val_end < w.test_start
        assert w.test_start <= w.test_end


def test_first_window_trains_from_2015():
    assert make_windows()[0].train_start.startswith("2015")


def test_last_test_period_ends_with_the_sample():
    assert make_windows()[-1].test_end == config.SAMPLE_END


# ---------------------------------------------------------------------------
# Window filtering (task-level parallelism)
# ---------------------------------------------------------------------------

def test_window_filter_selects_the_requested_years():
    """The filter used by run_walkforward(windows=...) picks exactly those years."""
    from src.experiments.walkforward import make_windows

    wanted = {2022, 2025}
    chosen = [w for w in make_windows() if w.test_year in wanted]
    assert [w.test_year for w in chosen] == [2022, 2025]
    # Filtering must not alter the windows themselves.
    full = {w.test_year: w for w in make_windows()}
    assert all(w == full[w.test_year] for w in chosen)


def test_every_window_writes_a_distinct_directory():
    """Task-level parallelism is only safe if no two windows share a path."""
    from src.experiments.walkforward import make_windows

    paths = [f"M4_eta3.0/{w.test_year}" for w in make_windows()]
    assert len(set(paths)) == len(paths)
