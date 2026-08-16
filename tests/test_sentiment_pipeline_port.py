"""Tests for the frozen sentiment pipeline port (src/sentiment_pipeline/).

Two things are worth checking about frozen code: that every ported module still
imports cleanly (heavy scoring dependencies — transformers, openai,
google-cloud-bigquery — are lazy imports, so a plain import must work without
them), and that the committed outputs the experiments depend on are present
with the shape the methodology chapter documents (93,928 scored stock-days).
"""

import importlib

import pandas as pd
import pytest

from src import config

PORTED_MODULES = [
    "src.sentiment_pipeline.collect_gdelt",
    "src.sentiment_pipeline.build_headlines",
    "src.sentiment_pipeline.select_headlines",
    "src.sentiment_pipeline.score_llm",
    "src.sentiment_pipeline.score_llm_sector",
    "src.sentiment_pipeline.score_finbert",
    "src.sentiment_pipeline.compare_llm_finbert",
    "src.sentiment_pipeline.compare_sector_context",
]

# Row count fixed by the selection step: one headline per (ticker, day with
# news) over Feb 2015 - Jun 2026 (methodology, "News headline cleaning and
# selection").
N_SCORED = 93_928

EXPECTED_COLUMNS = {
    config.LLM_SCORES_FILE: [
        "date", "ticker", "headline",
        "direction", "magnitude", "horizon", "confidence",
    ],
    config.LLM_SECTOR_SCORES_FILE: [
        "date", "ticker", "sector", "headline",
        "direction", "magnitude", "horizon", "confidence",
    ],
    config.FINBERT_SCORES_FILE: [
        "date", "ticker", "headline",
        "p_pos", "p_neg", "p_neu", "finbert_score", "finbert_label",
    ],
}


@pytest.mark.parametrize("module_name", PORTED_MODULES)
def test_module_imports_without_heavy_deps(module_name):
    """Each ported module imports without transformers/openai/google.cloud."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_pure_helpers_behave_as_documented():
    """Cheap sanity check of the shared cleaning/anonymisation/parsing trio."""
    from src.sentiment_pipeline.score_llm import (
        anonymise, clean_headline, parse_scores,
    )
    # embedded date and wire-service ID are stripped, words survive
    cleaned = clean_headline("apple sued over batteries 2015 02 19 cm445826")
    assert cleaned == "apple sued over batteries"
    # the company alias disappears after anonymisation
    assert anonymise(cleaned, "AAPL") == "the company sued over batteries"
    # a valid score object round-trips; an out-of-range one is rejected
    good = '{"direction": 1, "magnitude": 3, "horizon": "days", "confidence": 4}'
    assert parse_scores(good) == {"direction": 1, "magnitude": 3,
                                  "horizon": "days", "confidence": 4}
    bad = '{"direction": 2, "magnitude": 3, "horizon": "days", "confidence": 4}'
    assert parse_scores(bad) is None


def test_committed_score_files_exist_with_expected_shape():
    """The frozen outputs are committed: right columns, 93,928 rows each."""
    for path, expected_cols in EXPECTED_COLUMNS.items():
        assert path.exists(), f"committed file missing: {path}"
        df = pd.read_csv(path)
        assert list(df.columns) == expected_cols, f"unexpected columns in {path}"
        assert len(df) == N_SCORED, (
            f"{path} has {len(df)} rows, expected {N_SCORED}"
        )


def test_committed_headline_file_exists_with_expected_columns():
    """The raw headline table behind the selection step is committed too."""
    assert config.HEADLINES_FILE.exists()
    head = pd.read_csv(config.HEADLINES_FILE, nrows=5)
    assert list(head.columns) == [
        "date", "ticker", "headline", "domain", "url", "gdelt_tone",
    ]
