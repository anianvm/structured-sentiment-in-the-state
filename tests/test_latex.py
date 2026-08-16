"""The generated LaTeX bodies must say what the chapter says.

These tests pin the conventions that a reader of the PDF depends on and that
a CSV cannot express: how a negative number is typeset, when a cell is
blank because the quantity does not apply, and which rows carry a verdict.
"""

import pandas as pd
import pytest

from src.evaluation import latex


def test_negatives_are_typeset_in_math_mode():
    """A bare hyphen sets as a dash, not a minus, so negatives need $...$."""
    assert latex.num(1.234, 2) == "1.23"
    assert latex.num(-1.234, 2) == "$-1.23$"


def test_bold_negatives_use_boldmath():
    r"""\textbf around $-1.2$ emboldens nothing, so the spelling differs."""
    assert latex.num(2.4, 1, bold=True) == "\\textbf{2.4}"
    assert latex.num(-0.13, 2, bold=True) == "{\\boldmath$-0.13$}"


def test_missing_values_render_as_an_em_dash():
    assert latex.num(None, 2) == "---"
    assert latex.num(float("nan"), 2) == "---"


def _contrasts(hypothesis: str, supported: bool = True) -> dict:
    row = {
        "algo": "SAC", "eta": 1.0, "hypothesis": hypothesis, "arm": "M2",
        "baseline": "M1", "difference": 0.07, "ci_low": -0.20, "ci_high": 0.34,
        "p_one_sided": 0.3114, "p_holm": 0.3114, "seed_band": 0.19,
        "exceeds_seed_band": False, "supported": supported,
        "loo_sign_stable": False, "regime_robust": False,
    }
    frame = pd.DataFrame([row])
    return {"PPO": frame, "SAC": frame}


def test_supplementary_contrast_prints_no_holm_and_no_verdict():
    """M2-M1 sits outside every family: no multiplicity applies to it and no
    verdict is drawn from it. The CSV passes p_holm through unadjusted, and
    printing that would imply a family membership the design denies."""
    body = latex.contrasts_h1(_contrasts("SUPP"))
    row = [ln for ln in body.splitlines() if ln.startswith("M2")][0]
    cells = [c.strip() for c in row.rstrip("\\").split("&")]
    assert cells[3] == "$0.311$"      # the one-sided p is still reported
    assert cells[4] == "---"          # Holm is not
    assert cells[7] == "---"          # nor is a verdict


def test_family_contrast_does_print_its_holm_value():
    body = latex.contrasts_h1(_contrasts("H1", supported=False))
    row = [ln for ln in body.splitlines() if ln.startswith("M2")][0]
    cells = [c.strip() for c in row.rstrip("\\").split("&")]
    assert cells[4] == "$0.311$"
    assert cells[7] == "no"


def test_panels_are_emitted_for_both_algorithms():
    body = latex.contrasts_h1(_contrasts("H1", supported=False))
    assert "Panel A: PPO" in body and "Panel B: SAC" in body
    # the two panels are separated by a rule, which the fragment owns
    assert body.count("\\midrule") == 1


def test_h2_body_pairs_every_value_row_with_a_dispersion_row():
    """The SD sits under the cash share it qualifies; one row each, always,
    so a reader never has to guess which level a bracket belongs to."""
    columns = {"arm": latex.ARMS}
    for measure in ("cash", "vol", "mdd"):
        for eta in (1, 3, 7):
            columns[f"{measure}_eta{eta}"] = [0.05] * 6
    for eta in (1, 3, 7):
        columns[f"cash_sd_eta{eta}"] = [0.028] * 6
    for flag in ("cash_monotone_up", "vol_monotone_down", "mdd_monotone_down"):
        columns[flag] = [True] * 6
    frame = pd.DataFrame(columns)

    body = latex.h2_separation({"PPO": frame, "SAC": frame})
    value_rows = [ln for ln in body.splitlines() if ln[:2] in
                  {f"M{i}" for i in range(1, 7)}]
    sd_rows = [ln for ln in body.splitlines() if "scriptsize" in ln]
    assert len(value_rows) == len(sd_rows) == 12      # six arms, two panels
    assert sd_rows[0].count("{\\scriptsize (2.8)}") == 3


@pytest.mark.parametrize("name", [
    "res-perf-pooled", "res-perf-window", "res-h1", "res-h3", "res-h4",
    "res-regime", "res-h2", "res-h2-dist", "res-h5-r", "res-h5-dr",
    "res-behaviour", "resapp-seeds", "resapp-seed-cash",
])
def test_every_committed_body_is_present_and_non_empty(name):
    """The chapter \\inputs these by name; a missing file is a broken build."""
    from src import config

    path = config.RESULTS_DIR / "tables" / "latex" / f"{name}.tex"
    if not path.exists():
        pytest.skip("tables not built in this checkout; run 06_evaluate.py")
    text = path.read_text().strip()
    # a body ends on a row terminator, optionally carrying extra row spacing
    assert text and (text.endswith("\\\\") or text.rstrip("]").rstrip(
        "0123456789.pt").endswith("\\\\["))
