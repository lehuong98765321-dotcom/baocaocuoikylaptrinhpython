"""Automated acceptance tests for externally executed Stage 6 RFM outputs."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "input" / "online_retail_rfm_eligible.csv"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables" / "rfm"
FIGURES = ROOT / "outputs" / "figures" / "rfm"
EXPECTED_ROWS = 392_692
EXPECTED_CUSTOMERS = 4_338

CSV_OUTPUTS = [
    PROCESSED / "rfm_customers.csv",
    PROCESSED / "rfm_with_scores.csv",
    PROCESSED / "rfm_standard_raw.csv",
    PROCESSED / "rfm_log_standard.csv",
    PROCESSED / "rfm_robust_raw.csv",
    TABLES / "rfm_distribution_summary.csv",
    TABLES / "rfm_manual_validation.csv",
    TABLES / "rfm_outlier_summary.csv",
    TABLES / "rfm_score_distribution.csv",
    TABLES / "rfm_preprocessing_comparison.csv",
    TABLES / "rfm_acceptance_matrix.csv",
]
FIGURE_OUTPUTS = [
    FIGURES / "rfm_histograms_raw.png",
    FIGURES / "rfm_boxplots_raw.png",
    FIGURES / "rfm_histograms_log.png",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_input_baseline_and_integrity() -> None:
    assert INPUT.exists() and INPUT.stat().st_size > 0
    source = pd.read_csv(INPUT, dtype={"CustomerID": "string", "InvoiceNo": "string"}, low_memory=False)
    assert len(source) == EXPECTED_ROWS
    assert source["CustomerID"].nunique() == EXPECTED_CUSTOMERS
    checksum = pd.read_csv(TABLES / "input_checksum_report.csv")
    assert checksum.loc[0, "INPUT_SHA256_BEFORE"] == checksum.loc[0, "INPUT_SHA256_AFTER"]
    assert checksum.loc[0, "INPUT_SHA256_AFTER"] == sha256_file(INPUT)


def test_rfm_table_validity() -> None:
    rfm = pd.read_csv(PROCESSED / "rfm_customers.csv", dtype={"CustomerID": "string"}, parse_dates=["LastPurchaseDate"])
    assert len(rfm) == EXPECTED_CUSTOMERS
    assert rfm["CustomerID"].is_unique
    assert not rfm.isna().any().any()
    assert rfm["Recency"].ge(0).all()
    assert rfm["Frequency"].gt(0).all()
    assert rfm["Monetary"].gt(0).all()


def test_frequency_matches_source_nunique() -> None:
    source = pd.read_csv(INPUT, dtype={"CustomerID": "string", "InvoiceNo": "string"}, usecols=["CustomerID", "InvoiceNo"], low_memory=False)
    expected = source.groupby("CustomerID")["InvoiceNo"].nunique().sort_index()
    actual = pd.read_csv(PROCESSED / "rfm_customers.csv", dtype={"CustomerID": "string"}).set_index("CustomerID")["Frequency"].sort_index()
    pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False)


def test_manual_validation_five_of_five() -> None:
    validation = pd.read_csv(TABLES / "rfm_manual_validation.csv")
    assert len(validation) == 5
    assert validation["ValidationStatus"].eq("PASS").all()


def test_score_range() -> None:
    scored = pd.read_csv(PROCESSED / "rfm_with_scores.csv")
    for column in ["R_Score", "F_Score", "M_Score"]:
        assert scored[column].between(1, 5).all()


def test_three_candidates_are_finite_and_aligned() -> None:
    expected_ids = pd.read_csv(PROCESSED / "rfm_customers.csv", dtype={"CustomerID": "string"})["CustomerID"]
    for name in ["rfm_standard_raw.csv", "rfm_log_standard.csv", "rfm_robust_raw.csv"]:
        candidate = pd.read_csv(PROCESSED / name, dtype={"CustomerID": "string"})
        assert len(candidate) == EXPECTED_CUSTOMERS
        assert candidate["CustomerID"].equals(expected_ids)
        numeric = candidate.drop(columns="CustomerID")
        assert not numeric.isna().any().any()
        assert np.isfinite(numeric.to_numpy()).all()


def test_required_outputs_and_figures() -> None:
    for path in CSV_OUTPUTS:
        assert path.exists() and path.stat().st_size > 0, path
        pd.read_csv(path, low_memory=False)
    for path in FIGURE_OUTPUTS:
        assert path.exists() and path.stat().st_size > 0, path


def test_acceptance_matrix_passes() -> None:
    matrix = pd.read_csv(TABLES / "rfm_acceptance_matrix.csv")
    assert matrix["Status"].eq("PASS").all()
