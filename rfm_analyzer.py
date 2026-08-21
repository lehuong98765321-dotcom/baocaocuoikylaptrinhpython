"""RFM feature engineering and preprocessing candidates for DT25.

This module reads an approved RFM-eligible transaction CSV, creates continuous
RFM features, descriptive quantile scores, customer-level outlier flags, and
three clustering-input candidates. It never modifies the input file and does
not run clustering.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

REQUIRED_COLUMNS = [
    "CustomerID", "InvoiceNo", "InvoiceDate", "Quantity",
    "UnitPrice", "TransactionAmount",
]
EXPECTED_INPUT_ROWS = 392_692
EXPECTED_INPUT_CUSTOMERS = 4_338
RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]
RANDOM_STATE = 42


class RFMInputValidationError(ValueError):
    """Raised when the uploaded processed input violates its accepted baseline."""


class RFMAcceptanceError(RuntimeError):
    """Raised when an RFM acceptance condition fails."""


@dataclass(frozen=True)
class ProjectPaths:
    """Project paths rooted at a caller-provided directory."""
    root: Path

    @property
    def input_file(self) -> Path:
        return self.root / "data" / "input" / "online_retail_rfm_eligible.csv"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def table_dir(self) -> Path:
        return self.root / "outputs" / "tables" / "rfm"

    @property
    def figure_dir(self) -> Path:
        return self.root / "outputs" / "figures" / "rfm"

    @property
    def log_dir(self) -> Path:
        return self.root / "outputs" / "logs"

    def create_output_dirs(self) -> None:
        for path in (self.processed_dir, self.table_dir, self.figure_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class RFMArtifacts:
    """All data artifacts generated before export."""
    rfm: pd.DataFrame
    rfm_with_scores: pd.DataFrame
    standard_raw: pd.DataFrame
    log_standard: pd.DataFrame
    robust_raw: pd.DataFrame
    manual_validation: pd.DataFrame
    distribution_summary: pd.DataFrame
    outlier_summary: pd.DataFrame
    score_distribution: pd.DataFrame
    preprocessing_comparison: pd.DataFrame
    acceptance_matrix: pd.DataFrame


class RFMAnalyzer:
    """Build reproducible RFM features from accepted transaction records."""

    def __init__(self, paths: ProjectPaths, random_state: int = RANDOM_STATE) -> None:
        self.paths = paths
        self.random_state = random_state
        self.transactions: pd.DataFrame | None = None
        self.reference_date: pd.Timestamp | None = None
        self.input_sha256_before: str | None = None
        self.input_sha256_after: str | None = None
        self.min_invoice_date: pd.Timestamp | None = None
        self.max_invoice_date: pd.Timestamp | None = None

    @staticmethod
    def sha256_file(path: Path) -> str:
        """Return a file's SHA-256 hash."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def validate_input(self) -> pd.DataFrame:
        """Read and validate the accepted RFM-eligible CSV and locked baselines."""
        path = self.paths.input_file
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.stat().st_size <= 0:
            raise RFMInputValidationError("Input CSV is empty.")
        try:
            df = pd.read_csv(
                path,
                dtype={"CustomerID": "string", "InvoiceNo": "string"},
                low_memory=False,
            )
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise RFMInputValidationError(f"Input CSV cannot be read: {exc}") from exc
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise RFMInputValidationError(f"Missing required columns: {missing}")

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        for column in ("Quantity", "UnitPrice", "TransactionAmount"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        failures: list[str] = []
        if len(df) != EXPECTED_INPUT_ROWS:
            failures.append(f"rows={len(df)} expected={EXPECTED_INPUT_ROWS}")
        customers = int(df["CustomerID"].nunique(dropna=True))
        if customers != EXPECTED_INPUT_CUSTOMERS:
            failures.append(f"customers={customers} expected={EXPECTED_INPUT_CUSTOMERS}")
        if df["CustomerID"].isna().any(): failures.append("CustomerID contains missing values")
        if df["InvoiceNo"].isna().any(): failures.append("InvoiceNo contains missing values")
        if df["InvoiceDate"].isna().any(): failures.append("InvoiceDate contains invalid values")
        if not df["Quantity"].gt(0).all(): failures.append("Quantity contains non-positive or invalid values")
        if not df["UnitPrice"].gt(0).all(): failures.append("UnitPrice contains non-positive or invalid values")
        if not df["TransactionAmount"].gt(0).all(): failures.append("TransactionAmount contains non-positive or invalid values")
        if df["InvoiceNo"].str.strip().str.upper().str.startswith("C", na=False).any():
            failures.append("Cancellation invoice remains in input")
        if failures:
            raise RFMInputValidationError("; ".join(failures))
        self.transactions = df
        return df

    def determine_reference_date(self) -> pd.Timestamp:
        """Set reference date to max valid invoice date plus exactly one day."""
        if self.transactions is None:
            raise RFMAcceptanceError("validate_input must run first.")
        self.min_invoice_date = self.transactions["InvoiceDate"].min()
        self.max_invoice_date = self.transactions["InvoiceDate"].max()
        self.reference_date = self.max_invoice_date + pd.Timedelta(days=1)
        if self.reference_date - self.max_invoice_date != pd.Timedelta(days=1):
            raise RFMAcceptanceError("Reference date is not exactly max date plus one day.")
        return self.reference_date

    def compute_rfm(self) -> pd.DataFrame:
        """Aggregate one RFM row per CustomerID."""
        if self.transactions is None or self.reference_date is None:
            raise RFMAcceptanceError("Input validation and reference date are required.")
        grouped = self.transactions.groupby("CustomerID", as_index=False).agg(
            LastPurchaseDate=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TransactionAmount", "sum"),
        )
        grouped["Recency"] = (
            self.reference_date.normalize() - grouped["LastPurchaseDate"].dt.normalize()
        ).dt.days.astype("int64")
        grouped["Frequency"] = grouped["Frequency"].astype("int64")
        grouped = grouped[["CustomerID", "LastPurchaseDate", "Recency", "Frequency", "Monetary"]]
        return grouped.sort_values("CustomerID", kind="stable").reset_index(drop=True)

    @staticmethod
    def validate_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
        """Return an acceptance matrix and raise for an invalid RFM table."""
        checks = [
            ("RFM-01", "One row per accepted customer", len(rfm) == EXPECTED_INPUT_CUSTOMERS),
            ("RFM-02", "CustomerID is unique", rfm["CustomerID"].is_unique),
            ("RFM-03", "No missing RFM values", not rfm.isna().any().any()),
            ("RFM-04", "Recency is integer and non-negative", pd.api.types.is_integer_dtype(rfm["Recency"]) and rfm["Recency"].ge(0).all()),
            ("RFM-05", "Frequency is integer and positive", pd.api.types.is_integer_dtype(rfm["Frequency"]) and rfm["Frequency"].gt(0).all()),
            ("RFM-06", "Monetary is positive", rfm["Monetary"].gt(0).all()),
        ]
        matrix = pd.DataFrame([
            {"CheckID": key, "Condition": text, "Result": bool(result), "Status": "PASS" if result else "FAIL"}
            for key, text, result in checks
        ])
        if not matrix["Status"].eq("PASS").all():
            raise RFMAcceptanceError(f"RFM validation failed:\n{matrix.loc[matrix.Status.eq('FAIL')]}")
        return matrix

    def create_manual_validation_sample(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Validate five required customer-selection roles against source transactions."""
        if self.transactions is None or self.reference_date is None:
            raise RFMAcceptanceError("Transactions and reference date are required.")
        customer_numeric = pd.to_numeric(rfm["CustomerID"], errors="raise")
        selectors = [
            ("MIN_CUSTOMER_ID", rfm.loc[customer_numeric.idxmin(), "CustomerID"]),
            ("MAX_CUSTOMER_ID", rfm.loc[customer_numeric.idxmax(), "CustomerID"]),
            ("MAX_FREQUENCY", rfm.loc[rfm["Frequency"].idxmax(), "CustomerID"]),
            ("MAX_MONETARY", rfm.loc[rfm["Monetary"].idxmax(), "CustomerID"]),
        ]
        rng = np.random.default_rng(self.random_state)
        available = [value for value in rfm["CustomerID"].tolist() if value not in {item[1] for item in selectors}]
        random_customer = available[int(rng.integers(0, len(available)))]
        selectors.append((f"RANDOM_STATE_{self.random_state}", random_customer))

        records: list[dict[str, Any]] = []
        indexed_rfm = rfm.set_index("CustomerID")
        for role, customer_id in selectors:
            source = self.transactions.loc[self.transactions["CustomerID"].eq(customer_id)]
            source_last = source["InvoiceDate"].max()
            source_frequency = int(source["InvoiceNo"].nunique())
            source_monetary = float(source["TransactionAmount"].sum())
            source_recency = int((self.reference_date.normalize() - source_last.normalize()).days)
            row = indexed_rfm.loc[customer_id]
            last_match = bool(source_last == row["LastPurchaseDate"])
            frequency_match = bool(source_frequency == int(row["Frequency"]))
            monetary_match = bool(np.isclose(source_monetary, float(row["Monetary"]), rtol=1e-9, atol=1e-6))
            recency_match = bool(source_recency == int(row["Recency"]))
            records.append({
                "SelectionRole": role, "CustomerID": customer_id,
                "SourceLastPurchaseDate": source_last, "RFM_LastPurchaseDate": row["LastPurchaseDate"],
                "SourceFrequency": source_frequency, "RFM_Frequency": int(row["Frequency"]),
                "SourceMonetary": source_monetary, "RFM_Monetary": float(row["Monetary"]),
                "SourceRecency": source_recency, "RFM_Recency": int(row["Recency"]),
                "LastDateMatch": last_match, "FrequencyMatch": frequency_match,
                "MonetaryMatch": monetary_match, "RecencyMatch": recency_match,
                "ValidationStatus": "PASS" if all([last_match, frequency_match, monetary_match, recency_match]) else "FAIL",
            })
        result = pd.DataFrame(records)
        if len(result) != 5 or not result["ValidationStatus"].eq("PASS").all():
            raise RFMAcceptanceError("Manual five-customer validation failed.")
        return result

    @staticmethod
    def compute_distribution_summary(rfm: pd.DataFrame) -> pd.DataFrame:
        """Compute required distribution statistics for continuous RFM."""
        records = []
        for variable in RFM_COLUMNS:
            series = rfm[variable]
            records.append({
                "Variable": variable, "Count": int(series.count()), "Mean": float(series.mean()),
                "Std": float(series.std()), "Min": float(series.min()), "Q1": float(series.quantile(.25)),
                "Median": float(series.median()), "Q3": float(series.quantile(.75)), "Max": float(series.max()),
                "Skewness": float(series.skew()), "Q90": float(series.quantile(.90)),
                "Q95": float(series.quantile(.95)), "Q99": float(series.quantile(.99)),
                "Q999": float(series.quantile(.999)), "MissingCount": int(series.isna().sum()),
                "InvalidCount": int((series.lt(0) if variable == "Recency" else series.le(0)).sum()),
            })
        return pd.DataFrame(records)

    @staticmethod
    def flag_customer_outliers(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Flag customer RFM outliers with IQR and retain every customer."""
        out = rfm.copy()
        records = []
        for variable in RFM_COLUMNS:
            q1, q3 = out[variable].quantile([.25, .75])
            iqr = q3 - q1
            lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            flag_name = f"{variable}_Outlier"
            out[flag_name] = out[variable].lt(lower) | out[variable].gt(upper)
            records.append({
                "Variable": variable, "Method": "IQR 1.5", "Q1": float(q1), "Q3": float(q3),
                "IQR": float(iqr), "LowerLimit": lower, "UpperLimit": upper,
                "FlaggedCustomers": int(out[flag_name].sum()),
                "FlaggedRatePercent": float(out[flag_name].mean() * 100),
                "Treatment": "FLAG_ONLY_NO_AUTOMATIC_REMOVAL",
            })
        flag_columns = [f"{variable}_Outlier" for variable in RFM_COLUMNS]
        out["Any_RFM_Outlier"] = out[flag_columns].any(axis=1)
        records.append({
            "Variable": "Any_RFM_Outlier", "Method": "Union of three IQR flags",
            "Q1": np.nan, "Q3": np.nan, "IQR": np.nan, "LowerLimit": np.nan, "UpperLimit": np.nan,
            "FlaggedCustomers": int(out["Any_RFM_Outlier"].sum()),
            "FlaggedRatePercent": float(out["Any_RFM_Outlier"].mean() * 100),
            "Treatment": "FLAG_ONLY_REVIEW_CONTEXT",
        })
        return out, pd.DataFrame(records)

    @staticmethod
    def _rank_quantile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
        """Create stable quintile scores despite duplicated raw values."""
        percentile = series.rank(method="first", pct=True)
        score = np.ceil(percentile * 5).clip(1, 5).astype("int64")
        return (6 - score).astype("int64") if reverse else score

    def create_rfm_scores(self, rfm_flagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create descriptive 1-5 scores without replacing continuous clustering features."""
        out = rfm_flagged.copy()
        out["R_Score"] = self._rank_quantile_score(out["Recency"], reverse=True)
        out["F_Score"] = self._rank_quantile_score(out["Frequency"], reverse=False)
        out["M_Score"] = self._rank_quantile_score(out["Monetary"], reverse=False)
        out["RFM_Score_String"] = out[["R_Score", "F_Score", "M_Score"]].astype(str).agg("".join, axis=1)
        out["RFM_TotalScore"] = out[["R_Score", "F_Score", "M_Score"]].sum(axis=1).astype("int64")
        for column in ("R_Score", "F_Score", "M_Score"):
            if not out[column].between(1, 5).all():
                raise RFMAcceptanceError(f"{column} contains a value outside 1..5")
        distribution = (
            out.groupby(["R_Score", "F_Score", "M_Score"], as_index=False)
            .size().rename(columns={"size": "Customers"})
        )
        return out, distribution

    @staticmethod
    def build_preprocessing_candidates(rfm: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Build three verified feature candidates without selecting a winner."""
        features = rfm[RFM_COLUMNS].astype("float64")
        transformed = {
            "STANDARD_RAW": StandardScaler().fit_transform(features),
            "LOG_STANDARD": StandardScaler().fit_transform(np.log1p(features)),
            "ROBUST_RAW": RobustScaler().fit_transform(features),
        }
        candidates: dict[str, pd.DataFrame] = {}
        comparisons: list[dict[str, Any]] = []
        for name, matrix in transformed.items():
            frame = pd.DataFrame(matrix, columns=[f"{column}_Scaled" for column in RFM_COLUMNS])
            frame.insert(0, "CustomerID", rfm["CustomerID"].to_numpy())
            numeric = frame.drop(columns="CustomerID")
            same_ids = frame["CustomerID"].astype("string").reset_index(drop=True).equals(
                rfm["CustomerID"].astype("string").reset_index(drop=True)
            )
            if len(frame) != len(rfm) or not same_ids:
                raise RFMAcceptanceError(f"Customer identity changed in candidate {name}")
            if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
                raise RFMAcceptanceError(f"NaN or infinity in candidate {name}")
            candidates[name] = frame
            for column in numeric.columns:
                comparisons.append({
                    "Candidate": name, "Feature": column,
                    "Rows": len(frame), "Customers": int(frame["CustomerID"].nunique()),
                    "MissingCount": int(numeric[column].isna().sum()),
                    "InfiniteCount": int(np.isinf(numeric[column]).sum()),
                    "Mean": float(numeric[column].mean()), "Median": float(numeric[column].median()),
                    "Std": float(numeric[column].std(ddof=0)), "Min": float(numeric[column].min()),
                    "Max": float(numeric[column].max()), "Skewness": float(numeric[column].skew()),
                    "SelectionStatus": "CANDIDATE_NOT_SELECTED",
                })
        return candidates, pd.DataFrame(comparisons)

    def create_evidence_summary(self, artifacts: RFMArtifacts) -> pd.DataFrame:
        """Create a single-row execution summary from computed artifacts."""
        if self.transactions is None or self.reference_date is None:
            raise RFMAcceptanceError("Execution state incomplete.")
        rfm = artifacts.rfm
        return pd.DataFrame([{
            "INPUT_ROWS": len(self.transactions),
            "INPUT_CUSTOMERS": int(self.transactions["CustomerID"].nunique()),
            "MIN_INVOICE_DATE": self.min_invoice_date,
            "MAX_INVOICE_DATE": self.max_invoice_date,
            "REFERENCE_DATE": self.reference_date,
            "RFM_ROWS": len(rfm), "RFM_CUSTOMERS": int(rfm["CustomerID"].nunique()),
            "RFM_DUPLICATE_CUSTOMER_IDS": int(rfm["CustomerID"].duplicated().sum()),
            "RFM_MISSING_VALUES": int(rfm.isna().sum().sum()),
            "RFM_INVALID_RECENCY": int(rfm["Recency"].lt(0).sum()),
            "RFM_INVALID_FREQUENCY": int(rfm["Frequency"].le(0).sum()),
            "RFM_INVALID_MONETARY": int(rfm["Monetary"].le(0).sum()),
            "MANUAL_VALIDATION_PASS": int(artifacts.manual_validation["ValidationStatus"].eq("PASS").sum()),
            "RFM_OUTLIER_CUSTOMERS": int(artifacts.rfm_with_scores["Any_RFM_Outlier"].sum()),
        }])

    def export_outputs(self, artifacts: RFMArtifacts) -> dict[str, Path]:
        """Export ten required CSV files; figures are created by the notebook."""
        self.paths.create_output_dirs()
        outputs = {
            "rfm_customers": self.paths.processed_dir / "rfm_customers.csv",
            "rfm_with_scores": self.paths.processed_dir / "rfm_with_scores.csv",
            "rfm_standard_raw": self.paths.processed_dir / "rfm_standard_raw.csv",
            "rfm_log_standard": self.paths.processed_dir / "rfm_log_standard.csv",
            "rfm_robust_raw": self.paths.processed_dir / "rfm_robust_raw.csv",
            "rfm_distribution_summary": self.paths.table_dir / "rfm_distribution_summary.csv",
            "rfm_manual_validation": self.paths.table_dir / "rfm_manual_validation.csv",
            "rfm_outlier_summary": self.paths.table_dir / "rfm_outlier_summary.csv",
            "rfm_score_distribution": self.paths.table_dir / "rfm_score_distribution.csv",
            "rfm_preprocessing_comparison": self.paths.table_dir / "rfm_preprocessing_comparison.csv",
            "rfm_acceptance_matrix": self.paths.table_dir / "rfm_acceptance_matrix.csv",
        }
        frames = {
            "rfm_customers": artifacts.rfm,
            "rfm_with_scores": artifacts.rfm_with_scores,
            "rfm_standard_raw": artifacts.standard_raw,
            "rfm_log_standard": artifacts.log_standard,
            "rfm_robust_raw": artifacts.robust_raw,
            "rfm_distribution_summary": artifacts.distribution_summary,
            "rfm_manual_validation": artifacts.manual_validation,
            "rfm_outlier_summary": artifacts.outlier_summary,
            "rfm_score_distribution": artifacts.score_distribution,
            "rfm_preprocessing_comparison": artifacts.preprocessing_comparison,
            "rfm_acceptance_matrix": artifacts.acceptance_matrix,
        }
        for key, path in outputs.items():
            frames[key].to_csv(path, index=False, encoding="utf-8-sig")
        return outputs

    def run(self) -> tuple[RFMArtifacts, dict[str, Path], pd.DataFrame]:
        """Execute feature engineering without clustering or customer removal."""
        self.paths.create_output_dirs()
        self.input_sha256_before = self.sha256_file(self.paths.input_file)
        self.validate_input()
        self.determine_reference_date()
        rfm = self.compute_rfm()
        acceptance = self.validate_rfm(rfm)
        manual = self.create_manual_validation_sample(rfm)
        distribution = self.compute_distribution_summary(rfm)
        flagged, outlier_summary = self.flag_customer_outliers(rfm)
        scored, score_distribution = self.create_rfm_scores(flagged)
        candidates, comparison = self.build_preprocessing_candidates(rfm)
        artifacts = RFMArtifacts(
            rfm=rfm, rfm_with_scores=scored,
            standard_raw=candidates["STANDARD_RAW"],
            log_standard=candidates["LOG_STANDARD"],
            robust_raw=candidates["ROBUST_RAW"],
            manual_validation=manual, distribution_summary=distribution,
            outlier_summary=outlier_summary, score_distribution=score_distribution,
            preprocessing_comparison=comparison, acceptance_matrix=acceptance,
        )
        outputs = self.export_outputs(artifacts)
        self.input_sha256_after = self.sha256_file(self.paths.input_file)
        if self.input_sha256_before != self.input_sha256_after:
            raise RFMAcceptanceError("Input file checksum changed during execution.")
        return artifacts, outputs, self.create_evidence_summary(artifacts)
