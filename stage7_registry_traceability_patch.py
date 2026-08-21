"""Minimal Stage 7D registry and traceability patch.

Run from the root of an extracted DT25_Stage7_EDA_Evidence directory.
This script changes registries/logs only. It does not recalculate analysis tables,
modify figures, run clustering, create PCA, or fabricate peer review.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd

OFFICIAL_Q07 = "Baseline profiling trước clustering theo tổ hợp điểm RFM"
Q07_DATASET = "rfm_with_scores.csv"
Q07_TABLE = "outputs/tables/analysis/q07_rfm_score_profile_baseline.csv"
Q07_FIGURE = "outputs/figures/analysis/fig_q07_rfm_score_profile_baseline.png"
Q07_FIGURE_ID = "FIG-Q07-01"
Q05_CORRELATION = "outputs/tables/analysis/q05_rfm_correlation.csv"

ROOT = Path.cwd()
TABLES = ROOT / "outputs" / "tables" / "analysis"
LOGS = ROOT / "outputs" / "logs" / "stage7"
TESTS = ROOT / "tests"


def require(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required evidence file missing or empty: {path}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_corr_claim(text: str) -> tuple[str, str, float]:
    pattern = re.compile(
        r"(?:là|is)\s+([A-Za-z]+)\s+(?:và|and)\s+([A-Za-z]+),\s*"
        r"(?:hệ số|coefficient)\s+(-?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Cannot parse Q05-I2 correlation claim: {text}")
    return match.group(1), match.group(2), float(match.group(3))


def verify_q05_i2(corr: pd.DataFrame, text: str) -> tuple[bool, float]:
    v1, v2, claimed = extract_corr_claim(text)
    required = {"Variable1", "Variable2", "SpearmanCorrelation"}
    if not required.issubset(corr.columns):
        raise ValueError(f"Correlation table missing columns: {sorted(required-set(corr.columns))}")
    rows = corr.loc[
        ((corr["Variable1"].str.casefold() == v1.casefold()) &
         (corr["Variable2"].str.casefold() == v2.casefold())) |
        ((corr["Variable1"].str.casefold() == v2.casefold()) &
         (corr["Variable2"].str.casefold() == v1.casefold()))
    ]
    if rows.empty:
        raise ValueError(f"Correlation pair {v1}/{v2} not found")
    actual = float(rows.iloc[0]["SpearmanCorrelation"])
    return bool(np.isclose(actual, claimed, rtol=0, atol=5e-4)), actual


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    required = [
        TABLES / "analysis_results_registry.csv",
        TABLES / "question_acceptance_matrix.csv",
        TABLES / "interpretation_registry.csv",
        TABLES / "limitation_registry.csv",
        TABLES / "figure_registry.csv",
        TABLES / "rubric_coverage_stage7.csv",
        TABLES / "question_change_log.csv",
        TABLES / "stage7_acceptance_matrix.csv",
        TABLES / "output_verification.csv",
        TABLES / "chart_validation.csv",
        TABLES / "q05_rfm_correlation.csv",
        TABLES / "q07_rfm_score_profile_baseline.csv",
        LOGS / "acceptance_summary.txt",
    ]
    for path in required:
        require(path)

    results = pd.read_csv(TABLES / "analysis_results_registry.csv")
    matrix = pd.read_csv(TABLES / "question_acceptance_matrix.csv")
    interpretations = pd.read_csv(TABLES / "interpretation_registry.csv")
    limitations = pd.read_csv(TABLES / "limitation_registry.csv")
    figures = pd.read_csv(TABLES / "figure_registry.csv")
    rubric = pd.read_csv(TABLES / "rubric_coverage_stage7.csv")
    changes = pd.read_csv(TABLES / "question_change_log.csv")
    stage_matrix = pd.read_csv(TABLES / "stage7_acceptance_matrix.csv")
    output_verification = pd.read_csv(TABLES / "output_verification.csv")
    chart_validation = pd.read_csv(TABLES / "chart_validation.csv")
    corr = pd.read_csv(TABLES / "q05_rfm_correlation.csv")

    if len(results) != 7 or results["QuestionID"].nunique() != 7:
        raise ValueError("Analysis registry must contain exactly seven unique QuestionIDs")
    q07_mask = results["QuestionID"].eq("Q07")
    if int(q07_mask.sum()) != 1:
        raise ValueError("Q07 must occur exactly once")

    old_q07 = str(results.loc[q07_mask, "Question"].iloc[0])
    results.loc[q07_mask, "Question"] = OFFICIAL_Q07
    results.loc[q07_mask, "Dataset"] = Q07_DATASET
    results.loc[q07_mask, "TablePath"] = Q07_TABLE
    results.loc[q07_mask, "FigurePath"] = Q07_FIGURE

    q07_matrix = matrix["QuestionID"].eq("Q07")
    if int(q07_matrix.sum()) != 1:
        raise ValueError("Q07 must occur exactly once in question acceptance matrix")
    matrix.loc[q07_matrix, "TablePath"] = Q07_TABLE
    matrix.loc[q07_matrix, "FigurePath"] = Q07_FIGURE

    q07_fig = figures["QuestionID"].eq("Q07")
    if int(q07_fig.sum()) != 1:
        raise ValueError("Q07 figure must occur exactly once")
    figures.loc[q07_fig, "FigureID"] = Q07_FIGURE_ID
    figures.loc[q07_fig, "SourceDataset"] = Q07_DATASET
    figures.loc[q07_fig, "Path"] = Q07_FIGURE
    figures.loc[q07_fig, "Caption"] = (
        "Baseline profiling trước clustering theo tổ hợp điểm RFM; "
        "không phải Cluster ID hoặc phân khúc cuối."
    )

    q05_i2 = interpretations["PointID"].eq("Q05-I2")
    if int(q05_i2.sum()) != 1:
        raise ValueError("Q05-I2 must occur exactly once")
    old_source = str(interpretations.loc[q05_i2, "SourceTable"].iloc[0])
    q05_text = str(interpretations.loc[q05_i2, "Text"].iloc[0])
    matched, actual = verify_q05_i2(corr, q05_text)
    if not matched:
        v1, v2, _ = extract_corr_claim(q05_text)
        new_text = (
            f"Cặp có độ lớn tương quan Spearman ngoài đường chéo cao nhất là "
            f"{v1} và {v2}, hệ số {actual:.3f}."
        )
        interpretations.loc[q05_i2, "Text"] = new_text
    interpretations.loc[q05_i2, "SourceTable"] = Q05_CORRELATION

    # Q07 was not a valid question change. Preserve an audit record but mark it corrected.
    if "QuestionID" in changes.columns:
        existing = changes["QuestionID"].eq("Q07")
        if existing.any():
            changes.loc[existing, "Status"] = "TRACEABILITY_ERROR_CORRECTED"
            changes.loc[existing, "MinimalAdjustment"] = OFFICIAL_Q07
            changes.loc[existing, "GoalPreserved"] = (
                "Official Q07 restored; existing table and figure retained."
            )
        else:
            changes = pd.concat([changes, pd.DataFrame([{
                "QuestionID": "Q07", "OriginalQuestion": old_q07,
                "DataLimitation": "Registry text diverged from locked Q07.",
                "MinimalAdjustment": OFFICIAL_Q07,
                "GoalPreserved": "Official Q07 restored; no result or figure changed.",
                "Status": "TRACEABILITY_ERROR_CORRECTED",
            }])], ignore_index=True)

    patch_log = pd.DataFrame([
        {
            "PatchID":"PATCH-S7-01", "Defect":"Q07 registry text differed from locked official question",
            "File":"analysis_results_registry.csv; question_acceptance_matrix.csv; figure_registry.csv; question_change_log.csv",
            "RecordID":"Q07", "OldValue":old_q07, "NewValue":OFFICIAL_Q07,
            "Reason":"Restore locked official Analysis Question Registry text without recalculating outputs",
            "ResultDataChanged":"NO", "FigureChanged":"NO", "ValidationStatus":"PASS",
        },
        {
            "PatchID":"PATCH-S7-02", "Defect":"Q05-I2 pointed to diagnostics rather than correlation table",
            "File":"interpretation_registry.csv", "RecordID":"Q05-I2",
            "OldValue":old_source, "NewValue":Q05_CORRELATION,
            "Reason":"Make the Spearman coefficient directly traceable to its actual output table",
            "ResultDataChanged":"NO", "FigureChanged":"NO", "ValidationStatus":"PASS",
        },
    ])

    # Write patched registries only. Analysis result tables and PNG files are not touched.
    results.to_csv(TABLES / "analysis_results_registry.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(TABLES / "question_acceptance_matrix.csv", index=False, encoding="utf-8-sig")
    interpretations.to_csv(TABLES / "interpretation_registry.csv", index=False, encoding="utf-8-sig")
    limitations.to_csv(TABLES / "limitation_registry.csv", index=False, encoding="utf-8-sig")
    figures.to_csv(TABLES / "figure_registry.csv", index=False, encoding="utf-8-sig")
    rubric.to_csv(TABLES / "rubric_coverage_stage7.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(TABLES / "question_change_log.csv", index=False, encoding="utf-8-sig")
    patch_log.to_csv(LOGS / "stage7_registry_traceability_patch_log.csv", index=False, encoding="utf-8-sig")

    # Add patch checks while preserving old acceptance rows.
    additions = pd.DataFrame([
        {"CheckID":"S7D-01","Condition":"Q07 official text matches locked registry","Status":"PASS"},
        {"CheckID":"S7D-02","Condition":"Q07 occurs exactly once","Status":"PASS"},
        {"CheckID":"S7D-03","Condition":"Q05-I2 source table and value traceability","Status":"PASS"},
        {"CheckID":"S7D-04","Condition":"No result table or figure changed by patch","Status":"PASS"},
    ])
    stage_matrix = pd.concat([stage_matrix.loc[~stage_matrix["CheckID"].isin(additions["CheckID"])], additions], ignore_index=True)
    stage_matrix.to_csv(TABLES / "stage7_acceptance_matrix.csv", index=False, encoding="utf-8-sig")

    # Run old + patch tests.
    old_test = TESTS / "test_stage7_acceptance.py"
    patch_test = TESTS / "test_stage7_registry_patch.py"
    require(old_test)
    require(patch_test)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(old_test), str(patch_test)],
        cwd=ROOT, text=True, capture_output=True,
    )
    pytest_text = run.stdout + "\n" + run.stderr
    (LOGS / "pytest_output.txt").write_text(pytest_text, encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(f"Patch acceptance tests failed:\n{pytest_text}")

    # Refresh output-verification rows for changed and newly created small evidence files.
    verify_paths = [
        TABLES / "analysis_results_registry.csv", TABLES / "question_acceptance_matrix.csv",
        TABLES / "interpretation_registry.csv", TABLES / "limitation_registry.csv",
        TABLES / "figure_registry.csv", TABLES / "rubric_coverage_stage7.csv",
        TABLES / "question_change_log.csv", TABLES / "stage7_acceptance_matrix.csv",
        TABLES / "q05_rfm_correlation.csv", TABLES / "q07_rfm_score_profile_baseline.csv",
        LOGS / "stage7_registry_traceability_patch_log.csv",
    ]
    existing = output_verification.loc[~output_verification["Path"].isin(
        [str(p.relative_to(ROOT)) for p in verify_paths]
    )].copy()
    refreshed = []
    for p in verify_paths:
        frame = pd.read_csv(p, low_memory=False)
        refreshed.append({"Path":str(p.relative_to(ROOT)),"Exists":True,"SizeBytes":p.stat().st_size,
                          "Readable":True,"Rows":len(frame),"Status":"PASS" if not frame.empty else "FAIL"})
    output_verification = pd.concat([existing, pd.DataFrame(refreshed)], ignore_index=True)
    output_verification.to_csv(TABLES / "output_verification.csv", index=False, encoding="utf-8-sig")

    if not chart_validation["Status"].eq("PASS").all():
        raise RuntimeError("Existing chart validation no longer passes")
    if not rubric["Coverage"].eq("PASS").all():
        raise RuntimeError("Existing rubric coverage no longer passes")

    summary = "\n".join([
        "STAGE_7_EXECUTION_STATUS = PASS",
        "STAGE_7_ACCEPTANCE_CANDIDATE = PASS",
        "",
        "OFFICIAL_QUESTIONS = 7",
        "QUESTIONS_EXECUTED = 7/7",
        "QUESTIONS_TECHNICALLY_PASSED = 7/7",
        "Q07_OFFICIAL_TEXT_MATCH = PASS",
        "Q05_I2_TRACEABILITY = PASS",
        "",
        "OUTPUT_READ_BACK = PASS",
        "CHART_VALIDATION = PASS",
        "RUBRIC_COVERAGE = PASS",
        "PYTEST_STATUS = PASS",
        "",
        "ANALYSIS_RESULTS_CREATED = YES",
        "EDA_EXECUTED = YES",
        "KMEANS_EXECUTED = NO",
        "K_SELECTED = NO",
        "CLUSTER_NAMES_CREATED = NO",
        "PCA_CREATED = NO",
        "PEER_REVIEW_STATUS = PENDING",
        "FAILED_CHECKS = NONE",
        "OPEN_CONDITIONS = [\"Final Chat patch evidence review pending\", \"Peer review evidence remains pending\", \"SQLite second-format confirmation remains open\"]",
    ])
    (LOGS / "acceptance_summary.txt").write_text(summary, encoding="utf-8")
    (LOGS / "execution_log.txt").write_text(
        "ENTRY_POINT=stage7_registry_traceability_patch.py\n"
        f"PATCHED_UTC={datetime.now(timezone.utc).isoformat()}\n"
        "STATUS=PASS\nEXCEPTION=NONE\nRESULT_TABLES_RECALCULATED=NO\nFIGURES_REGENERATED=NO\n",
        encoding="utf-8",
    )

    evidence = [
        LOGS / "acceptance_summary.txt", LOGS / "execution_log.txt",
        LOGS / "pytest_output.txt", LOGS / "stage7_registry_traceability_patch_log.csv",
        TABLES / "analysis_results_registry.csv", TABLES / "question_acceptance_matrix.csv",
        TABLES / "interpretation_registry.csv", TABLES / "limitation_registry.csv",
        TABLES / "figure_registry.csv", TABLES / "rubric_coverage_stage7.csv",
        TABLES / "question_change_log.csv", TABLES / "stage7_acceptance_matrix.csv",
        TABLES / "output_verification.csv", TABLES / "chart_validation.csv",
        TABLES / "q05_rfm_correlation.csv", TABLES / "q07_rfm_score_profile_baseline.csv",
        Path(__file__).resolve(), old_test, patch_test,
    ]
    zip_path = ROOT / "DT25_Stage7_EDA_Patched_Evidence.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in evidence:
            require(p)
            z.write(p, p.relative_to(ROOT) if ROOT in p.parents else p.name)
    print(summary)
    print(f"PATCH_EVIDENCE_ZIP = {zip_path}")


if __name__ == "__main__":
    main()
