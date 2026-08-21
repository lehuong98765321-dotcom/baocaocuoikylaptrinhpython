"""Stage 7D patch acceptance tests."""
from pathlib import Path
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/"outputs"/"tables"/"analysis"
L=ROOT/"outputs"/"logs"/"stage7"
OFFICIAL="Baseline profiling trước clustering theo tổ hợp điểm RFM"

def test_q07_official_registry():
    r=pd.read_csv(T/"analysis_results_registry.csv")
    assert len(r)==7 and r.QuestionID.nunique()==7
    q=r.loc[r.QuestionID.eq("Q07")]
    assert len(q)==1
    row=q.iloc[0]
    assert row.Question==OFFICIAL
    assert row.Dataset=="rfm_with_scores.csv"
    assert row.TablePath.endswith("q07_rfm_score_profile_baseline.csv")
    assert row.FigurePath.endswith("fig_q07_rfm_score_profile_baseline.png")
    f=pd.read_csv(T/"figure_registry.csv")
    qf=f.loc[f.QuestionID.eq("Q07")]
    assert len(qf)==1 and qf.iloc[0].FigureID=="FIG-Q07-01"
    assert not r.Question.str.contains("Trước khi clustering, các tổ hợp",regex=False).any()

def test_q05_i2_traceability():
    i=pd.read_csv(T/"interpretation_registry.csv")
    q=i.loc[i.PointID.eq("Q05-I2")]
    assert len(q)==1
    source=ROOT/q.iloc[0].SourceTable
    assert source.name=="q05_rfm_correlation.csv"
    assert source.exists() and source.stat().st_size>0
    c=pd.read_csv(source)
    m=re.search(r"là ([A-Za-z]+) và ([A-Za-z]+), hệ số (-?\d+(?:\.\d+)?)",q.iloc[0].Text)
    assert m
    v1,v2,claim=m.group(1),m.group(2),float(m.group(3))
    rows=c.loc[((c.Variable1==v1)&(c.Variable2==v2))|((c.Variable1==v2)&(c.Variable2==v1))]
    assert len(rows)>=1
    assert abs(float(rows.iloc[0].SpearmanCorrelation)-claim)<=0.0005

def test_patch_log_and_no_result_changes_claim():
    p=pd.read_csv(L/"stage7_registry_traceability_patch_log.csv")
    assert set(p.PatchID)=={"PATCH-S7-01","PATCH-S7-02"}
    assert p.ResultDataChanged.eq("NO").all()
    assert p.FigureChanged.eq("NO").all()
    assert p.ValidationStatus.eq("PASS").all()

def test_existing_acceptance_still_passes():
    assert pd.read_csv(T/"chart_validation.csv").Status.eq("PASS").all()
    assert pd.read_csv(T/"rubric_coverage_stage7.csv").Coverage.eq("PASS").all()
    assert pd.read_csv(T/"stage7_acceptance_matrix.csv").Status.eq("PASS").all()

def test_no_forbidden_artifacts():
    names=[p.name.lower() for p in ROOT.rglob("*") if p.is_file()]
    assert not any("kmeans" in n or "pca" in n or "cluster_id" in n for n in names)
