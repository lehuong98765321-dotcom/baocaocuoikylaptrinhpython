"""Acceptance tests for externally executed Stage 7 EDA outputs."""
from pathlib import Path
import re
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/"outputs"/"tables"/"analysis"; F=ROOT/"outputs"/"figures"/"analysis"

def test_seven_questions_and_outputs():
    r=pd.read_csv(T/"analysis_results_registry.csv")
    assert len(r)==7 and r.QuestionID.nunique()==7 and set(r.QuestionID)=={f"Q{i:02d}" for i in range(1,8)}
    for _,x in r.iterrows():
        assert (ROOT/x.TablePath).exists() and (ROOT/x.TablePath).stat().st_size>0
        assert (ROOT/x.FigurePath).exists() and (ROOT/x.FigurePath).stat().st_size>0
        assert not pd.read_csv(ROOT/x.TablePath,low_memory=False).empty

def test_seven_distinct_chart_types_and_unique_figures():
    f=pd.read_csv(T/"figure_registry.csv")
    assert f.FigureID.is_unique and f.Path.is_unique and len(f)==7
    normalized=set()
    for value in f.ChartType:
        for item in re.split(r"\s*\+\s*|\s*;\s*", value.lower()):
            item=item.strip()
            if item == "profile heatmap": item = "heatmap"
            normalized.add(item)
    assert {"line chart","horizontal bar chart","histogram","boxplot","stacked bar chart","heatmap","pareto chart"}.issubset(normalized)
    assert not f.ChartType.str.contains("3d",case=False).any()

def test_interpretations_limitations_and_source_links():
    i=pd.read_csv(T/"interpretation_registry.csv"); l=pd.read_csv(T/"limitation_registry.csv")
    assert set(i.QuestionID)=={f"Q{x:02d}" for x in range(1,8)}
    assert set(l.QuestionID)==set(i.QuestionID)
    assert i.Text.str.len().gt(10).all() and l.Text.str.len().gt(10).all()
    assert i.SourceTable.map(lambda p:(ROOT/p).exists()).all()
    forbidden=re.compile(r"\b(causes?|leads? to|results? in)\b",re.I)
    assert not i.Text.map(lambda x:bool(forbidden.search(str(x)))).any()

def test_chart_files_valid():
    f=pd.read_csv(T/"figure_registry.csv")
    for _,x in f.iterrows():
        path=ROOT/x.Path
        assert path.exists() and path.stat().st_size>0
        with Image.open(path) as im:
            assert im.width>=400 and im.height>=300
        assert str(x.Title).strip() and str(x.XLabel).strip() and str(x.YLabel).strip()

def test_input_integrity_and_output_readback():
    c=pd.read_csv(T/"input_checksum_report.csv"); assert c.Unchanged.all()
    o=pd.read_csv(T/"output_verification.csv"); assert o.Status.eq("PASS").all()

def test_rubric_coverage_and_acceptance():
    rubric=pd.read_csv(T/"rubric_coverage_stage7.csv"); assert rubric.Coverage.eq("PASS").all()
    matrix=pd.read_csv(T/"stage7_acceptance_matrix.csv"); assert matrix.Status.eq("PASS").all()

def test_no_clustering_artifacts():
    files=[p.name.lower() for p in ROOT.rglob("*") if p.is_file()]
    assert not any("pca" in x or "kmeans" in x or "cluster_id" in x for x in files)
