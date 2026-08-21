from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; T=ROOT/'outputs/tables/clustering'; F=ROOT/'outputs/figures/clustering'; P=ROOT/'data/processed'

def test_inputs_candidates():
 v=pd.read_csv(T/'input_verification.csv'); assert set(v.Candidate)=={'RFM_REFERENCE','STANDARD_RAW','LOG_STANDARD','ROBUST_RAW'}; assert v.Status.eq('PASS').all(); assert (v.Rows==4338).all()
def test_k_grid_and_metrics():
 m=pd.read_csv(T/'kmeans_metrics_all.csv'); assert len(m)==27; assert set(m.K)==set(range(2,11)); assert set(m.Candidate)=={'STANDARD_RAW','LOG_STANDARD','ROBUST_RAW'}; assert m[['Inertia','Silhouette','DaviesBouldin','CalinskiHarabasz']].notna().all().all()
def test_unique_selection_and_assignments():
 s=pd.read_csv(T/'candidate_selection_matrix.csv'); assert s.SelectionStatus.eq('SELECTED').sum()==1; row=s.loc[s.SelectionStatus.eq('SELECTED')].iloc[0]; assert int(row.K) in range(2,11)
 a=pd.read_csv(P/'kmeans_customer_assignments.csv',dtype={'CustomerID':'string'}); assert len(a)==4338 and a.CustomerID.is_unique; assert a.ClusterID.nunique()==int(row.K); assert not any('ClusterName' in c for c in a.columns)
def test_stability_and_ari():
 st=pd.read_csv(T/'multi_seed_stability.csv'); ari=pd.read_csv(T/'ari_stability_matrix.csv'); assert set(st.Seed)=={0,7,21,42,84,123,2026}; assert ari.ARI.between(-1,1).all()
def test_integrity_outputs_charts():
 c=pd.read_csv(T/'input_checksum_report.csv'); assert c.Unchanged.all(); o=pd.read_csv(T/'output_verification.csv'); assert o.Status.eq('PASS').all(); cv=pd.read_csv(T/'chart_validation.csv'); assert cv.Status.eq('PASS').all()
 for p in F.glob('*.png'):
  with Image.open(p) as im: assert im.width>=400 and im.height>=300
def test_acceptance_and_prohibitions():
 a=pd.read_csv(T/'stage8_acceptance_matrix.csv'); assert a.Status.eq('PASS').all(); names=[p.name.lower() for p in ROOT.rglob('*') if p.is_file()]; assert not any('pca' in n or 'clustername' in n for n in names)
