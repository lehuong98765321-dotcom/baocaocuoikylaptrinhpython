"""DT25 Stage 8 K-Means model-selection engine.

Evaluates three accepted RFM preprocessing candidates across k=2..10,
checks multi-seed stability and cluster-size risks, selects exactly one
candidate-k pair using an explicit composite evidence rule, and exports
technical ClusterID assignments. No PCA, ClusterName, synthetic data, or
non-K-Means model is created.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score)

EXPECTED_CUSTOMERS=4338
K_VALUES=list(range(2,11))
SEEDS=[0,7,21,42,84,123,2026]
PRIMARY_SEED=42
N_INIT=20
MAX_ITER=500
TINY_THRESHOLD=0.01
CANDIDATES={
 "STANDARD_RAW":"rfm_standard_raw.csv",
 "LOG_STANDARD":"rfm_log_standard.csv",
 "ROBUST_RAW":"rfm_robust_raw.csv",
}
FEATURES=["Recency_Scaled","Frequency_Scaled","Monetary_Scaled"]

class ClusterInputError(ValueError): pass
class ClusterAcceptanceError(RuntimeError): pass

@dataclass(frozen=True)
class Paths:
 root: Path
 @property
 def input(self): return self.root/"data"/"input"
 @property
 def processed(self): return self.root/"data"/"processed"
 @property
 def tables(self): return self.root/"outputs"/"tables"/"clustering"
 @property
 def figures(self): return self.root/"outputs"/"figures"/"clustering"
 @property
 def logs(self): return self.root/"outputs"/"logs"/"stage8"
 def create(self):
  for p in (self.processed,self.tables,self.figures,self.logs): p.mkdir(parents=True,exist_ok=True)

class ClusterEvaluator:
 def __init__(self,paths:Paths):
  self.paths=paths; self.frames={}; self.hash_before={}; self.hash_after={}
  self.rfm=None
 @staticmethod
 def sha256(path:Path)->str:
  h=hashlib.sha256()
  with path.open("rb") as f:
   for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
  return h.hexdigest()
 def validate_inputs(self)->pd.DataFrame:
  self.paths.create(); rows=[]
  rpath=self.paths.input/"rfm_customers.csv"
  if not rpath.exists() or rpath.stat().st_size<=0: raise ClusterInputError("Missing rfm_customers.csv")
  self.hash_before[rpath.name]=self.sha256(rpath)
  self.rfm=pd.read_csv(rpath,dtype={"CustomerID":"string"})
  need={"CustomerID","Recency","Frequency","Monetary"}
  if not need.issubset(self.rfm.columns) or len(self.rfm)!=EXPECTED_CUSTOMERS or not self.rfm.CustomerID.is_unique: raise ClusterInputError("Invalid rfm_customers.csv baseline/schema")
  rows.append({"File":rpath.name,"Candidate":"RFM_REFERENCE","Rows":len(self.rfm),"Customers":self.rfm.CustomerID.nunique(),"Missing":int(self.rfm[list(need)].isna().sum().sum()),"Infinite":0,"SHA256":self.hash_before[rpath.name],"Status":"PASS"})
  ids=self.rfm.CustomerID.reset_index(drop=True)
  for name,file in CANDIDATES.items():
   p=self.paths.input/file
   if not p.exists() or p.stat().st_size<=0: raise ClusterInputError(f"Missing {file}")
   self.hash_before[file]=self.sha256(p)
   df=pd.read_csv(p,dtype={"CustomerID":"string"})
   missing=sorted(set(["CustomerID"]+FEATURES)-set(df.columns))
   numeric=df[FEATURES] if not missing else pd.DataFrame()
   if missing or len(df)!=EXPECTED_CUSTOMERS or not df.CustomerID.is_unique: raise ClusterInputError(f"Invalid {file}: {missing}")
   if not df.CustomerID.reset_index(drop=True).equals(ids): raise ClusterInputError(f"Customer order/set mismatch: {file}")
   if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all(): raise ClusterInputError(f"NaN/inf: {file}")
   self.frames[name]=df
   rows.append({"File":file,"Candidate":name,"Rows":len(df),"Customers":df.CustomerID.nunique(),"Missing":0,"Infinite":0,"SHA256":self.hash_before[file],"Status":"PASS"})
  return pd.DataFrame(rows)
 def fit_candidate(self,name:str,k:int,seed:int=PRIMARY_SEED)->dict[str,Any]:
  x=self.frames[name][FEATURES].to_numpy(); start=perf_counter()
  model=KMeans(n_clusters=k,init="k-means++",n_init=N_INIT,max_iter=MAX_ITER,random_state=seed,algorithm="lloyd")
  labels=model.fit_predict(x); runtime=perf_counter()-start
  sizes=np.bincount(labels,minlength=k); min_size=int(sizes.min()); max_size=int(sizes.max())
  return {"Candidate":name,"K":k,"Seed":seed,"Inertia":float(model.inertia_),"Silhouette":float(silhouette_score(x,labels)),"DaviesBouldin":float(davies_bouldin_score(x,labels)),"CalinskiHarabasz":float(calinski_harabasz_score(x,labels)),"Iterations":int(model.n_iter_),"RuntimeSeconds":runtime,"MinClusterSize":min_size,"MaxClusterSize":max_size,"MinClusterRatio":min_size/len(x),"MaxClusterRatio":max_size/len(x),"ImbalanceRatio":max_size/min_size,"ClusterSizes":json.dumps(sizes.tolist()),"Labels":labels,"Model":model}
 def evaluate_k_range(self)->pd.DataFrame:
  records=[]
  for name in CANDIDATES:
   for k in K_VALUES:
    d=self.fit_candidate(name,k); records.append({x:y for x,y in d.items() if x not in {"Labels","Model"}})
  return pd.DataFrame(records)
 @staticmethod
 def detect_elbow(metrics:pd.DataFrame)->pd.DataFrame:
  rows=[]
  for name,g in metrics.groupby("Candidate"):
   g=g.sort_values("K"); y=g.Inertia.to_numpy(); curvature=np.full(len(y),np.nan)
   if len(y)>=3: curvature[1:-1]=y[:-2]-2*y[1:-1]+y[2:]
   for (_,r),c in zip(g.iterrows(),curvature): rows.append({"Candidate":name,"K":int(r.K),"Inertia":r.Inertia,"DiscreteCurvature":c,"ElbowCandidate":bool(np.isfinite(c) and c==np.nanmax(curvature))})
  return pd.DataFrame(rows)
 @staticmethod
 def shortlist(metrics:pd.DataFrame)->pd.DataFrame:
  d=metrics.copy(); d["TinyWarning"]=d.MinClusterRatio<TINY_THRESHOLD
  for c,asc in [("Silhouette",False),("DaviesBouldin",True),("CalinskiHarabasz",False),("ImbalanceRatio",True)]: d[c+"Rank"]=d.groupby("Candidate")[c].rank(ascending=asc,method="min")
  d["WithinCandidateComposite"]=d[["SilhouetteRank","DaviesBouldinRank","CalinskiHarabaszRank","ImbalanceRatioRank"]].mean(axis=1)+d.TinyWarning.astype(int)*3
  d["Shortlisted"]=False
  for name,g in d.groupby("Candidate"): d.loc[g.nsmallest(2,"WithinCandidateComposite").index,"Shortlisted"]=True
  return d
 def evaluate_multi_seed_stability(self,shortlisted:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
  runs=[]; ari_rows=[]; summaries=[]
  for _,row in shortlisted.loc[shortlisted.Shortlisted].iterrows():
   labels={}
   for seed in SEEDS:
    d=self.fit_candidate(row.Candidate,int(row.K),seed); labels[seed]=d["Labels"]
    runs.append({x:y for x,y in d.items() if x not in {"Labels","Model"}})
   matrix=pd.DataFrame(index=SEEDS,columns=SEEDS,dtype=float)
   for a in SEEDS:
    for b in SEEDS: matrix.loc[a,b]=adjusted_rand_score(labels[a],labels[b]); ari_rows.append({"Candidate":row.Candidate,"K":int(row.K),"SeedA":a,"SeedB":b,"ARI":matrix.loc[a,b]})
   vals=[matrix.loc[a,b] for i,a in enumerate(SEEDS) for b in SEEDS[i+1:]]
   summaries.append({"Candidate":row.Candidate,"K":int(row.K),"MeanARI":float(np.mean(vals)),"StdARI":float(np.std(vals)),"MinARI":float(np.min(vals)),"MaxARI":float(np.max(vals)),"StabilityStatus":"PASS" if np.mean(vals)>=0.80 and np.min(vals)>=0.60 else "WARNING"})
  return pd.DataFrame(runs),pd.DataFrame(ari_rows),pd.DataFrame(summaries)
 def build_selection_matrix(self,shortlisted,stability)->pd.DataFrame:
  d=shortlisted.merge(stability,on=["Candidate","K"],how="left"); d["WarningFlags"]=np.where(d.TinyWarning,"TINY_CLUSTER",""); d["InterpretabilityNote"]="Technical RFM-space partition; marketing names deferred"
  sub=d.loc[d.Shortlisted].copy()
  for c,asc in [("Silhouette",False),("DaviesBouldin",True),("CalinskiHarabasz",False),("ImbalanceRatio",True),("MeanARI",False)]: sub[c+"GlobalRank"]=sub[c].rank(ascending=asc,method="min")
  sub["SelectionScore"]=sub[[c+"GlobalRank" for c in ["Silhouette","DaviesBouldin","CalinskiHarabasz","ImbalanceRatio","MeanARI"]]].mean(axis=1)+sub.TinyWarning.astype(int)*4+sub.StabilityStatus.ne("PASS").astype(int)*3
  selected=sub.sort_values(["SelectionScore","Silhouette"],ascending=[True,False]).index[0]
  d["SelectionStatus"]="REJECTED"; d.loc[d.Shortlisted,"SelectionStatus"]="SHORTLISTED"; d.loc[selected,"SelectionStatus"]="SELECTED"
  if d.SelectionStatus.eq("SELECTED").sum()!=1: raise ClusterAcceptanceError("Selection must be unique")
  return d
 def tiny_cluster_review(self,selection:pd.DataFrame)->pd.DataFrame:
  rows=[]
  for _,r in selection.loc[selection.SelectionStatus.isin(["SHORTLISTED","SELECTED"])].iterrows():
   d=self.fit_candidate(r.Candidate,int(r.K)); labels=d["Labels"]; raw=self.rfm.copy(); raw["ClusterID"]=labels
   for cid,g in raw.groupby("ClusterID"):
    rows.append({"Candidate":r.Candidate,"K":int(r.K),"ClusterID":int(cid),"Customers":len(g),"Ratio":len(g)/len(raw),"MedianRecency":g.Recency.median(),"MedianFrequency":g.Frequency.median(),"MedianMonetary":g.Monetary.median(),"MaxFrequency":g.Frequency.max(),"MaxMonetary":g.Monetary.max(),"TinyWarning":len(g)/len(raw)<TINY_THRESHOLD,"Review":"WARNING_REVIEW_EXTREMES_NO_AUTOMATIC_REMOVAL" if len(g)/len(raw)<TINY_THRESHOLD else "PASS"})
  return pd.DataFrame(rows)
 def export_final(self,selection:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
  s=selection.loc[selection.SelectionStatus.eq("SELECTED")].iloc[0]; d=self.fit_candidate(s.Candidate,int(s.K),PRIMARY_SEED)
  out=self.rfm[["CustomerID","Recency","Frequency","Monetary"]].copy(); out["ClusterID"]=d["Labels"]; out["PreprocessingCandidate"]=s.Candidate; out["K"]=int(s.K); out["RandomState"]=PRIMARY_SEED; out["NInit"]=N_INIT; out["MaxIter"]=MAX_ITER
  cfg=pd.DataFrame([{"SelectedPreprocessing":s.Candidate,"SelectedK":int(s.K),"Init":"k-means++","NInit":N_INIT,"MaxIter":MAX_ITER,"RandomState":PRIMARY_SEED,"Algorithm":"lloyd","Silhouette":d["Silhouette"],"DaviesBouldin":d["DaviesBouldin"],"CalinskiHarabasz":d["CalinskiHarabasz"],"Inertia":d["Inertia"],"Iterations":d["Iterations"],"RuntimeSeconds":d["RuntimeSeconds"],"ClusterNamesCreated":"NO","PCACreated":"NO"}])
  return out,cfg
 def verify_input_integrity(self)->pd.DataFrame:
  rows=[]
  for file in ["rfm_customers.csv",*CANDIDATES.values()]:
   p=self.paths.input/file; self.hash_after[file]=self.sha256(p); rows.append({"File":file,"SHA256Before":self.hash_before[file],"SHA256After":self.hash_after[file],"Unchanged":self.hash_before[file]==self.hash_after[file],"SizeBytes":p.stat().st_size})
  d=pd.DataFrame(rows)
  if not d.Unchanged.all(): raise ClusterAcceptanceError("Input changed")
  return d
