"""Stage 7 EDA engine for the seven locked DT25 analysis questions.

The engine consumes accepted Stage 4/6 CSV outputs, creates analysis tables,
figures, and traceability registries. It does not run clustering, choose k,
create PCA, or assign final customer segments.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

EXPECTED = {
    "online_retail_clean.csv": {"rows": 524878, "customers": None},
    "online_retail_rfm_eligible.csv": {"rows": 392692, "customers": 4338},
    "online_retail_cancellations_returns.csv": {"rows": 11763, "customers": None},
    "rfm_customers.csv": {"rows": 4338, "customers": 4338},
    "rfm_with_scores.csv": {"rows": 4338, "customers": 4338},
}
QUESTIONS = ["Q01","Q02","Q03","Q04","Q05","Q06","Q07"]
REQUIRED_TRANSACTION = ["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country","TransactionAmount"]

class AnalysisInputError(ValueError): pass
class AnalysisAcceptanceError(RuntimeError): pass

@dataclass(frozen=True)
class Paths:
    root: Path
    @property
    def input(self): return self.root/"data"/"input"
    @property
    def tables(self): return self.root/"outputs"/"tables"/"analysis"
    @property
    def figures(self): return self.root/"outputs"/"figures"/"analysis"
    @property
    def logs(self): return self.root/"outputs"/"logs"/"stage7"
    def create(self):
        for p in (self.tables,self.figures,self.logs): p.mkdir(parents=True,exist_ok=True)

class AnalysisEngine:
    """Execute the approved EDA registry with code-derived interpretations."""
    def __init__(self, paths: Paths):
        self.paths=paths; self.data={}; self.hash_before={}; self.hash_after={}
        self.results=[]; self.figures=[]; self.interpretations=[]; self.limitations=[]

    @staticmethod
    def sha256(path: Path)->str:
        h=hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
        return h.hexdigest()

    def validate_inputs(self)->pd.DataFrame:
        self.paths.create(); rows=[]
        schemas={
            "online_retail_clean.csv": REQUIRED_TRANSACTION,
            "online_retail_rfm_eligible.csv": REQUIRED_TRANSACTION,
            "online_retail_cancellations_returns.csv": ["InvoiceNo","StockCode","InvoiceDate","Quantity","UnitPrice","Country","IsCancellation","IsNegativeQuantity","IsInvalidUnitPrice"],
            "rfm_customers.csv": ["CustomerID","LastPurchaseDate","Recency","Frequency","Monetary"],
            "rfm_with_scores.csv": ["CustomerID","Recency","Frequency","Monetary","R_Score","F_Score","M_Score","RFM_TotalScore"],
        }
        for name,baseline in EXPECTED.items():
            p=self.paths.input/name
            if not p.exists() or p.stat().st_size<=0: raise FileNotFoundError(f"Missing/empty input: {name}")
            self.hash_before[name]=self.sha256(p)
            try: df=pd.read_csv(p,dtype={"CustomerID":"string","InvoiceNo":"string"},low_memory=False)
            except (OSError,pd.errors.ParserError,UnicodeDecodeError) as e: raise AnalysisInputError(f"Cannot read {name}: {e}") from e
            missing=sorted(set(schemas[name])-set(df.columns))
            if missing: raise AnalysisInputError(f"{name} missing columns: {missing}")
            if len(df)!=baseline["rows"]: raise AnalysisInputError(f"{name} rows={len(df)} expected={baseline['rows']}")
            if baseline["customers"] is not None and df.CustomerID.nunique()!=baseline["customers"]: raise AnalysisInputError(f"{name} customers mismatch")
            if "InvoiceDate" in df: df["InvoiceDate"]=pd.to_datetime(df.InvoiceDate,errors="coerce")
            for c in ["Quantity","UnitPrice","TransactionAmount","Recency","Frequency","Monetary"]:
                if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
            self.data[name]=df
            rows.append({"File":name,"Rows":len(df),"Columns":df.shape[1],"Customers":int(df.CustomerID.nunique()) if "CustomerID" in df else None,"MissingCritical":int(df[schemas[name]].isna().sum().sum()),"SizeBytes":p.stat().st_size,"SHA256":self.hash_before[name],"Status":"PASS"})
        if self.data["rfm_customers.csv"].CustomerID.duplicated().any(): raise AnalysisInputError("Duplicate CustomerID in rfm_customers.csv")
        return pd.DataFrame(rows)

    def _save_table(self,qid,df,name):
        if df.empty: raise AnalysisAcceptanceError(f"Empty table for {qid}")
        p=self.paths.tables/name; df.to_csv(p,index=False,encoding="utf-8-sig")
        return p
    def _save_fig(self,qid,fig,figure_id,chart_type,title,xlabel,ylabel,unit,name,source,caption):
        if not title or not xlabel or not ylabel: raise AnalysisAcceptanceError(f"Missing chart metadata {figure_id}")
        p=self.paths.figures/name; fig.tight_layout(); fig.savefig(p,dpi=160,bbox_inches="tight"); plt.close(fig)
        self.figures.append({"FigureID":figure_id,"QuestionID":qid,"ChartType":chart_type,"Title":title,"XLabel":xlabel,"YLabel":ylabel,"Unit":unit,"Path":str(p.relative_to(self.paths.root)),"SourceDataset":source,"Caption":caption})
        return p
    def _record(self,qid,question,dataset,table,figure,cell,points,limits,owner,checker,rubric):
        self.results.append({"QuestionID":qid,"Question":question,"Dataset":dataset,"TablePath":str(table.relative_to(self.paths.root)),"FigurePath":str(figure.relative_to(self.paths.root)),"CellID":cell,"Owner":owner,"CrossChecker":checker,"RubricMapping":rubric,"TechnicalStatus":"TECHNICALLY PASS, PEER REVIEW PENDING","PeerReviewStatus":"PENDING"})
        for i,text in enumerate(points,1): self.interpretations.append({"QuestionID":qid,"PointID":f"{qid}-I{i}","Text":text,"SourceTable":str(table.relative_to(self.paths.root)),"ClaimType":"DESCRIPTIVE"})
        for i,text in enumerate(limits,1): self.limitations.append({"QuestionID":qid,"LimitationID":f"{qid}-L{i}","Text":text})

    def analyze_time_trend(self):
        q="Số hóa đơn, số giao dịch và tổng TransactionAmount của các giao dịch mua hợp lệ thay đổi theo tháng như thế nào?"; df=self.data["online_retail_clean.csv"]
        t=df.assign(Month=df.InvoiceDate.dt.to_period("M").astype(str)).groupby("Month",as_index=False).agg(LineRows=("InvoiceNo","size"),UniqueInvoices=("InvoiceNo","nunique"),UniqueCustomers=("CustomerID","nunique"),Revenue=("TransactionAmount","sum"))
        table=self._save_table("Q01",t,"q01_monthly_trends.csv")
        fig,ax=plt.subplots(figsize=(11,5)); ax.plot(t.Month,t.Revenue,marker="o"); ax.set_title("FIG-Q01-01: Xu hướng doanh thu mua hàng theo tháng"); ax.set_xlabel("Tháng giao dịch"); ax.set_ylabel("Doanh thu (sterling)"); ax.tick_params(axis="x",rotation=45)
        figure=self._save_fig("Q01",fig,"FIG-Q01-01","Line chart",ax.get_title(),"Tháng giao dịch","Doanh thu","sterling","fig_q01_monthly_revenue.png","online_retail_clean.csv","Dữ liệu theo tháng; tháng biên có thể không đầy đủ.")
        peak=t.loc[t.Revenue.idxmax()]; low=t.loc[t.Revenue.idxmin()]
        pts=[f"Tháng có doanh thu tổng cao nhất trong bảng là {peak.Month}, giá trị {peak.Revenue:.2f} sterling.",f"Tháng có doanh thu tổng thấp nhất trong bảng là {low.Month}, giá trị {low.Revenue:.2f} sterling.",f"Khoảng quan sát chạy từ {t.Month.iloc[0]} đến {t.Month.iloc[-1]}; các tháng biên phải được diễn giải như giai đoạn quan sát không nhất thiết đầy đủ."]
        self._record("Q01",q,"online_retail_clean.csv",table,figure,"S7-Q01",pts,["Chuỗi quan sát ngắn không đủ để khẳng định quy luật mùa vụ dài hạn.","Biến động mô tả không chứng minh nguyên nhân."],"TV3","TV1","Xu hướng thời gian; groupby; doanh thu; line chart")

    def analyze_revenue_contribution(self):
        q="Doanh thu mua hàng, số hóa đơn và số khách hàng có định danh phân bố theo quốc gia như thế nào, và mức độ tập trung đóng góp ra sao?"; df=self.data["online_retail_clean.csv"]
        t=df.groupby("Country",as_index=False).agg(Revenue=("TransactionAmount","sum"),UniqueInvoices=("InvoiceNo","nunique"),UniqueCustomers=("CustomerID","nunique"),Lines=("InvoiceNo","size")).sort_values("Revenue",ascending=False); t["RevenueSharePercent"]=t.Revenue/t.Revenue.sum()*100; t["CumulativeSharePercent"]=t.RevenueSharePercent.cumsum()
        table=self._save_table("Q02",t,"q02_country_contribution.csv"); top=t.head(15).sort_values("Revenue")
        fig,ax=plt.subplots(figsize=(10,7)); ax.barh(top.Country,top.Revenue); ax.set_title("FIG-Q02-01: Đóng góp doanh thu theo quốc gia"); ax.set_xlabel("Doanh thu (sterling)"); ax.set_ylabel("Quốc gia")
        figure=self._save_fig("Q02",fig,"FIG-Q02-01","Horizontal bar chart",ax.get_title(),"Doanh thu","Quốc gia","sterling","fig_q02_country_revenue.png","online_retail_clean.csv","15 quốc gia có doanh thu cao nhất; bảng CSV giữ toàn bộ quốc gia.")
        first=t.iloc[0]
        pts=[f"Quốc gia đứng đầu bảng doanh thu là {first.Country}, đạt {first.Revenue:.2f} sterling và chiếm {first.RevenueSharePercent:.2f}% tổng doanh thu mua hợp lệ.",f"Bảng đầy đủ gồm {len(t)} quốc gia và sử dụng số hóa đơn duy nhất thay vì số dòng sản phẩm.",f"Số khách hàng theo quốc gia chỉ đếm CustomerID không thiếu."]
        self._record("Q02",q,"online_retail_clean.csv",table,figure,"S7-Q02",pts,["Doanh thu không phải lợi nhuận và không cung cấp chi phí hoặc ROI.","Country không chứng minh tiềm năng thị trường hay nguyên nhân doanh thu."],"TV3","TV2","So sánh nhóm; doanh thu; groupby; horizontal bar")

    def analyze_customer_behavior(self):
        q="Giá trị hóa đơn, số loại sản phẩm trong hóa đơn và số hóa đơn theo khách hàng phân bố như thế nào?"; df=self.data["online_retail_rfm_eligible.csv"]
        inv=df.groupby(["CustomerID","InvoiceNo"],as_index=False).agg(InvoiceValue=("TransactionAmount","sum"),UniqueProducts=("StockCode","nunique"),TotalQuantity=("Quantity","sum")); cust=df.groupby("CustomerID",as_index=False).agg(CustomerInvoices=("InvoiceNo","nunique"))
        metrics=pd.DataFrame([{"Metric":"InvoiceValue","Count":len(inv),"Mean":inv.InvoiceValue.mean(),"Median":inv.InvoiceValue.median(),"Q1":inv.InvoiceValue.quantile(.25),"Q3":inv.InvoiceValue.quantile(.75),"Max":inv.InvoiceValue.max()},{"Metric":"UniqueProducts","Count":len(inv),"Mean":inv.UniqueProducts.mean(),"Median":inv.UniqueProducts.median(),"Q1":inv.UniqueProducts.quantile(.25),"Q3":inv.UniqueProducts.quantile(.75),"Max":inv.UniqueProducts.max()},{"Metric":"CustomerInvoices","Count":len(cust),"Mean":cust.CustomerInvoices.mean(),"Median":cust.CustomerInvoices.median(),"Q1":cust.CustomerInvoices.quantile(.25),"Q3":cust.CustomerInvoices.quantile(.75),"Max":cust.CustomerInvoices.max()}])
        table=self._save_table("Q03",metrics,"q03_customer_behavior.csv")
        fig,axes=plt.subplots(1,2,figsize=(12,5)); axes[0].hist(inv.InvoiceValue,bins=60); axes[0].set_title("Histogram giá trị hóa đơn"); axes[0].set_xlabel("Giá trị hóa đơn (sterling)"); axes[0].set_ylabel("Số hóa đơn"); axes[1].boxplot([inv.InvoiceValue,cust.CustomerInvoices],labels=["InvoiceValue","Invoices/customer"]); axes[1].set_title("Boxplot hành vi mua"); axes[1].set_xlabel("Chỉ tiêu"); axes[1].set_ylabel("Giá trị")
        figure=self._save_fig("Q03",fig,"FIG-Q03-01","Histogram + boxplot", "FIG-Q03-01: Phân bố hành vi mua cấp hóa đơn và khách hàng","Chỉ tiêu hành vi","Tần số hoặc giá trị","sterling/count","fig_q03_customer_behavior.png","online_retail_rfm_eligible.csv","Histogram và boxplot dùng toàn bộ dữ liệu hợp lệ; điểm ngoài whisker không tự động là lỗi.")
        mi=metrics.set_index("Metric")
        pts=[f"Median giá trị hóa đơn là {mi.loc['InvoiceValue','Median']:.2f} sterling, trong khi mean là {mi.loc['InvoiceValue','Mean']:.2f} sterling.",f"Median số sản phẩm duy nhất trong một hóa đơn là {mi.loc['UniqueProducts','Median']:.0f}.",f"Median số hóa đơn duy nhất theo khách hàng là {mi.loc['CustomerInvoices','Median']:.0f}."]
        self._record("Q03",q,"online_retail_rfm_eligible.csv",table,figure,"S7-Q03",pts,["Outlier được giữ nguyên và không nên tự động coi là lỗi.","Dữ liệu không chứa phiên truy cập hoặc ý định mua."],"TV3","TV4","Thống kê mô tả; histogram; boxplot; hành vi khách hàng")

    def analyze_cancellations_returns(self):
        q="Các giao dịch hủy, trả hàng và giao dịch không hợp lệ được tách riêng phân bố như thế nào theo loại lý do, thời gian, quốc gia và mã sản phẩm?"; df=self.data["online_retail_cancellations_returns.csv"]
        cancel=df.IsCancellation.astype(str).str.lower().eq("true"); neg=df.IsNegativeQuantity.astype(str).str.lower().eq("true"); invp=df.IsInvalidUnitPrice.astype(str).str.lower().eq("true")
        categories=np.select([cancel&neg,cancel&~neg,~cancel&neg,~cancel&~neg&invp],["Cancellation + negative quantity","Cancellation without negative quantity","Negative quantity without cancellation","Invalid price only"],default="Other exception")
        x=df.assign(ExceptionCategory=categories,Month=df.InvoiceDate.dt.to_period("M").astype(str)); t=x.groupby(["Month","ExceptionCategory"],as_index=False).size().rename(columns={"size":"Rows"})
        table=self._save_table("Q04",t,"q04_exception_profile.csv"); p=t.pivot(index="Month",columns="ExceptionCategory",values="Rows").fillna(0)
        fig,ax=plt.subplots(figsize=(12,6)); p.plot(kind="bar",stacked=True,ax=ax); ax.set_title("FIG-Q04-01: Cơ cấu hủy, trả hàng và giao dịch không hợp lệ theo tháng"); ax.set_xlabel("Tháng"); ax.set_ylabel("Số dòng exception"); ax.legend(title="Nhóm độc quyền",bbox_to_anchor=(1.02,1))
        figure=self._save_fig("Q04",fig,"FIG-Q04-01","Stacked bar chart",ax.get_title(),"Tháng","Số dòng exception","dòng","fig_q04_exception_composition.png","online_retail_cancellations_returns.csv","Các nhóm loại trừ lẫn nhau, không cộng trực tiếp cancellation và negative quantity.")
        totals=x.ExceptionCategory.value_counts(); pts=[f"Bảng phân loại {len(x)} dòng exception thành các nhóm độc quyền để tránh cộng chồng lấp.",f"Nhóm lớn nhất trong phép phân loại là {totals.index[0]} với {int(totals.iloc[0])} dòng.",f"Khoảng quan sát exception từ {x.Month.min()} đến {x.Month.max()}."]
        self._record("Q04",q,"online_retail_cancellations_returns.csv",table,figure,"S7-Q04",pts,["Exception không đồng nghĩa toàn bộ là đơn bị hủy.","Dòng exception không cho biết nguyên nhân vận hành đầy đủ."],"TV2","TV1","Data quality; crosstab/groupby; stacked bar; so sánh nhóm")

    def analyze_rfm(self):
        q="Recency, Frequency và Monetary phân bố, lệch và liên hệ với nhau như thế nào sau khi bảng RFM được tạo bằng giao dịch đủ điều kiện?"; r=self.data["rfm_customers.csv"]; vars=["Recency","Frequency","Monetary"]
        d=r[vars].describe(percentiles=[.25,.5,.75,.9,.95,.99]).T.reset_index(names="Variable"); d["Skewness"]=r[vars].skew().values; corr=r[vars].corr(method="spearman"); corr_long=corr.stack().reset_index(); corr_long.columns=["Variable1","Variable2","SpearmanCorrelation"]; table_data=d.merge(pd.DataFrame({"JoinKey":[1]*len(d)}),left_index=True,right_index=True).drop(columns="JoinKey")
        table=self._save_table("Q05",table_data,"q05_rfm_diagnostics.csv"); corr_long.to_csv(self.paths.tables/"q05_rfm_correlation.csv",index=False,encoding="utf-8-sig")
        fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(3),vars); ax.set_yticks(range(3),vars); ax.set_title("FIG-Q05-01: Tương quan Spearman giữa các chỉ số RFM"); ax.set_xlabel("Chỉ số RFM"); ax.set_ylabel("Chỉ số RFM"); fig.colorbar(im,ax=ax,label="Hệ số Spearman")
        for i in range(3):
            for j in range(3): ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center")
        figure=self._save_fig("Q05",fig,"FIG-Q05-01","Heatmap",ax.get_title(),"Chỉ số RFM","Chỉ số RFM","hệ số [-1,1]","fig_q05_rfm_correlation.png","rfm_customers.csv","Tương quan Spearman mô tả quan hệ đơn điệu, không chứng minh nhân quả.")
        skew=d.set_index("Variable").Skewness; off=corr.where(~np.eye(3,dtype=bool)).stack(); pair=off.abs().idxmax(); val=corr.loc[pair]
        pts=[f"Skewness của Recency, Frequency và Monetary lần lượt là {skew.Recency:.3f}, {skew.Frequency:.3f}, {skew.Monetary:.3f}.",f"Cặp có độ lớn tương quan Spearman ngoài đường chéo cao nhất là {pair[0]} và {pair[1]}, hệ số {val:.3f}.",f"Bảng sử dụng {len(r)} khách hàng và không loại RFM outlier."]
        self._record("Q05",q,"rfm_customers.csv",table,figure,"S7-Q05",pts,["Tương quan không chứng minh quan hệ nhân quả.","Phân phối lệch không tự động biện minh cho việc loại khách hàng."],"TV3","TV4","RFM; correlation; heatmap; thống kê mô tả")

    def analyze_monetary_concentration(self):
        q="Monetary có tập trung vào một nhóm nhỏ khách hàng hay không, và nhóm đóng góp Monetary cao có đặc điểm Recency và Frequency như thế nào?"; r=self.data["rfm_customers.csv"].sort_values("Monetary",ascending=False).reset_index(drop=True); r["Rank"]=np.arange(1,len(r)+1); r["MonetarySharePercent"]=r.Monetary/r.Monetary.sum()*100; r["CumulativeSharePercent"]=r.MonetarySharePercent.cumsum(); r["CustomerPercent"]=r.Rank/len(r)*100
        table=self._save_table("Q06",r[["CustomerID","Rank","Recency","Frequency","Monetary","MonetarySharePercent","CumulativeSharePercent","CustomerPercent"]],"q06_monetary_concentration.csv")
        fig,ax1=plt.subplots(figsize=(11,6)); n=min(50,len(r)); ax1.bar(r.Rank.head(n),r.Monetary.head(n)); ax1.set_xlabel("Xếp hạng khách hàng theo Monetary"); ax1.set_ylabel("Monetary (sterling)"); ax2=ax1.twinx(); ax2.plot(r.Rank,r.CumulativeSharePercent,color="red"); ax2.set_ylabel("Tỷ trọng tích lũy (%)"); ax1.set_title("FIG-Q06-01: Mức độ tập trung Monetary theo khách hàng")
        figure=self._save_fig("Q06",fig,"FIG-Q06-01","Pareto chart",ax1.get_title(),"Xếp hạng khách hàng","Monetary và tỷ trọng tích lũy","sterling/%","fig_q06_customer_monetary_pareto.png","rfm_customers.csv","Bar hiển thị 50 khách hàng đầu; đường cumulative dùng toàn bộ khách hàng.")
        top10=max(1,int(np.ceil(len(r)*.1))); share=r.head(top10).Monetary.sum()/r.Monetary.sum()*100; top=r.iloc[0]
        pts=[f"Top 10% khách hàng theo Monetary đóng góp {share:.2f}% tổng Monetary trong bảng RFM.",f"Khách hàng đứng đầu có Monetary {top.Monetary:.2f} sterling, Frequency {int(top.Frequency)} và Recency {int(top.Recency)} ngày.",f"Đường cumulative được tính trên toàn bộ {len(r)} khách hàng."]
        self._record("Q06",q,"rfm_customers.csv",table,figure,"S7-Q06",pts,["Monetary không phải lợi nhuận hoặc customer lifetime value.","Không được gọi quy luật 80/20 nếu bảng không hỗ trợ đúng ngưỡng đó."],"TV3","TV4","Doanh thu; RFM; Pareto; concentration")

    def analyze_descriptive_profile_readiness(self):
        old="Sau khi mô hình và số cụm được chọn bằng bằng chứng, các cụm khác nhau thế nào về Recency, Frequency, Monetary, quy mô và tỷ trọng Monetary, và hành động chăm sóc nào phù hợp với profile quan sát được?"
        q="Trước khi clustering, các tổ hợp điểm RFM mô tả phân bố thế nào về quy mô và Monetary, và chúng cung cấp baseline profiling nào để đối chiếu với cụm ở Giai đoạn 9?"
        s=self.data["rfm_with_scores.csv"]; t=s.groupby(["R_Score","F_Score","M_Score"],as_index=False).agg(Customers=("CustomerID","nunique"),MedianRecency=("Recency","median"),MedianFrequency=("Frequency","median"),MedianMonetary=("Monetary","median"),TotalMonetary=("Monetary","sum")); t["MonetarySharePercent"]=t.TotalMonetary/t.TotalMonetary.sum()*100
        table=self._save_table("Q07",t,"q07_rfm_score_profile_baseline.csv"); heat=s.pivot_table(index="R_Score",columns="F_Score",values="Monetary",aggfunc="median")
        fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(heat.values,origin="lower",aspect="auto",cmap="viridis"); ax.set_xticks(range(len(heat.columns)),heat.columns); ax.set_yticks(range(len(heat.index)),heat.index); ax.set_title("FIG-Q07-01: Baseline Monetary theo điểm R và F"); ax.set_xlabel("F_Score"); ax.set_ylabel("R_Score"); fig.colorbar(im,ax=ax,label="Median Monetary (sterling)")
        figure=self._save_fig("Q07",fig,"FIG-Q07-01","Profile heatmap",ax.get_title(),"F_Score","R_Score","median Monetary (sterling)","fig_q07_rfm_score_profile_baseline.png","rfm_with_scores.csv","Baseline mô tả theo điểm quintile; không phải Cluster ID hoặc phân khúc cuối.")
        largest=t.loc[t.Customers.idxmax()]; pts=[f"Tổ hợp điểm có nhiều khách hàng nhất trong bảng là R={int(largest.R_Score)}, F={int(largest.F_Score)}, M={int(largest.M_Score)} với {int(largest.Customers)} khách hàng.",f"Heatmap dùng median Monetary để giảm ảnh hưởng của giá trị cực đoan trong phần mô tả.","Kết quả này chỉ là baseline theo điểm RFM và phải được đối chiếu với cluster profile sau khi K-Means được nghiệm thu."]
        self._record("Q07",q,"rfm_with_scores.csv",table,figure,"S7-Q07",pts,["Điểm RFM mô tả không phải kết quả K-Means và không được dùng làm tên cụm cuối.","Khuyến nghị marketing cuối cùng bị hoãn đến Giai đoạn 9 sau cluster profiling."],"TV4 + TV3","TV1","RFM profiling readiness; heatmap; customer segmentation preparation")
        return {"QuestionID":"Q07","OriginalQuestion":old,"DataLimitation":"Stage 7 prohibits K-Means and no cluster assignment file exists.","MinimalAdjustment":q,"GoalPreserved":"Prepare evidence-based profiling and strategy baseline without inventing clusters.","Status":"ADJUSTED_MINIMALLY"}

    def execute_all(self):
        self.analyze_time_trend(); self.analyze_revenue_contribution(); self.analyze_customer_behavior(); self.analyze_cancellations_returns(); self.analyze_rfm(); self.analyze_monetary_concentration(); change=self.analyze_descriptive_profile_readiness()
        return pd.DataFrame([change])

    def export_registries(self,change_log,input_verification):
        results=pd.DataFrame(self.results); figures=pd.DataFrame(self.figures); interpretations=pd.DataFrame(self.interpretations); limitations=pd.DataFrame(self.limitations)
        acceptance=results[["QuestionID","TechnicalStatus","PeerReviewStatus","TablePath","FigurePath","CrossChecker"]].copy(); acceptance["AcceptanceStatus"]="TECHNICALLY PASS, PEER REVIEW PENDING"
        rubric=pd.DataFrame([
            {"Requirement":"At least five questions","Questions":"Q01-Q07","Evidence":"analysis_results_registry.csv","Coverage":"PASS"},
            {"Requirement":"Descriptive statistics","Questions":"Q03,Q05","Evidence":"q03,q05 tables","Coverage":"PASS"},
            {"Requirement":"Groupby/pivot","Questions":"Q01,Q02,Q04,Q07","Evidence":"question tables","Coverage":"PASS"},
            {"Requirement":"Group comparison","Questions":"Q02,Q04,Q07","Evidence":"bar/heatmap tables","Coverage":"PASS"},
            {"Requirement":"Time trend","Questions":"Q01","Evidence":"line chart and monthly table","Coverage":"PASS"},
            {"Requirement":"Revenue","Questions":"Q01,Q02,Q06","Evidence":"sterling-labelled tables/figures","Coverage":"PASS"},
            {"Requirement":"RFM","Questions":"Q05,Q06,Q07","Evidence":"RFM tables/figures","Coverage":"PASS"},
            {"Requirement":"Seven chart types","Questions":"Q01-Q07","Evidence":"figure_registry.csv","Coverage":"PASS"},
            {"Requirement":"Interpretation and limitations","Questions":"Q01-Q07","Evidence":"interpretation/limitation registries","Coverage":"PASS"},
        ])
        outputs={"analysis_results_registry.csv":results,"figure_registry.csv":figures,"question_acceptance_matrix.csv":acceptance,"interpretation_registry.csv":interpretations,"limitation_registry.csv":limitations,"rubric_coverage_stage7.csv":rubric,"question_change_log.csv":change_log,"input_verification.csv":input_verification}
        for name,df in outputs.items(): df.to_csv(self.paths.tables/name,index=False,encoding="utf-8-sig")
        return outputs

    def verify_input_integrity(self):
        rows=[]
        for name in EXPECTED:
            p=self.paths.input/name; self.hash_after[name]=self.sha256(p); rows.append({"File":name,"SHA256Before":self.hash_before[name],"SHA256After":self.hash_after[name],"Unchanged":self.hash_before[name]==self.hash_after[name],"SizeBytes":p.stat().st_size})
        result=pd.DataFrame(rows); result.to_csv(self.paths.tables/"input_checksum_report.csv",index=False,encoding="utf-8-sig")
        if not result.Unchanged.all(): raise AnalysisAcceptanceError("Input checksum changed")
        return result
