"""Largest tree leaves, and forecasts above 90% on one outcome (docs/decisions/0010 Finding 1).
Saved models only; never loads 2026 rows. Prints results; about 2 minutes.

Usage: PYTHONPATH=. python scripts/leaf_check.py
"""
import joblib, numpy as np, pandas as pd
from scipy.special import softmax
from murphy.forecast import features as ft
from murphy.forecast.climatology import OUT_DIR
from murphy.forecast.gbm import OUT, INFO, aligned
from murphy.forecast.sample import CLASSES
pd.set_option("display.width",200,"display.max_columns",30,"display.max_rows",100)
names=None
for name in ["final","grid_lr0.05_leaves63","grid_lr0.1_leaves63"]:
    m=joblib.load(OUT/f"{name}.joblib")["model"]
    rec=[]
    for i,pred in enumerate(m._predictors,1):
        for k,t in enumerate(pred):
            n=t.nodes; leaf=n["is_leaf"].astype(bool)
            for v,c in zip(n["value"][leaf],n["count"][leaf]):
                rec.append((i,m.classes_[k],float(v),int(c)))
    d=pd.DataFrame(rec,columns=["round","cls","value","count"])
    print("==",name,"leaves",len(d),"| |value|>20:",(d.value.abs()>20).sum(),"| first round |value|>20:",d[d.value.abs()>20]["round"].min())
    print("by count of leaf, |value|>20:",d[d.value.abs()>20]["count"].describe()[["min","50%","max"]].to_dict())
    print(d.reindex(d.value.abs().sort_values(ascending=False).index).head(6).to_string(index=False))
    b=d[d.value.abs()>20].copy(); b["sign"]=np.sign(b.value)
    print(b.groupby(["cls","sign"]).size().to_string())
    if name=="final":
        print("|value|>20 up to round 75:",(d[d["round"]<=75].value.abs()>20).sum(), " by class:",d[(d["round"]<=75)&(d.value.abs()>20)].cls.value_counts().to_dict())

# sharp forecasts of the final model on the 2025 sample, rounds 75 vs last
r=pd.read_parquet(OUT_DIR/"rows.parquet",columns=INFO)
h=pd.util.hash_pandas_object(r.flight_id,index=False).to_numpy()
keep=(r.split=="tune").to_numpy()&(h%8==0)
X=pd.concat([pd.read_parquet(ft.OUT/f"{g}.parquet").drop(columns="rid")[keep] for g in "abcde"],axis=1)
ri=r[keep].reset_index(drop=True); del r
s=joblib.load(OUT/"final.joblib"); m=s["model"]
X["carrier"]=pd.Categorical(X.carrier,categories=s["carriers"]).codes.astype("float32"); X.loc[X.carrier<0,"carrier"]=np.nan
Y=(ri.outcome.to_numpy()[:,None]==np.array(CLASSES)[None,:]).astype(float)
base=None
for i,raw in enumerate(m.staged_decision_function(X),1):
    if i not in (75,m.n_iter_): continue
    p=aligned(softmax(raw,axis=1),m.classes_); l=((p-Y)**2).sum(axis=1)
    top=p.argmax(1); sharp=p.max(1)>.9
    t=pd.DataFrame({"pred":np.array(CLASSES)[top][sharp],"hit":Y[np.arange(len(Y)),top][sharp]}).groupby("pred").hit.agg(["size","mean"])
    print(f"round {i} brier {l.mean():.4f}; sharp forecasts by predicted class\n",t.round(3).to_string())
    print(f" sharp rows by lead:",ri.lead_h[sharp].value_counts().to_dict())
    print(f" brier without sharp rows {l[~sharp].mean():.4f}; share of total loss from sharp rows {l[sharp].sum()/l.sum():.4f}")
    if base is None: base=l.copy()
    else:
        d=np.sort(l-base)[::-1]; n=len(d)
        print(f"rise 75->{i}: {d.sum()/n:.4f}; top 0.1% rows {d[:n//1000].sum()/n:.4f}; top 1% rows {d[:n//100].sum()/n:.4f}; rest {d[n//100:].sum()/n:.4f}")
