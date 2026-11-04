import pickle, numpy as np, pandas as pd
A="/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/"
G=pd.read_pickle(A+"exp63_graph_features.pkl"); D=pd.read_csv(A+"exp56_rows.csv")
W=pickle.load(open(A+"exp61_whitebox.pkl","rb")); X=W["X"].copy(); X["id"]=W["meta"].id.values
X["nzf"]=-X["noise: flip rate (-)"]; X=X.rename(columns={"reasoning ablation 'no reasoning': answer kept":"nrk","reasoning ablation 'first 1024 only': answer kept":"f1k","cf auto: state cosine, last quarter":"cfc","cf auto: trace log-lik change":"cfl"})
U=D[~D.finished.astype(bool)].merge(X[["id","nrk","f1k","cfc","cfl","nzf"]],on="id").merge(G,on="id").reset_index(drop=True)
y=U.robust.values.astype(int)
def auc(s,yy):
    p,n=s[yy==1],s[yy==0]; return float(((p[:,None]>n[None])+.5*(p[:,None]==n[None])).mean())
J=["block64 middle: mean jump (segments)","block64 middle: share > 20 back","block64 middle: share on previous 2","block64 late: mean jump (segments)"]
rk=lambda c:U.groupby("model")[c].rank(pct=True).values
print("Spearman (within-model ranks) with shipped/white-box features, unfinished rows:")
for j in J:
    print(f"  {j:40s}", " ".join(f"{f}={np.corrcoef(rk(j),rk(f))[0,1]:+.2f}" for f in ["fa","chk","fas","nrk","f1k","cfc","nzf","n sentences"]))
for m,g in U.groupby("model"): print(f"  per model {m:9s} n={len(g):2d} jump AUC {auc(g[J[0]].values,g.robust.values):.2f}")
S={"fa":1,"chk":1,"fas":1,"nrk":-1,"f1k":-1,"cfc":-1,"cfl":-1,"nzf":-1,J[0]:1}
R={f:rk(f)*s for f,s in S.items()}
for nm,fs in {"chk+fas":["chk","fas"],"chk+fas+jump":["chk","fas",J[0]],"chk+fas+5WB":["chk","fas","nrk","f1k","cfc","cfl","nzf"],"chk+fas+5WB+jump":["chk","fas","nrk","f1k","cfc","cfl","nzf",J[0]]}.items():
    print(f"  ensemble {nm:22s} unfinished AUC {auc(np.mean([R[f] for f in fs],0),y):.2f}")
