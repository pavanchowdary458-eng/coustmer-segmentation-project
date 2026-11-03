from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
OUT = ROOT / "outputs"
DATA.parent.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

def create_sample_data(path):
    rng = np.random.default_rng(21)
    rows = []
    for customer_no in range(1, 181):
        customer = f"CUST{customer_no:04d}"
        order_count = int(rng.integers(3, 25))
        for _ in range(order_count):
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
            value = round(float(rng.lognormal(mean=4.2, sigma=.65)), 2)
            rows.append((customer, date.date().isoformat(), value))
    pd.DataFrame(rows, columns=["customer_id", "order_date", "order_value"]).to_csv(path, index=False)

if not DATA.exists():
    create_sample_data(DATA)

df = pd.read_csv(DATA, parse_dates=["order_date"])
required = {"customer_id", "order_date", "order_value"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
df = df.drop_duplicates().dropna(subset=list(required))
df = df[df["order_value"] > 0]
if df["customer_id"].nunique() < 7:
    raise ValueError("At least 7 unique customers are required for cluster evaluation.")

snapshot = df["order_date"].max() + pd.Timedelta(days=1)
rfm = df.groupby("customer_id").agg(
    recency=("order_date", lambda x: (snapshot - x.max()).days),
    frequency=("order_date", "count"),
    monetary=("order_value", "sum")
).reset_index()

features = rfm[["recency", "frequency", "monetary"]].copy()
features = np.log1p(features.clip(lower=0))
scaled = StandardScaler().fit_transform(features)

scores = []
max_k = min(6, len(rfm) - 1)
for k in range(2, max_k + 1):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    scores.append((k, silhouette_score(scaled, labels)))
best_k, best_score = max(scores, key=lambda item: item[1])

model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm["cluster"] = model.fit_predict(scaled)
summary = rfm.groupby("cluster").agg(
    customers=("customer_id", "nunique"),
    avg_recency_days=("recency", "mean"),
    avg_orders=("frequency", "mean"),
    avg_customer_value=("monetary", "mean")
).round(2).reset_index()

coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
plt.figure(figsize=(8, 5))
plt.scatter(coords[:, 0], coords[:, 1], c=rfm["cluster"], alpha=.75)
plt.title(f"Customer Segments (K-Means, k={best_k})")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.tight_layout()
plt.savefig(OUT / "customer_segments_pca.png", dpi=160)
plt.close()

rfm.to_csv(OUT / "customer_rfm_segments.csv", index=False)
summary.to_csv(OUT / "segment_summary.csv", index=False)
pd.DataFrame(scores, columns=["k", "silhouette_score"]).to_csv(OUT / "cluster_evaluation.csv", index=False)

print(f"Selected k={best_k}; silhouette score={best_score:.3f}")
print(summary.to_string(index=False))
print(f"Outputs saved to {OUT}")
