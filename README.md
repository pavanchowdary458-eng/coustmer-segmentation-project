# E-Commerce Customer Segmentation

A customer analytics project that groups shoppers by purchasing behavior using Recency, Frequency, and Monetary (RFM) features and K-Means clustering.

## Objectives
- Calculate customer-level RFM metrics.
- Standardize behavioral features and evaluate candidate cluster counts.
- Select a cluster count using silhouette score.
- Visualize customer groups with PCA.
- Export segment-level summaries for business interpretation.

## Technology
Python, pandas, NumPy, scikit-learn, Matplotlib.

## Project structure
```
.
├── data/transactions.csv
├── src/segment_customers.py
├── outputs/
└── requirements.txt
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/segment_customers.py
```

If the input CSV is missing, the script creates a reproducible synthetic transaction dataset. It then exports customer RFM values, segment summaries, cluster evaluation scores, and a PCA chart to `outputs/`.

## Important interpretation note
Cluster numbers are arbitrary labels, not rankings. Review the segment summary before assigning business names or proposing campaigns. Results from synthetic data are for learning and demonstration only, not evidence about real customers.

## Author
Pavan Nalabothu