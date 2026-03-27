import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def process_data(data, features):
  
    np.random.seed(42)

   
    if 'Purchase Frequency' not in data.columns:
        data['Purchase Frequency'] = np.random.randint(1, 20, size=len(data))

    X = data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, data


def run_all_models(X_scaled, data, k=5):
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    data['KMeans'] = kmeans.fit_predict(X_scaled)

  
    hc = AgglomerativeClustering(n_clusters=k)
    data['Hierarchical'] = hc.fit_predict(X_scaled)

    
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    data['DBSCAN'] = dbscan.fit_predict(X_scaled)

    return data