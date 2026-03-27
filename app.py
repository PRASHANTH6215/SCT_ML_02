import sys
import os
sys.path.append(os.path.dirname(__file__))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import process_data, run_all_models

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation using ML")

file = st.file_uploader(" Upload CSV File", type=["csv"])

if file:
    data = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    
    features = st.multiselect(
        " Select Features for Clustering",
        data.columns,
        default=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    )

    
    k = st.slider("Select Number of Clusters (K-Means & Hierarchical)", 2, 10, 5)

    if len(features) > 1:
        
        X_scaled, data = process_data(data, features)

        
        result = run_all_models(X_scaled, data, k)

        st.subheader(" Clustered Data")
        st.dataframe(result)

        
        st.subheader("📈 3D Cluster Visualization (K-Means)")

        if len(features) >= 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(
                X_scaled[:, 0],
                X_scaled[:, 1],
                X_scaled[:, 2],
                c=result['KMeans']
            )

            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_zlabel(features[2])

            st.pyplot(fig)
        else:
            st.warning("⚠️ Select at least 3 features for 3D visualization")

        
        st.subheader(" Cluster Analysis (K-Means)")

        summary = result.groupby('KMeans')[features].mean()
        st.dataframe(summary)

        st.success("✅ Clustering Completed Successfully!")

else:
    st.info("⬆️ Upload a CSV file to begin")