'''
Questo script è il principale per analisi, classificazione e clustering sul Dataset Heart Disease UCI.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import silhouette_score

# Import Dataset
df = pd.read_csv('heart_disease_uci.csv')

# Preprocessing
def preprocess_data(df):


    # Check on duplicated rows and visualize them if there are

    # Numeric variables will impute with their mean
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column] = df[column].fillna(df[column].mean())

    # Categorical variables will impute with their mode
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])


# Analysis of features


def Kmeans(df):
    #pca = PCA(n_components=0.85)
    features = [col for col in ["age", "trestbps", "chol", "thalach", "oldpeak"] if col in df.columns]
    X = df[features]

    # --- 2. Scaling delle variabili ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #X_pca = pca.fit_transform(X_scaled)

    inertia = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)


    plt.figure(figsize=(6,4))
    plt.plot(K_range, inertia, marker='o')
    plt.title("Metodo del gomito")
    plt.xlabel("Numero di cluster (k)")
    plt.ylabel("Inertia")
    plt.show()

    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df["cluster"] = labels

    fig = plt.figure(figsize=(8, 20))

    # Plot originale in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # KMeans
    kmeans3d = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=df["cluster"], cmap='Set1', s=50, edgecolor='k')
    ax1.set_title("Dati Originali in 3D")
    ax1.set_xlabel("chol")
    ax1.set_ylabel("trestbps")
    ax1.set_zlabel("age")
    ax1.grid(True)

    plt.tight_layout()
    plt.show()

def main(df):
    preprocess_data(df)
    Kmeans(df)

    return True

if __name__ == "__main__":
    main(df)