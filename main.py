'''
Questo script è il principale per analisi, classificazione e clustering sul Dataset Heart Disease UCI.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import Dataset
df = pd.read_csv('heart_disease_uci.csv')

# Preprocessing
def preprocess_data(df):
    print(df.info())
    print(df.describe())

    # Check on duplicated rows and visualize them if there are
    print("Duplicated rows:", df.duplicated().sum())
    if df.duplicated().any():
        print("Duplicated rows found:")
        print(df[df.duplicated()])

    # Numeric variables will impute with their mean
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column] = df[column].fillna(df[column].mean())

    # Categorical variables will impute with their mode
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    print("Data preprocessing complete.")

# Analysis of features
def data_analysis(df):
    
    trestbps_by_age = df.groupby('age')['trestbps'].mean().reset_index()
    chol_by_age = df.groupby('age')['chol'].mean().reset_index()

    # Visualize distributions based on age and gender of blood pressure and cholesterol levels
    # Plot blood pressure based on age and based on gender in a single plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    sns.barplot(ax=axes[0], data=trestbps_by_age, x='trestbps', y='age', alpha=0.7)
    axes[0].set_title('Distribuzione trestbps vs age')
    axes[0].set_xlabel('trestbps')
    axes[0].set_ylabel('age')

    sns.barplot(ax=axes[1], data=chol_by_age, x='chol', y='age', alpha=0.7)
    axes[1].set_title('Distribuzione chol vs age')
    axes[1].set_xlabel('chol')
    axes[1].set_ylabel('age')

    plt.tight_layout()
    plt.show()

    # Standardize numeric features
    scaler = StandardScaler()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

def main(df):
    
    preprocess_data(df)
    data_analysis(df)

    return True

if __name__ == "__main__":
    main(df)