'''
This script contains all functions to preprocess and analyze the heart disease dataset.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function preprocesses the input DataFrame by performing the following steps:
    1. Displaying basic information and statistics about the dataset.
    2. Checking for and visualizing duplicated rows.
    3. Imputing missing values for numeric columns with their mean.
    4. Imputing missing values for categorical columns with their mode.
    '''
    # Display basic information and statistics
    print(df.info())
    print(df.describe())

    # Check on duplicated rows and visualize them if there are
    print("Duplicated rows:", df.duplicated().sum())
    if df.duplicated().any():
        print("Duplicated rows found:")
        print(df[df.duplicated()])

    try:

        # Numeric variables will impute with their mean
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            df[column] = df[column].fillna(df[column].mean())

        # Categorical variables will impute with their mode
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].fillna(df[column].mode()[0])

    except Exception as e:
        print("Error during data preprocessing:", e)
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Analysis of features
def data_analysis(df: pd.DataFrame) -> bool:
    '''
    This function helps visualize the relationships between some features in the dataset.
    Relationships between age with the blood pressure at rest and cholesterol levels.
    '''

    try:
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

        # Visualize correlations
        plt.figure(figsize=(14, 12))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

        return True
    
    except Exception as e:
        print("Error during data analysis:", e)
        return False 

def prepare_data(
        df: pd.DataFrame,
        learning_method: str
    ) -> tuple:
    '''
    This function prepares the input DataFrame for analysis by performing the following steps:
    1. Standardizing numeric features.
    2. Prepare features and target for supervised or unsupervised learning.
    '''

    # Drop 'id' column before standardize
    df = df.drop(columns=['id'])

    # Take only numeric features
    features = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']]

    # Standardize numeric features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    if learning_method == 'supervised':
        # Prepare target variable for supervised learning
        df['target'] = (df['num'] > 0).astype(int)
        target = df['target']

        return (features, target)
    
    if learning_method == 'unsupervised':
        return (features, None)