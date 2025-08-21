'''
Questo script è il principale per analisi, classificazione e clustering sul Dataset Heart Disease UCI.
'''

import pandas as pd
from utils.functions import preprocess_data, data_analysis, prepare_data
import argparse

def main(args) -> bool:
    
    print("Loading dataset...")
    # Import Dataset
    df = pd.read_csv(args.dataset_path)
    print("Dataset loaded successfully.")

    print("Start preprocessing the dataframe, basic checks and imputation")
    df = preprocess_data(df)
    if df.empty:
        print("Preprocessing failed.")
        return False
    print("Data preprocessing completed successfully.")

    print("Start data analysis for some visualizations")
    data_analysis(df)
    print("Data analysis completed successfully.")

    print("Start preparing the data for learning")
    X, y = prepare_data(df, args.learning_method)
    
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Heart Disease UCI Dataset Analysis and Learning")
    parser.add_argument("--dataset_path", type=str, default="heart_disease_uci.csv", help="Path to the dataset CSV file", required=True)
    parser.add_argument("--learning_method", type=str, choices=["supervised", "unsupervised"], default="supervised", help="Learning method to use", required=True)
    args = parser.parse_args()

    main()