'''
Questo script è il principale per analisi, classificazione e clustering sul Dataset Heart Disease UCI.
'''

import pandas as pd
from utils.functions import preprocess_data, data_analysis


def main(df: pd.DataFrame) -> bool:
    
    print("Loading dataset...")
    # Import Dataset
    df = pd.read_csv('heart_disease_uci.csv')
    print("Dataset loaded successfully.")

    print("Start preprocessing the dataframe, basic checks and imputation")
    df = preprocess_data(df)
    if df.empty:
        print("Preprocessing failed.")
        return False


    data_analysis(df)


    print("Start preprocessing the dataframe, basic checks and imputation")
    df = preprocess_data(df)
    if df.empty:
        print("Preprocessing failed.")
        return False
    

    data_analysis(df)

    return True

if __name__ == "__main__":
    main()