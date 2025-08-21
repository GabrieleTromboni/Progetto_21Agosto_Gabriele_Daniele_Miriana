'''
Questo script è il principale per analisi, classificazione e clustering sul Dataset Heart Disease UCI.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


# Supervised Learning
def supervised_learning(
        features: pd.DataFrame,
        target: pd.Series
    ) -> bool:
    '''
    Function to do supervised learning.
    '''

    try:
        # Check if features and target are not empty
        if features.empty or target.empty:
            print("Features or target are empty.")
            return False
        # split for train and test
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=y_train, random_state=42)

        #Logistic Regression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        #Valutazione dei modelli
        print(classification_report(y_test, y_pred))
        print('F1', f1_score(y_test, y_pred), 'Precision', precision_score(y_test, y_pred), 'Recall', recall_score(y_test, y_pred))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
        plt.title('Confusion Matrix')
        plt.show()

        return True

    except Exception as e:
        print("Error occurred during supervised learning:", e)
        return False