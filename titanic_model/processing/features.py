import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class preprocessing(BaseEstimator, TransformerMixin):
    """class to determine if the passenger is a baby"""

    def __init__(self):
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # is the passenger a baby
        X['is_baby'] = np.where(X['Age'] < 5, 1, 0)
        # was the passenger travelling alone
        X['alone'] = np.where((X['SibSp']==0) & (X['Parch']==0), 1, 0)
        # family member total
        X['family'] = X['SibSp'] + X['Parch']
        # create a title column
        X['title'] = X.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        X['title'] = X['title'].replace('Mlle', 'Miss')
        X['title'] = X['title'].replace('Ms', 'Miss')
        X['title'] = X['title'].replace('Mme', 'Mrs')
        X['title'] = X['title'].replace('Don', 'Mr')
        X['title'] = X['title'].replace('Dona', 'Mrs')
        # drop features not useful anymore
        X.drop(['SibSp', 'Parch', 'Name'], axis=1, inplace=True)
        return X