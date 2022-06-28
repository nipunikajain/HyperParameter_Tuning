import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/pearl/Downloads/mobile.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max-depth": [1, 3, 5, 7],
        "criterion": ["gini", "entropy"]
    }

    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )
    model.fit(X, y)
