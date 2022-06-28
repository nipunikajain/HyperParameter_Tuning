import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection, decomposition, preprocessing, pipeline

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/pearl/Downloads/mobile.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)

    classifier = pipeline.Pipeline(
        [
            ("scaling" , scl),
            ("pca", pca),
            ("rf", rf)
        ]
    )

    param_grid = {
        "pca__n_components": np.arange(5, 10),
        "rf__n_estimators": np.arange(100, 1500, 100),
        "rf__max-depth": np.arange(1, 20),
        "rf__criterion": ["gini", "entropy"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )
    model.fit(X, y)
