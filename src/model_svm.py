from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def train_svr(X_train, y_train, grid_search=False):
    """
    Entraîne un modèle SVR.
    - Si grid_search=True, utilise GridSearchCV pour optimiser les hyperparamètres.
    """
    if grid_search:
        param_grid = {
            "C": [1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "epsilon": [0.01, 0.1, 1]
        }
        grid = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=3, scoring="r2", verbose=2)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_, grid.best_score_
    else:
        model = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
        model.fit(X_train, y_train)
        return model, None, None
