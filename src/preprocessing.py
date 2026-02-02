import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_columns(df):
    """
    Sépare les colonnes numériques et catégorielles.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    return numeric_cols, categorical_cols

def preprocess_data(df, target="depenses"):
    """
    Prépare les données pour l'entraînement :
    - Sépare X et y
    - Encode les variables catégorielles
    - Normalise les variables numériques
    """
    X = df.drop(target, axis=1)
    y = df[target]

    numeric_cols, categorical_cols = split_columns(X)

    # Encodage
    X = pd.get_dummies(X, columns=categorical_cols)

    # Normalisation
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler
