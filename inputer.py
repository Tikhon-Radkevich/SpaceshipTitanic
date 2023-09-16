from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def input_miss_data(data: pd.DataFrame, n_neighbors=5):

    # Standardize data
    scaler = MinMaxScaler((0, 1))
    x_scaled = scaler.fit_transform(data)

    # Perform KNN imputation
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    x_imputed = knn_imputer.fit_transform(x_scaled)

    data[data.columns] = scaler.inverse_transform(x_imputed)
    return data
