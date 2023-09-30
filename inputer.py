from sklearn.impute import KNNImputer
import pandas as pd


def round_predict(data: pd.DataFrame, column) -> pd.DataFrame:
    data.loc[:, column] = data[column].round(0)
    return data


def input_miss_data(data: pd.DataFrame, n_neighbors=5):
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    x_imputed = knn_imputer.fit_transform(data)

    data.loc[:, data.columns] = x_imputed
    return data


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    all_data = pd.read_csv("data/Encoded.csv")
    X_data = all_data.drop(columns="Transported")
    y = all_data["Transported"]
    all_data = input_miss_data(X_data, 6)
    all_data.loc[:, "side"] = all_data["side"].round(0)

    all_data["Transported"] = y

    print(all_data.isna().sum())
    all_data.to_csv("data/Filled.csv", index=False)
