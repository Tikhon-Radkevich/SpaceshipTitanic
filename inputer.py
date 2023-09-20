from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np


def round_predict(data: pd.DataFrame, column) -> pd.DataFrame:
    data.loc[:, column] = data[column].round(0)
    return data


def input_miss_data(data: pd.DataFrame, n_neighbors=5):

    # Standardize data
    # scaler = MinMaxScaler((0, 1))
    # x_scaled = scaler.fit_transform(data)

    # Perform KNN imputation
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    x_imputed = knn_imputer.fit_transform(data)

    # data.loc[:, data.columns] = scaler.inverse_transform(x_imputed)
    data.loc[:, data.columns] = x_imputed
    return data


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    all_data = pd.read_csv("data/Dummies.csv")
    X_data = all_data.drop(columns="Transported")
    y = all_data["Transported"]
    all_data = input_miss_data(X_data, 6)
    all_data.loc[:, "side"] = all_data["side"].round(0)

    all_data["Transported"] = y

    print(all_data.isna().sum())
    all_data.to_csv("data/FilledAllData_2.csv", index=False)

    # test_pos = all_data["Transported"].isna()

    # probability_true = 0.1
    # test_pos = np.random.choice([True, False], size=X_data.shape[0], p=[probability_true, 1 - probability_true])
    # y_test = X_data[test_pos]["Transported"]
    # X_data.loc[test_pos, "Transported"] = None

    # all_data = input_miss_data(all_data, 6)
    # for c in ["CryoSleep", "VIP"]:
    #     all_data = round_predict(all_data, c)
    #
    # all_data.loc[test_pos, "Transported"] = None
    # all_data.to_csv("data/FilledAllData.csv", index=False)
    # print(all_data.head(7))

    # y_pred = predict[test_pos]["Transported"]
    # accuracy = accuracy_score(y_pred, y_test)
    # print(f"Accuracy: {accuracy * 100:.2f}%")


