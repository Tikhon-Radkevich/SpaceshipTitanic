from sklearn.preprocessing import LabelEncoder
import pandas as pd


def l_encod(data: pd.DataFrame, column) -> pd.DataFrame:
    label_encoder = LabelEncoder()

    # Identify rows with missing values in the specified column
    none_mask = data[column].isna()

    data[column] = data[column].astype(str)
    data[column].fillna("Unknown", inplace=True)

    data[column] = label_encoder.fit_transform(data[column])

    data.loc[none_mask, column] = None

    return data


def to_bool(data: pd.DataFrame, column):
    data[column] = data[column].astype(bool)
    data[column] = data[column].astype(int)
    return data


def main():
    # List of columns to encode using LabelEncoder
    l_encod_list = ["HomePlanet", "Destination", "deck", "side", "AgeClass", "Movement"]

    # List of columns to convert to boolean
    to_bool_list = ["CryoSleep", "VIP", "Transported"]

    # Columns to drop
    drop_list = ["PassengerId", "Cabin", "Name", "FirstName", "Surname", "GroupId", "PassengerNum"]

    all_data = pd.read_csv("data/Preprocessed.csv")

    all_data.drop(columns=drop_list, inplace=True)

    # Encode categorical columns using LabelEncoder
    for c in l_encod_list:
        all_data = l_encod(all_data, c)

    # Convert selected columns to boolean
    for c in to_bool_list:
        all_data = to_bool(all_data, c)

    all_data.to_csv("data/EncodedAllData.csv", index=False)

    print(all_data.isna().sum())
    print(all_data.head(7))


if __name__ == "__main__":
    main()
