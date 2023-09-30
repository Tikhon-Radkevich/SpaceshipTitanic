import pandas as pd
from sklearn.preprocessing import MinMaxScaler


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def bool_encode(data: pd.DataFrame, column) -> pd.DataFrame:
    # Convert the column to string type
    data[column] = data[column].astype(str)

    # Define mapping for boolean encoding
    mapping = {"False": 0, "0.0": 0, "True": 1, "1.0": 1, "P": 0, "S": 1, "nan": None}

    # Apply the mapping to the column
    data[column] = data[column].map(mapping)

    return data


def one_hot_encode(data: pd.DataFrame, column) -> pd.DataFrame:
    # Use get_dummies to perform one-hot encoding on the specified column
    data = pd.get_dummies(data, columns=[column], prefix=column[0])

    return data


def main():
    # Load the preprocessed data
    all_data = pd.read_csv("data/Preprocessed.csv")

    # Drop specified columns
    all_data.drop(columns=["PassengerId", "Surname", "GroupId", "PassengerNum"], inplace=True)

    # Lists of columns to apply encoding and scaling
    bool_encoder_list = ["CryoSleep", "VIP", "Transported", "side"]
    one_hot_encoder_list = ["HomePlanet", "Destination", "deck", "AgeClass", "Movement"]
    scaler_list = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Expenses", "GroupSize",
                   "Siblings", "CryoSleepInFamily", "DestinationToTransported", "AgeClassToExpenses",
                   "MovementToCryoSleep", "AgeToTransported", "CryoSleepInGroup"]

    # Apply boolean encoding
    for c in bool_encoder_list:
        all_data = bool_encode(all_data, c)

    # Apply one-hot encoding
    for c in one_hot_encoder_list:
        all_data = one_hot_encode(all_data, c)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler((0, 1))

    # Scale the specified columns
    all_data[scaler_list] = scaler.fit_transform(all_data[scaler_list])

    # Save the encoded and scaled data to a CSV file
    all_data.to_csv("data/Encoded.csv", index=False)

    print(all_data.isna().sum())


if __name__ == "__main__":
    main()
