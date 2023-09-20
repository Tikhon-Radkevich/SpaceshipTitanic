import pandas as pd

from sklearn.preprocessing import MinMaxScaler


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def bool_encod(data: pd.DataFrame, column) -> pd.DataFrame:
    data[column] = data[column].astype(str)
    mapping = {"False": 0, "0.0": 0, "True": 1, "1.0": 1, "P": 0, "S": 1, "nan": None}
    data[column] = data[column].map(mapping)
    return data


def one_hot_encod(data: pd.DataFrame, column) -> pd.DataFrame:
    data = pd.get_dummies(data, columns=[column], prefix=column[0])
    return data


all_data = pd.read_csv("data/Preprocessed.csv")
all_data.drop(columns=["PassengerId", "Surname", "GroupId", "PassengerNum"], inplace=True)

bool_encoder_list = ["CryoSleep", "VIP", "Transported", "side"]
one_hot_encoder_list = ["HomePlanet", "Destination", "deck", "AgeClass", "Movement"]
scaler_list = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Expenses", "GroupSize", "Siblings",
               "CryoSleepInFamily", "DestinationToTransported", "AgeClassToExpenses", "MovementToCryoSleep",
               "AgeToTransported", "CryoSleepInGroup"]

for c in bool_encoder_list:
    all_data = bool_encod(all_data, c)

for c in one_hot_encoder_list:
    unique_columns = list(map(lambda x: f"x_{x}", all_data[c].unique()))
    all_data = pd.get_dummies(all_data, columns=[c], prefix="x")
    for x_column in unique_columns:
        all_data = bool_encod(all_data, x_column)


scaler = MinMaxScaler((0, 1))
all_data[scaler_list] = scaler.fit_transform(all_data[scaler_list])
all_data.to_csv("data/Dummies.csv", index=False)

print(all_data.isna().sum())
