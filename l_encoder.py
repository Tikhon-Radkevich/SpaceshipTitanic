from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from main import get_data
from inputer import input_miss_data


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def l_encod(data: pd.DataFrame, column) -> pd.DataFrame:
    label_encoder = LabelEncoder()
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


l_encod_list = ["HomePlanet", "Destination", "deck", "side", "AgeClass", "Movement"]
to_bool_list = ["CryoSleep", "VIP", "Transported"]
drop_list = ["PassengerId", "Cabin", "Name", "FirstName", "Surname"]
all_data = pd.read_csv("data/Preprocessed.csv")

all_data.drop(columns=["PassengerId", "Surname", "GroupId", "PassengerNum"], inplace=True)
print(all_data.head(6))

for c in l_encod_list:
    all_data = l_encod(all_data, c)

for c in to_bool_list:
    all_data = to_bool(all_data, c)

# all_data.drop(columns=drop_list, inplace=True)
all_data.to_csv("data/EncodedAllData_1.csv", index=False)
print(all_data.isna().sum())
print(all_data.head(7))
