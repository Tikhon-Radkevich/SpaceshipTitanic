import pandas as pd
import numpy as np


def fill_missing(row, mean_values):
    if np.all(row == 0):
        return row.fillna(0)
    else:
        return row.fillna(mean_values)


def process_data(all_data):
    mask = all_data["CryoSleep"].isna()
    all_data.loc[mask, "CryoSleep"] = False
    columns_to_set_zero = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    all_data.loc[all_data["CryoSleep"], columns_to_set_zero] = 0
    all_data.loc[mask, "CryoSleep"] = None

    columns_to_fill = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    mean_values = all_data[columns_to_fill].replace(0, np.nan).mean()

    for c in columns_to_fill:
        con = ((all_data["Age"] < 13) & (all_data[c].isna()))
        all_data.loc[con, c] = 0

    all_data[columns_to_fill] = all_data[columns_to_fill].apply(lambda row: fill_missing(row, mean_values), axis=1)

    columns_to_sum = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    all_data["Expenses"] = all_data[columns_to_sum].sum(axis=1)

    all_data["CryoSleep"] = np.float16(all_data["CryoSleep"] * 1.0)
    all_data.loc[(all_data.CryoSleep.isna() == True) & (all_data.Expenses == 0.0), ["CryoSleep"]] = 1.0
    all_data.loc[(all_data.CryoSleep.isna() == True) & (all_data.Age <= 12), ["CryoSleep"]] = 1.0
    all_data.loc[(all_data.CryoSleep.isna() == True) & (all_data["Expenses"] > 0.0), ["CryoSleep"]] = 0.0

    # Siblings by Surname
    all_data[["FirstName", "Surname"]] = all_data["Name"].str.split(" ", n=1, expand=True)

    all_data = all_data.merge(all_data.loc[all_data.Age > 12, ['Surname', 'Age']].dropna().groupby('Surname').agg(_Age=pd.NamedAgg('Age', np.median)),
                  how='left', left_on='Surname', right_on='Surname', suffixes=('', ''))
    all_data.loc[(all_data.Age.isna() == True) & ((all_data.Expenses > 0.0) | (all_data.CryoSleep == 0)), ['Age']] = all_data._Age
    all_data['Age'] = all_data['Age'].fillna(all_data.Age.median())

    all_data['Is_Child'] = np.where(all_data.Age <= 12, 1, 0)
    all_data = all_data.drop(['_Age'], axis=1)

    # Assuming that members of the same family have the same VIP ID
    all_data = all_data.merge(all_data[['VIP', 'Surname']].groupby('Surname')['VIP'].max(),
                              how='left', left_on='Surname', right_on='Surname', suffixes=('', '_'))
    all_data['VIP'] = all_data['VIP'].fillna(all_data.VIP_)
    all_data['VIP'] = all_data['VIP'].fillna(0.0)
    all_data = all_data.drop(["VIP_"], axis=1)

    # Deck-based HomePlanet and Destination
    all_data[["deck", "num", "side"]] = all_data["Cabin"].str.split("/", n=2, expand=True)

    con = (all_data["deck"].isin(["C", "T", "B", "A"])) & (all_data["HomePlanet"].isna())
    all_data.loc[con, "HomePlanet"] = "Europa"
    all_data.loc[(all_data["deck"].isin(["G"])) & (all_data["HomePlanet"].isna()), "HomePlanet"] = "Earth"

    con = (all_data["deck"].isin(["F", "E", "T"])) & (all_data["Destination"].isna())
    all_data.loc[con, "Destination"] = "TRAPPIST-1e"

    all_data.loc[all_data["HomePlanet"].isna(), "HomePlanet"] = "Earth"
    all_data.loc[all_data["Destination"].isna(), "Destination"] = "TRAPPIST-1e"

    all_data[["GroupId", "PassengerNum"]] = all_data["PassengerId"].str.split("_", n=1, expand=True)
    all_data["GroupId"] = all_data["GroupId"].astype(int)
    all_data["PassengerNum"] = all_data["PassengerNum"].astype(int)
    all_data["GroupSize"] = all_data.groupby("GroupId").transform("size")

    most_common_values = all_data.groupby("GroupId").agg({
        "deck": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "num": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "side": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    }).reset_index()
    all_data = all_data.merge(most_common_values, how="left", on="GroupId", suffixes=("", "_most_common"))
    con = all_data["Cabin"].isna()
    for column in ["deck", "num", "side"]:
        all_data.loc[con, column] = all_data.loc[con, f"{column}_most_common"]
        all_data.drop(columns=f"{column}_most_common", inplace=True)

    all_data.loc[all_data["deck"].isna(), "deck"] = "G"

    most_common_values = all_data.groupby("GroupId").agg({
        "Surname": lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).reset_index()
    all_data = all_data.merge(most_common_values, how="left", on="GroupId", suffixes=("", "_most_common"))
    con = all_data["Surname"].isna()
    all_data.loc[con, "Surname"] = all_data.loc[con, "Surname_most_common"]
    all_data.drop(columns="Surname_most_common", inplace=True)

    singles_count = all_data["Surname"].isna().sum()
    all_data.loc[all_data["Surname"].isna(), "Surname"] = np.array([f"Unknown_{i}" for i in range(singles_count)])

    all_data.drop(columns=["Cabin", "Name", "FirstName", "num"], inplace=True)

    all_data["Siblings"] = all_data.groupby(["Surname", "GroupId"]).transform("size")

    bins = [0, 1, 4, 12, 17, 42, float("inf")]
    labels = ["0", "1-4", "5-12", "13-17", "18-42", "43+"]
    all_data["AgeClass"] = pd.cut(all_data["Age"], bins=bins, labels=labels, right=False)

    all_data["AgeToTransported"] = all_data.groupby("AgeClass", observed=False)["Transported"].transform("mean")
    all_data["CryoSleepInGroup"] = all_data.groupby("GroupId", observed=False)["CryoSleep"].transform("mean")
    all_data["CryoSleepInFamily"] = all_data.groupby("Siblings", observed=False)["CryoSleep"].transform("mean")
    all_data["DestinationToTransported"] = all_data.groupby("Destination", observed=False)["Transported"].transform(
        "mean")
    all_data["AgeClassToExpenses"] = all_data.groupby("AgeClass", observed=False)["Expenses"].transform("mean")

    all_data["Movement"] = all_data["HomePlanet"] + " -> " + all_data["Destination"]
    all_data["MovementToCryoSleep"] = all_data.groupby("Movement", observed=False)["CryoSleep"].transform("mean")

    return all_data


def main():
    test = pd.read_csv("data/test.csv")
    train = pd.read_csv("data/train.csv")

    train["type"] = 0
    test["type"] = 1
    all_data = pd.concat([train, test], axis=0)

    data = process_data(all_data)
    data.to_csv("data/Preprocessed.csv", index=False)


if __name__ == "__main__":
    main()
