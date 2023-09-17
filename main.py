import pandas as pd

import gender_guesser.detector as gender

import name_to_gender
import plot


def get_data() -> pd.DataFrame:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    test = pd.read_csv("data/test.csv")
    train = pd.read_csv("data/train.csv")

    train["type"] = 0
    test["type"] = 1
    all_data = pd.concat([train, test], axis=0)

    # fill 0 where CryoSleep is True
    mask = all_data["CryoSleep"].isna()
    all_data.loc[mask, "CryoSleep"] = False
    columns_to_set_zero = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    all_data.loc[all_data["CryoSleep"], columns_to_set_zero] = 0
    all_data.loc[mask, "CryoSleep"] = None

    # From graph: I can set some HomePlanet and Destination by deck value
    all_data[["deck", "num", "side"]] = all_data["Cabin"].str.split("/", n=2, expand=True)

    con = (all_data["deck"].isin(["C", "T", "B", "A"])) & (all_data["HomePlanet"].isna())
    all_data.loc[con, "HomePlanet"] = "Europa"
    all_data.loc[(all_data["deck"].isin(["G"])) & (all_data["HomePlanet"].isna()), "HomePlanet"] = "Earth"

    con = (all_data["deck"].isin(["F", "E", "T"])) & (all_data["Destination"].isna())
    all_data.loc[con, "Destination"] = "TRAPPIST-1e"

    all_data.loc[all_data["HomePlanet"].isna(), "HomePlanet"] = "Earth"
    all_data.loc[all_data["Destination"].isna(), "Destination"] = "TRAPPIST-1e"
    # all_data.loc[all_data["deck"].isna(), "deck"] = "G"

    all_data[["GroupId", "PassengerNum"]] = all_data["PassengerId"].str.split("_", n=1, expand=True)
    all_data["GroupId"] = all_data["GroupId"].astype(int)
    all_data["PassengerNum"] = all_data["PassengerNum"].astype(int)
    all_data["GroupSize"] = all_data.groupby("GroupId").transform("size")

    print(all_data.head(20))
    print(all_data.isna().sum())

    # plot.age_to_transported(all_data.dropna())

    # siblings by Surname
    all_data[["FirstName", "Surname"]] = all_data["Name"].str.split(" ", n=1, expand=True)
    all_data["Siblings"] = all_data.groupby(["Surname", "GroupId"]).transform("size")

    # set gender
    n_to_gen_dict = name_to_gender.get()
    n_to_gen_dict.update({None: None})
    all_data["Gender"] = None
    all_data["Gender"] = all_data["FirstName"].map(n_to_gen_dict)

    mask = all_data["Gender"].isna()
    d = gender.Detector()
    all_data.loc[mask, "Gender"] = all_data.loc[mask, "FirstName"].map(d.get_gender)
    all_data["Gender"] = all_data["Gender"].map({
        "unknown": None,  # 439
        "male": "male",  # 8206
        "female": "female",  # 4325
        "mostly_male": "male",  # 124
        "mostly_female": "female",  # 41
        "andy": "female"  # 107
    })

    bins = [0, 1, 4, 12, 17, 42, float("inf")]
    labels = ["0", "1-4", "5-12", "13-17", "18-42", "43+"]
    # labels = [0, 1, 2, 3, 4, 5]
    all_data["AgeClass"] = pd.cut(all_data["Age"], bins=bins, labels=labels, right=False)

    all_data["AgeToTransported"] = all_data.groupby("AgeClass", observed=False)["Transported"].transform("mean")
    all_data["CryoSleepInGroup"] = all_data.groupby("GroupId", observed=False)["CryoSleep"].transform("mean")
    all_data["CryoSleepInFamily"] = all_data.groupby("Siblings", observed=False)["CryoSleep"].transform("mean")
    all_data["DestinationToTransported"] = all_data.groupby("Destination", observed=False)["Transported"].transform("mean")

    columns_to_sum = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    all_data["Expenses"] = all_data[columns_to_sum].sum(axis=1)
    all_data["AgeClassToExpenses"] = all_data.groupby("AgeClass", observed=False)["Expenses"].transform("mean")

    all_data["Movement"] = all_data["HomePlanet"] + " -> " + all_data["Destination"]
    all_data["MovementToCryoSleep"] = all_data.groupby("Movement", observed=False)["CryoSleep"].transform("mean")

    return all_data


if __name__ == "__main__":
    data = get_data()
    print(data.head(20))
