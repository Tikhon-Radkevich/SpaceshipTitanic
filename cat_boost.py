import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

all_data = pd.read_csv("data/FilledAllData_2.csv")

scaler = MinMaxScaler((0, 1))
all_data[all_data.columns] = scaler.fit_transform(all_data)

train = all_data[all_data.type == 0]
train = train.drop(columns=["type"])
train.reset_index(inplace=True, drop=True)
test = all_data[all_data.type == 1]
test = test.drop(columns=["type"])
test.reset_index(inplace=True, drop=True)

test.drop(columns="Transported", inplace=True)
y = train["Transported"]
X = train.drop(columns="Transported")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.07, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.07, random_state=42)

final_model = CatBoostRegressor(learning_rate=0.0012, iterations=25000, depth=9)
final_model.fit(X_train, y_train, eval_set=(X_val, y_val))
# final_model.fit(X, y)

threshold = 0.5
y_pred = final_model.predict(X_test)
y_val_pred = final_model.predict(X_val)
y_val_pred = np.where(y_val_pred > threshold, 1, 0)
y_pred = np.where(y_pred > threshold, 1, 0)
print(f'Val: {accuracy_score(y_val, y_val_pred) * 100:.2f}%')
print(f'Test: {accuracy_score(y_test, y_pred) * 100:.2f}%')

# y_pred = final_model.predict(test)
# y_pred = np.where(y_pred > threshold, 1, 0)
# sub = pd.read_csv("data/sample_submission.csv")
# sub["Transported"] = y_pred.astype(bool)
# print(sub.head(30))
#
# sub.to_csv("data/result_1.csv", index=False)
