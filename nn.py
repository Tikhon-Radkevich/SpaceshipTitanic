import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras import optimizers


# Load your dataset
all_data = pd.read_csv("data/FilledAllData_2.csv")

scaler = MinMaxScaler((0, 1))
all_data[all_data.columns] = scaler.fit_transform(all_data)

# Separate the data into train and test sets
train = all_data[all_data.type == 0]
train = train.drop(columns=["type"])
train.reset_index(inplace=True, drop=True)
test = all_data[all_data.type == 1]
test = test.drop(columns=["type"])
test.reset_index(inplace=True, drop=True)

# Drop the "Transported" column from the test set
test.drop(columns="Transported", inplace=True)

# Prepare the training data
y = train["Transported"]
X = train.drop(columns="Transported")

# Split the data into train, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.07, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.07, random_state=42)


# Create a Sequential model
model = Sequential()

# Add input layer with BatchNormalization
model.add(BatchNormalization(input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation="relu"))

model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))


model.add(BatchNormalization())
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.0))  # Adjust dropout rate

# Add output layer with sigmoid activation
model.add(Dense(1, activation="sigmoid"))
# Compile the model

custom_optimizer = optimizers.Adam(learning_rate=0.00002)
model.compile(loss="binary_crossentropy", optimizer=custom_optimizer, metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=16, batch_size=16)

# Make predictions
y_pred = model.predict(X_test)
y_val_pred = model.predict(X_val)

# Apply threshold for binary classification
threshold = 0.5
y_val_pred = np.where(y_val_pred > threshold, 1, 0)
y_pred = np.where(y_pred > threshold, 1, 0)

# Calculate accuracy
print(f'Validation Accuracy: {accuracy_score(y_val, y_val_pred) * 100:.2f}%')
print(f'Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
