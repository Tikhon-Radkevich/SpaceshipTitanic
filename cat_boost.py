import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def load_data(data_file):
    return pd.read_csv(data_file)


def scale_data(data):
    scaler = MinMaxScaler((0, 1))
    data[data.columns] = scaler.fit_transform(data)
    return data


def split_data(data):
    train = data[data.type == 0]
    train = train.drop(columns=["type"])
    train.reset_index(inplace=True, drop=True)
    test = data[data.type == 1]
    test = test.drop(columns=["type"])
    test.reset_index(inplace=True, drop=True)
    return train, test


def train_catboost_model(X_train, y_train, X_val, y_val, learning_rate=0.0021, iterations=9000, depth=8):
    model = CatBoostRegressor(learning_rate=learning_rate, iterations=iterations, depth=depth)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred = model.predict(X_test)
    y_pred_binary = np.where(y_pred > threshold, 1, 0)
    accuracy = accuracy_score(y_test, y_pred_binary)
    return accuracy


def predict_test_data(model, test_data, threshold=0.5):
    y_pred = model.predict(test_data)
    y_pred_binary = np.where(y_pred > threshold, 1, 0)
    return y_pred_binary


def save_submission(submission_file, predictions):
    sub = pd.read_csv("data/sample_submission.csv")
    sub["Transported"] = predictions.astype(bool)
    sub.to_csv(submission_file, index=False)


if __name__ == "__main__":
    # Load and preprocess the data
    all_data = load_data("data/Filled.csv")
    all_data = scale_data(all_data)

    # Split the data into train and test sets
    train_data, test_data = split_data(all_data)

    # Split the train data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_data.drop(columns="Transported"), train_data["Transported"],
                                                      test_size=0.07, random_state=42)

    # Train the CatBoost model
    model = train_catboost_model(X_train, y_train, X_val, y_val)

    # Evaluate the model on the validation set
    validation_accuracy = evaluate_model(model, X_val, y_val)
    print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')

    # Create the final model and make predictions on the test set
    final_model = train_catboost_model(train_data.drop(columns="Transported"), train_data["Transported"], X_val,
                                       y_val)  # Train on the entire train data

    # Predict on the test data
    test_predictions = predict_test_data(final_model, test_data.drop(columns="Transported"))

    # Save the submission file
    save_submission("data/submission.csv", test_predictions)

