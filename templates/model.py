import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
import joblib


def load_data(data_path):
    data = np.load(data_path)
    X, y = data['X'], data['y']
    return X, y


def preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    return img.flatten()


def train_model(X, y):
    model = SVC()
    model.fit(X.reshape(X.shape[0], -1), y)
    return model


def save_model(model, model_path):
    joblib.dump(model, model_path)


if __name__ == "__main__":
    data_path = "action_data.npz"
    model_path = "action_model.joblib"

    if os.path.exists(data_path):
        X, y = load_data(data_path)
    else:
        print("Training data not found. Please provide a dataset.")
        # Add code to load your dataset and preprocess it

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    save_model(model, model_path)
