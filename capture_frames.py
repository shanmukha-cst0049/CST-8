import cv2
import numpy as np
from sklearn.svm import SVC
import os
import joblib


class ActionRecognition:
    def __init__(self):
        self.model_path = "action_model.joblib"
        self.data_path = "action_data.npz"
        self.model = self.load_model() or self.train_model()

    def train_model(self):
        if os.path.exists(self.data_path):
            X, y = np.load(self.data_path)['X'], np.load(self.data_path)['y']
        else:
            print("Training data not found. Please run the training script.")
            return None

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = SVC()
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        joblib.dump(model, self.model_path)
        return model

    def load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def capture_frames(self):
        cap = cv2.VideoCapture(0)
        frame_buffer = []
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray, (64, 64))
            frame_buffer.append(resized_frame.flatten())
            if len(frame_buffer) == 5:  # Process every 5 frames
                frame_buffer_np = np.array(frame_buffer)
                frame_buffer_reshaped = frame_buffer_np.reshape(
                    len(frame_buffer_np), -1)
                predicted_actions = self.model.predict(frame_buffer_reshaped)
                print("Predicted Actions:", predicted_actions)
                frame_buffer = []
        cap.release()

    def recognize_action(self, image_data):
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Resize the frame
        resized_frame = cv2.resize(frame, (64, 64))

        # Predict the action
        predicted_action = self.model.predict(
            resized_frame.flatten().reshape(1, -1))[0]

        return predicted_action
