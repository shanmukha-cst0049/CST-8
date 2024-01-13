import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
import time

# Function to load images and labels from a given directory
window_size = (1200, 700)


def load_data(dataframe, directory, target_size=(64, 64)):
    images = []
    labels = []
    for index, row in dataframe.iterrows():
        filename = row['filename']
        img_path = os.path.join(directory, filename)
        # Extract label from the dataframe
        label = row['label'].lower()
        labels.append(label)
        # Read as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)  # Resize the image
        images.append(img)  # Do not flatten the image

    return np.array(images), np.array(labels)


# Load the CSV file
csv_path = "training_set1.csv"
df = pd.read_csv(csv_path)

# Load training data
train_dir = "train1"
X, y = load_data(df, train_dir)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a simple SVM model
model = SVC()
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Open video file for video capture
video_path = "my_video3.mp4"
cap = cv2.VideoCapture(video_path)
max_duration_sec = 60  # Set the maximum duration in seconds
start_time = time.time()
batch_size = 5  # Adjust the batch size as needed
frame_buffer = []
frame_counter = 0

# Initialize accuracy variables
total_frames = 0
correct_predictions = 0

while time.time() - start_time < max_duration_sec:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video is finished

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (64, 64))  # Resize the frame

    # Increment frame counter
    frame_counter += 1

    # Append the frame to the buffer
    # Flatten the resized image
    frame_buffer.append(resized_frame.flatten())

    # Predict on every 5th frame or when the buffer reaches the batch size
    if frame_counter % 5 == 0 or len(frame_buffer) == batch_size:
        # Convert the frame buffer to a NumPy array and reshape
        frame_buffer_np = np.array(frame_buffer)
        frame_buffer_reshaped = frame_buffer_np.reshape(
            len(frame_buffer_np), -1)

        # Use the trained model to predict the action on the batch
        predicted_actions = model.predict(frame_buffer_reshaped)

        # Increment total_frames correctly
        total_frames += len(frame_buffer)

        # Calculate accuracy only if true_labels is not empty
        true_labels = y_test[total_frames - len(frame_buffer):total_frames]
        if len(true_labels) > 0:
            correct_predictions += np.sum(predicted_actions == true_labels)

            # Calculate accuracy
            accuracy = correct_predictions / total_frames

            # Display the predicted action and accuracy on the frame
            cv2.putText(frame, f"Action: {predicted_actions[0]}", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Accuracy: {accuracy:.2%}", (
                10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Action Recognition', cv2.resize(frame, window_size))

            # Reset the frame buffer
            frame_buffer = []

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
