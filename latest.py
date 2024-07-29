import random
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import keyboard
import mediapipe as mp

# Define the LSTM model for action recognition
action_recognition_model = Sequential()
action_recognition_model.add(LSTM(units=128, return_sequences=True, input_shape=(10, 99))) # 10 timesteps, 99 features (33 keypoints * 3 (x, y, z))
action_recognition_model.add(LSTM(units=64))
action_recognition_model.add(Dense(8, activation='softmax')) # 8 possible actions

# Compile the LSTM model
action_recognition_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the YOLOv8 model
model = YOLO("weights/yolov8n.pt", "v8")

# Define the object detection function
def object_detection():
    with open("utils/coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")
    return class_list

class_list = object_detection()

detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_list))]

frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

person_tracks = {}
person_features = {}

plt.ion()  # turn on interactive mode

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_pose_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    if result.pose_landmarks:
        keypoints = []
        for landmark in result.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        return keypoints
    else:
        return []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            font = cv2.FONT_HERSHEY_COMPLEX

            # Extract person detections
            if clsID == 0:  # person class ID
                person_id = f"person_{i}"
                if person_id not in person_tracks:
                    person_tracks[person_id] = []
                    person_features[person_id] = []

                person_tracks[person_id].append(box.xyxy.numpy()[0])

                # Extract features from person detections using pose estimation
                cropped_person = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                keypoints = extract_pose_keypoints(cropped_person)
                if keypoints:
                    person_features[person_id].append(keypoints)

                    # Feed features into the LSTM model
                    if len(person_features[person_id]) > 10:  # 10 timesteps
                        person_features_array = np.array(person_features[person_id][-10:])  # take the last 10 timesteps
                        person_features_array = person_features_array.reshape((1, 10, len(keypoints)))  # reshape for LSTM input
                        action_probs = action_recognition_model.predict(person_features_array)
                        action_id = np.argmax(action_probs)
                        action_label = ["walking", "running", "jumping", "sitting", "standing", "lying", "crouching", "unknown"][action_id]
                        print(f"Action for {person_id}:", action_label)

                        cv2.putText(
                            frame,
                            action_label,
                            (int(bb[0]), int(bb[1]) - 30),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.01)
    plt.clf()

    if keyboard.is_pressed('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # turn off interactive mode
plt.show()
