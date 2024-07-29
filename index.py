import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import random
import face_recognition
import os
import math
from collections import deque
from ultralytics import YOLO

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2))
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2))

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

    def recognize_face(self, frame, face_location):
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        face_encodings = face_recognition.face_encodings(rgb_face_image)
        if face_encodings:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])
                if float(confidence) < 90:
                    return 'Unknown'
                return name.split(".")[0]
        return 'Person'

def object_detection_with_face_recognition():
    
    face_recognition_obj = FaceRecognition()

    
    with open("utils/coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    detection_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(class_list))]

    model = YOLO("weights/yolov8n.pt", "v8")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    
    reader = easyocr.Reader(['en'], gpu=False)
    threshold = 0.25

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
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                if class_list[int(clsID)] == 'person':
                    
                    face_locations = face_recognition.face_locations(frame)
                    recognized_name = 'Person'
                    
                    for face_location in face_locations:
                        top, right, bottom, left = face_location
                        if (int(bb[0]) < right < int(bb[2]) and int(bb[1]) < bottom < int(bb[3])):
                            recognized_name = face_recognition_obj.recognize_face(frame, face_location)
                            break 

                    print(f'Detected: {recognized_name}')
                    text_to_display = recognized_name
                else:
                    print(f'Detected: {class_list[int(clsID)]}')
                    text_to_display = class_list[int(clsID)]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    text_to_display + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

        # Detect text on frame
        text_ = reader.readtext(frame)
        
        # Draw bbox and text
        for t_, t in enumerate(text_):
            bbox, text, score = t

            if score > threshold:
                # Draw bounding box
                cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
                # Put text
                cv2.putText(frame, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
                print(f"Detected text: {text}")

        cv2.imshow("Object Detection with Face Recognition", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

object_detection_with_face_recognition()
