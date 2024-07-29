import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# Instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# Open webcam (0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Threshold for text detection score
threshold = 0.25

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

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

    # Display the resulting frame
    cv2.imshow('Text Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
