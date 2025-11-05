# Language: Python
# Simple program to open webcam and display the video stream

import cv2
import time

# 0 is default camera, 1 is external usb cam
cam_index = 0

# Open the default camera (0 is usually the built-in webcam)
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_time = 1
prev_time = 0
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    cv2.imshow('Webcam Stream', frame)
    
    current_time = time.time()

    # grayscales, resizes, and saves a new image every second
    # change frame_time to adjust save time
    if current_time - prev_time >= frame_time:
        prev_time = current_time
        cv2.imwrite(f"gray_image.jpg", frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
