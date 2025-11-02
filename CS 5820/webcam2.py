# Language: Python
# Simple program to open webcam and display the video stream

import cv2

# 0 is default camera, 1 is external usb cam
cam_index = 1

# Open the default camera (0 is usually the built-in webcam)
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    cv2.imshow('Webcam Stream', frame)
    
    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()