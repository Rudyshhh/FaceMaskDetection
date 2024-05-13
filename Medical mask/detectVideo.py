import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from imutils.video import FPS
from cv2 import VideoWriter

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load face mask detection model
model = load_model("mask_detector.model")

# Initialize video capture
cap = cv2.VideoCapture('Input Videos/Test_video1.mp4')

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = VideoWriter('output.mp4', fourcc, 30, (width, height))

# Initialize FPS counter
fps = FPS().start()

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = net.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through faces
    for (x, y, w, h) in faces:
        # Crop face from frame
        face = frame[y:y+h, x:x+w]

        # Resize face to 224x224 for face mask detection model
        face = cv2.resize(face, (224, 224))

        # Convert face to array and preprocess
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        # Make prediction
        pred = model.predict(face)[0]

        # Determine label
        label = 'Mask' if pred[1] > pred[0] else 'No Mask'

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Update FPS counter
    fps.update()

# Release resources
cap.release()
out.release()
fps.stop()
cv2.destroyAllWindows()