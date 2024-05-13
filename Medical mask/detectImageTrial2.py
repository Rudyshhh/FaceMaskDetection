# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import numpy as np
img_size = 224

def mask_image():
	# construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,help="path to input image")
	# # ap.add_argument("-f", "--face", type=str,
	# # 	default="face_detector",
	# # 	help="path to face detector model directory")
	# # ap.add_argument("-m", "--model", type=str,
	# # 	default="mask_detector.model",
	# # 	help="path to trained face mask detector model")
	# # ap.add_argument("-c", "--confidence", type=float, default=0.5,
	# # 	help="minimum probability to filter weak detections")
    # args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	
	
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector5.model")
    image = cv2.imread("Medical Mask/images/0004.jpg")
    orig = image.copy()
    assign = {'0':'Mask','1':"No Mask"}
    (h, w) = image.shape[:2]
    
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        	# extract the confidence (i.e., probability) associated with
		# the detection
       box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
       (startX, startY, endX, endY) = box.astype("int")
       frame = image[startY:endY, startX:endX]
       confidence = detections[0, 0, i, 2]
       if confidence > 0.2:
                im = cv2.resize(frame,(img_size,img_size))
                im = np.array(im)/255.0
                im = im.reshape(1,224,224,3)
                result = model.predict(im)
                if result>0.5:
                    label_Y = 1
                else:
                    label_Y = 0
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image,assign[str(label_Y)] , (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)
     
     # show the output image       
    cv2.imshow("Output", image)
    cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()