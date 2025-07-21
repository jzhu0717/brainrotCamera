# packages installed: opencv-python, boto3
# pip install (package name) 
import cv2
import boto3

import credentials

# create AWS rekognition client
reko_client = boto3.client('rekognition', 
                           aws_access_key_id = credentials.access_Key, 
                           aws_secret_access_key = credentials.secret_Key, 
                           region_name='us-east-1')

# set target class
target_class = 'Person'

# load video (camera capture)

# Use 0 for webcam or replace with video file path
video_source = 0  
video_capture = cv2.VideoCapture(video_source)

# check if video capture is opened
if not video_capture.isOpened():
    print("Error: Could not open video source.")
    exit()

ret = True
while ret:
    ret, frame = video_capture.read()

    # convert to jpg
    _, buffer = cv2.imencode('.jpg', frame) 

    # convert buffer to bytes
    image_bytes = buffer.tobytes()
    break

    # detect object
    res = reko_client.detect_labels(
        Image={'Bytes': image_bytes},
        MinConfidence=50)
    

# write detections