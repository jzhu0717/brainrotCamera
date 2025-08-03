# packages installed: opencv-python, boto3
# pip install (package name) 
import cv2
import boto3
import time
import credentials

# create AWS rekognition client
reko_client = boto3.client('rekognition', 
                           aws_access_key_id = credentials.access_Key, 
                           aws_secret_access_key = credentials.secret_Key, 
                           region_name='us-east-1')

# set target class
# target_class = 'Person'
glizzy = {"Hot Dog", "Hotdog", "Hot-Dog", "Sausage"}


# load video (camera capture)

# Use 0 for webcam or replace with video file path
video_source = 0  
video_capture = cv2.VideoCapture(video_source)

# check if video capture is opened
if not video_capture.isOpened():
    print("Error: Could not open video source.")
    exit()

last_check = 0
check_interval = 2  # seconds between Rekognition calls to save API costs


ret = True
while ret:
    ret, frame = video_capture.read()


    if not ret:
        print("Error: Could not read frame.")
        break

    # open window
    cv2.imshow('Live webcam', frame)
    
    # Run every n seconds
    if time.time() - last_check >= check_interval:
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        res = reko_client.detect_labels(
            Image={'Bytes': image_bytes},
            MinConfidence=95) # this is the confidence threshold

        # all objects (temporary)
        detected = [label['Name'] for label in res['Labels']]                
        print("Objects:", detected)  
        
        # checking for glizzy
        for label in res['Labels']:
            
            if label['Name'] in glizzy: # if glizzy present
                print("glizzy detected")
                
                # bounding boxes
                w, h, _ =  frame.shape
                for i in label["Instances"]:
                    bbox = i['BoundingBox']
                    x1 = int(bbox['Left'] * w)
                    y1 = int(bbox['Top'] * h)
                    x2 = int((bbox['Left'] + bbox['Width']) * w)
                    y2 = int((bbox['Top'] + bbox['Height']) * h)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "glizzy", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            last_check = time.time()
        
        # display with boxes
        cv2.imshow('Live webcam', frame)

    # exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
# write detections