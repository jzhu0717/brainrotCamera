# packages installed: opencv-python, boto3, opencv-contrib-python
# pip install <package name> 
import cv2
import boto3
import time
import credentials

# create AWS rekognition client
reko_client = boto3.client('rekognition', 
                           aws_access_key_id = credentials.access_Key, 
                           aws_secret_access_key = credentials.secret_Key, 
                           region_name='us-east-1')

# set "target class"
glizzy = {"Hot Dog", "Hotdog", "Hot-Dog", "Sausage"}


# load video (camera capture)
# Use 0 for webcam or replace with video file path
video_source = 0  
video_capture = cv2.VideoCapture(video_source)

# check if video capture is opened
if not video_capture.isOpened():
    print("Error: Could not open video source.")
    exit()

# NOTE: TRY TO FIND INTERVAL / REFRESH RATE THAT RESULTS IN SMOOTHEST
last_check = 0
check_interval = 1  # tick interval (s). SENDS CALLS TO AWS
tracker = None
tracking = False
last_seen = 0
misses = 0
max_no_refresh = 0.5  # seconds before giving up if no AWS detection

print("CAMERA READY. Press ESC to exit.")
ret = True
while ret:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # open window
    # cv2.imshow('Live webcam', frame)
    
    w, h, _ =  frame.shape

    if tracking and tracker is not None:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # labels
            cv2.putText(frame, "glizzy", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            tracking = False
            tracker = None

 
    # Run every n seconds
    if time.time() - last_check >= check_interval:
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        res = reko_client.detect_labels(
            Image = {'Bytes': image_bytes},
            MinConfidence = 90) # this is the confidence threshold

        # all objects (temporary)
        # detected = [label['Name'] for label in res['Labels']]                
        # print("Objects:", detected)  
        
        boxes = []
        glizzy_detected = False
        # checking for if glizzy present
        for label in res['Labels']:
            if label['Name'] in glizzy:
                if label['Instances']:          
                    bbox = label['Instances'][0]['BoundingBox']
                    x1 = int(bbox['Left'] * w)
                    y1 = int(bbox['Top'] * h)
                    bw = int(bbox['Width'] * w)
                    bh = int(bbox['Height'] * h)
                    
                    if not tracking:
                        # only initialize/reinit if not tracking
                        tracker = cv2.legacy.TrackerKCF_create()
                        tracker.init(frame, (x1, y1, bw, bh))
                        tracking = True
                        print("glizzy detected")
                    
                    last_seen = time.time()
                    misses = 0
                    glizzy_detected = True
                    break
        
        if not glizzy_detected:
            misses += 1
            if misses > 3:
                tracking = False
                tracker = None
                if misses == 4:
                    print("no more glizzyt")
        # cache_boxes = boxes
        last_check = time.time()
        
    # # If we've gone too long without AWS confirming, stop tracking
    # if tracking and (time.time() - last_seen > max_no_refresh):
    #     tracking = False
    #     tracker = None
    #     print("no more glizzyt")
    
    # display window
    cv2.imshow('Live webcam', frame)

    # exit if 'esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27: # ESC key
        print("Exiting...")
        break
    
video_capture.release()
cv2.destroyAllWindows()
