from collections import defaultdict, deque
import cv2
from ultralytics import YOLO
import numpy as np
import os

cwd = os.getcwd()  # Get the current working directory

video_path = os.path.join(cwd, "data/raw_data/videos/people_walking.mp4")  # path to the video
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found at {video_path}")

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model

cap = cv2.VideoCapture(video_path)  # Open the video file

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video path and codec
output_path = os.path.join(cwd, "data/annotated_data/videos/tracking_with_trail_people.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

id_map = {}  # Dictionary to map object IDs to unique IDs
next_id = 0  # Counter for unique IDs

trail = defaultdict(lambda: deque(maxlen=30))  # Dictionary to store trails of objects
appear = defaultdict(int)  # Dictionary to count appearances of objects

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit the loop if no frame is read

    results = model.track(frame, classes=[0, 1], persist=True, verbose=False)  # Perform tracking on the frame for people and bicycles
    # results = model.track(frame, classes=[2], persist=True, verbose=False)  # Perform tracking on the frame for cars

    annotated_frame = frame.copy()  # Create a copy of the frame for annotation

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.numpy()
        ids = results[0].boxes.id.numpy()  # Get the IDs of the detected
        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            appear[oid] += 1  # Increment appearance count for the object

            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cv2.putText(annotated_frame, f"ID: {sid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Draw the trail
                points = list(trail[oid])
                if len(points) > 1:
                    for i in range(1, len(points)):
                        # Draw lines connecting consecutive points in the trail
                        cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 255), 2)

    # Write the frame to output video
    out.write(annotated_frame)
    
    cv2.imshow("Object Tracking", annotated_frame)  # Display the annotated frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()  # Release the video capture object
out.release()  # Release the video writer object
cv2.destroyAllWindows()  # Close all OpenCV windows

print(f"Annotated video saved to: {output_path}")