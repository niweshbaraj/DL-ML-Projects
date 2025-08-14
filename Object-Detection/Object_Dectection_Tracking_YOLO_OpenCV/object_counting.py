import cv2
from ultralytics import YOLO
import numpy
import os

cwd = os.getcwd()  # Get the current working directory

video_path = os.path.join(cwd, "data/raw_data/videos/bottle1.mp4")  # path to the video
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found at {video_path}")

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model

cap = cv2.VideoCapture(video_path)  # Open the video file

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video path and codec
output_path = os.path.join(cwd, "data/annotated_data/videos/bottle1_counting.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

unique_ids = set()  # Set to store unique object IDs

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit the loop if no frame is read

    results = model.track(frame, classes=[39], persist=True, verbose=False)  # Perform tracking on the frame for class 'bottle'

    annotated_frame = results[0].plot()  # Get the annotated frame with detections

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()  # Get the IDs of the detected objects
        for oid in ids:
            unique_ids.add(oid)
        cv2.putText(annotated_frame, f"Count: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Write the frame to output video
    out.write(annotated_frame)
    
    cv2.imshow("Object Tracking", annotated_frame)  # Display the annotated frame
    
    # Wait for a key press and break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
out.release()  # Release the video writer object
cv2.destroyAllWindows()  # Close all OpenCV windows

print(f"Annotated video saved to: {output_path}")
