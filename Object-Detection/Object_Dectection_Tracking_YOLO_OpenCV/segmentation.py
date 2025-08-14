import cv2
from ultralytics import YOLO
import numpy as np
import os

cwd = os.getcwd()  # Get the current working directory

video_path = os.path.join(cwd, "data/raw_data/videos/cars.mp4")  # path to the video
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found at {video_path}")

model = YOLO("yolov8n-seg.pt")  # Load a pre-trained YOLOv8 segmentation model

cap = cv2.VideoCapture(video_path)  # Open the video file

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video path and codec
output_path = os.path.join(cwd, "data/annotated_data/videos/segmentation_tracking_cars.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit the loop if no frame is read

    results = model.track(source=frame, persist=True, verbose=False)  # Perform tracking on the frame for people and bicycles
    for r in results:
        annotated_frame = frame.copy()  # Create a copy of the frame for annotation
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            masks = r.masks.data.numpy()  # Get the masks of the detected objects
            boxes = r.boxes.xyxy.numpy()  # Get the bounding boxes of the detected objects
            ids = r.boxes.id.numpy()  # Get the IDs of the detected objects

            for i, mask in enumerate(masks):
                person_id = ids[i]
                x1, y1, x2, y2 = boxes[i].astype(int)
                mask_resized = cv2.resize(mask.astype(np.uint8)*255, (frame.shape[1], frame.shape[0]))
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"ID: {person_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to output video
        out.write(annotated_frame)
        
        cv2.imshow("Object Tracking with Segmentation", annotated_frame)  # Display the annotated frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()  # Release the video capture object
out.release()  # Release the video writer object
cv2.destroyAllWindows()  # Close all OpenCV windows

print(f"Annotated video saved to: {output_path}")
