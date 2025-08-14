import os
import cv2
from ultralytics import YOLO

cwd = os.getcwd()  # Get the current working directory

video_path = os.path.join(cwd, "data/raw_data/videos/cars.mp4")  # Construct the path to the video
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found at {video_path}")

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model

cap = cv2.VideoCapture(video_path)  # Open the video file

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video path and codec
output_path = os.path.join(cwd, "data/annotated_data/videos/multi_object_detection_cars.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit the loop if no frame is read

    results = model(frame)  # Perform inference on the frame

    # Optionally filter results for specific classes (e.g., person, car, bicycle)

    # results = model(frame, class=[0, 1, 2])  # Perform inference on the frame for specific classes (e.g., person, car, bicycle)

    annotated_frame = results[0].plot()  # Get the annotated frame with detections

    # Write the frame to output video
    out.write(annotated_frame)

    cv2.imshow("Annotated Video", annotated_frame)  # Display the annotated frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()  # Release the video capture object
out.release()  # Release the video writer object
cv2.destroyAllWindows()  # Close all OpenCV windows

print(f"Annotated video saved to: {output_path}")
