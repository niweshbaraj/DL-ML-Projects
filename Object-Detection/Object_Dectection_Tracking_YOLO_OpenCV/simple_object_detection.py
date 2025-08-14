import os
import cv2
from ultralytics import YOLO

# This script performs object detection using a pre-trained YOLOv8 model on an image.

cwd = os.getcwd()  # Get the current working directory

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model

image_path = os.path.join(cwd, "data/raw_data/images/image1.jpg")  # Construct the path to the image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")

image = cv2.imread(image_path)  # Load an image from file

results = model(image)  # Perform inference on the image

annotated_image = results[0].plot()  # Get the annotated image with detections

cv2.imshow("Annotated Image", annotated_image)  # Display the annotated image
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the annotated image to a file
save_dir = os.path.join(cwd, "data/annotated_data/images")  # Specify the directory to save the annotated image
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create the directory if it doesn't exist
annotated_img_path = os.path.join(save_dir, "annotated_image1.jpg")
cv2.imwrite(annotated_img_path, annotated_image)