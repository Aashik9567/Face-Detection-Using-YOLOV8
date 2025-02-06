import cv2
import numpy as np  # Make sure numpy is imported
from ultralytics import YOLO
from PIL import Image
import io
import requests
import os  # For creating directories


# Load the trained YOLOv8 model (change the path to your model's path)
model = YOLO('yolov8_face.pt')  # Replace with your model's path

# Function to download the image from the URL
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

# Function to crop faces from the image
def crop_faces(image, detections):
    cropped_faces = []
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped_face = image.crop((x1, y1, x2, y2))
        cropped_faces.append(cropped_face)
    return cropped_faces

# Main processing
def process_image(image_url, output_dir="cropped_faces"):
    try:
        # Download the image
        image = download_image(image_url)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format for YOLO
        
        # Perform inference (face detection)
        results = model(image_np)
        
        # Ensure output directory exists
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Crop and save detected faces
        faces = crop_faces(image, results[0])
        for i, face in enumerate(faces):
            save_path = os.path.join(output_dir, f"face_{i + 1}.jpg")
            face.save(save_path)
            print(f"Saved cropped face to {save_path}")
        
        print(f"Total faces detected: {len(faces)}")
    
    except Exception as e:
        print(f"Error: {e}")

# Example usage
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1zqQ4iXMBvPLom6XxSfZ-NBvzZMuk_kM2bQ&s"  # Replace with your image URL
process_image(image_url)
