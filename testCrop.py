import cv2
import torch
from ultralytics import YOLO
import os

def detect_and_crop_faces(image_path, model_path, output_dir):
    """
    Detect faces in an image using YOLOv8 and crop individual faces
    
    Args:
    image_path (str): Path to the input image
    model_path (str): Path to the pre-trained YOLOv8 face detection weights
    output_dir (str): Directory to save cropped faces
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pre-trained YOLOv8 face detection model
    model = YOLO(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Perform face detection
    results = model(image)
    
    # Process each detected face
    for i, result in enumerate(results):
        # Convert result to numpy array for OpenCV processing
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        
        # Crop and save each detected face
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Extract face
            face = image[y1:y2, x1:x2]
            
            # Generate output filename
            output_filename = os.path.join(
                output_dir, 
                f"face_{i}_{j}_{os.path.basename(image_path)}"
            )
            
            # Save the cropped face
            cv2.imwrite(output_filename, face)
            
            print(f"Saved face {j} from {image_path}")
    
    return len(boxes)

def main():
    # Example usage
    image_path = '/Users/aashiqmahato/Downloads/Face detection api/test.jpg'
    model_path = 'yolov8_face.pt'
    output_dir = 'detected_faces'
    
    # Detect and crop faces
    num_faces = detect_and_crop_faces(image_path, model_path, output_dir)
    
    print(f"Total faces detected: {num_faces}")

if __name__ == "__main__":
    main()

# Additional requirements:
# pip install ultralytics opencv-python torch