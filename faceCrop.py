import cv2
import os
import numpy as np
from ultralytics import YOLO

class FaceCropper:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)
    
    def crop_faces(self, input_folder, output_folder, confidence_threshold=0.79):
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Counter for unique filenames
        face_counter = 0
        
        # Iterate through input images
        for filename in os.listdir(input_folder):
            image_path = os.path.join(input_folder, filename)
            
            # Skip non-files
            if not os.path.isfile(image_path):
                print(f"Skipping non-file: {image_path}")
                continue
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image: {image_path}")
                continue
            
            # Detect faces using YOLO
            results = self.model(image)
            
            # Process each detected face
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Check confidence threshold
                    confidence = box.conf[0]
                    if confidence < confidence_threshold:
                        print(f"Face skipped due to low confidence: {confidence}")
                        continue
                    
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Validate bounding box dimensions
                    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                        print(f"Invalid bounding box: {x1}, {y1}, {x2}, {y2}")
                        continue
                    
                    # Crop face
                    face_crop = image[y1:y2, x1:x2]
                    
                    # Validate crop
                    if face_crop.size > 0:
                        # Resize to standard size
                        face_crop = cv2.resize(face_crop, (224, 224))
                        
                        # Generate unique filename with confidence
                        face_filename = f"face_{face_counter}_conf_{confidence:.2f}.jpg"
                        face_path = os.path.join(output_folder, face_filename)
                        
                        # Save cropped face
                        cv2.imwrite(face_path, face_crop)
                        
                        face_counter += 1
        
        print(f"Processed {face_counter} faces with confidence >= {confidence_threshold}")

# Rest of the code remains the same as in the original script
def organize_faces_by_person():
    cropper = FaceCropper('yolov8_face.pt')
    
    # Input folders for different persons
    person_folders = {
       
        'Bhawana': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/bhawana j',
        'Rensa': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/rensa j',
        'Mandeep': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/mandeep j',
        'Anjita': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/anjita j',
        'Binisha': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/binisha j',
        'Bipin': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/bipin j',
        'Prabhas': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/prabhas j',
        'Muskan': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/muskan j',
        'Norgen': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/norgen j',
        'Darwin': '/Users/aashiqmahato/Downloads/project/Face detection api/dataset/darwin j',
    }
    
    # Process each person's images
    for person_name, input_folder in person_folders.items():
        output_folder = os.path.join('cropped_faces', person_name)
        cropper.crop_faces(input_folder, output_folder)

# Validate face quality
def validate_face_quality(image_path, min_size=100):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image: {image_path}")
        return False
    
    height, width = image.shape[:2]
    if height < min_size or width < min_size:
        print(f"Image too small: {image_path}")
        return False
    
    if not is_well_lit(image):
        print(f"Image not well-lit: {image_path}")
        return False
    
    if not has_good_contrast(image):
        print(f"Image lacks contrast: {image_path}")
        return False
    
    return True

def is_well_lit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return 100 < brightness < 200

def has_good_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    return contrast > 30

# Preprocess faces for recognition
def preprocess_faces(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for root, _, files in os.walk(input_folder):
        for filename in files:
            input_path = os.path.join(root, filename)
            
            # Validate face quality
            if validate_face_quality(input_path):
                image = cv2.imread(input_path)
                
                # Resize to standard size
                resized = cv2.resize(image, (224, 224))
                
                # Normalize pixel values
                normalized = resized / 255.0
                
                # Save preprocessed image
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                
                output_path = os.path.join(output_subfolder, filename)
                cv2.imwrite(output_path, (normalized * 255).astype(np.uint8))
            else:
                print(f"Skipping invalid face: {input_path}")

# Main execution
if __name__ == '__main__':
    organize_faces_by_person()
    preprocess_faces('cropped_faces', 'processed_faces')
