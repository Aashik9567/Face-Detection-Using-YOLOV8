import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io

class FaceDetector:
    def __init__(self, model_path='yolov8_face.pt'):
        """
        Initialize the face detection model
        
        :param model_path: Path to the YOLOv8 model weights
        """
        self.model = YOLO(model_path)
    
    def detect_faces(self, image):
        """
        Detect faces in the given image
        
        :param image: PIL Image or numpy array
        :return: List of detected face bounding boxes
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run inference
        results = self.model(image)
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0]
                class_id = box.cls[0]
                if class_id == 0:  # Assuming 0 is the face class
                    faces.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence)
                    })
        return faces

    def draw_faces(self, image, faces):
        """
        Draw bounding boxes around detected faces
        
        :param image: PIL Image
        :param faces: List of detected faces
        :return: PIL Image with bounding boxes
        """
        draw = ImageDraw.Draw(image)
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
        return image

    def crop_faces(self, image, faces):
        """
        Crop detected faces from the image
        
        :param image: PIL Image
        :param faces: List of detected faces
        :return: List of cropped face images
        """
        cropped_faces = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            # Crop the face from the original image
            cropped_face = image.crop((x1, y1, x2, y2))
            cropped_faces.append(cropped_face)
        return cropped_faces