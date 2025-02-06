
# Import necessary libraries
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8_face.pt")  # Use your YOLOv8 model path if different

# Load the image
image_path = ""  # Replace with the path to your image
image = cv2.imread(image_path)

# Run YOLOv8 inference with a confidence threshold
results = model(image, conf=0.5, verbose=False)  # Set confidence threshold to 0.5

# Parse results and draw bounding boxes
for result in results:
    for box in result.boxes.data:
        # Extract bounding box coordinates and confidence
        x1, y1, x2, y2, confidence = map(float, box[:5].tolist())
        
        # Draw only if confidence is above the threshold
        if confidence >= 0.5:  # Minimum confidence threshold
            label = f"{result.names[int(box[5])]} {confidence:.2f}"  # Label with class and confidence
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save or display the resulting image
output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"Result saved to {output_path}")

# Optionally, display the image in a Jupyter notebook
from matplotlib import pyplot as plt
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
