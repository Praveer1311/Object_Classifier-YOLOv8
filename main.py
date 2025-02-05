import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")

# Define a dictionary of colors for different labels
colors = {}

def get_color(label):
    """Assign a unique color to each label."""
    if label not in colors:
        import random
        colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return colors[label]

# Load input image
image_path = "input3.jpg"  # Change to your image path
image = cv2.imread(image_path)

# Perform inference
results = model(image)[0]

# Draw bounding boxes on the image
for box in results.boxes.data:
    x1, y1, x2, y2, score, class_id = box.tolist()
    label = results.names[int(class_id)]
    color = get_color(label)
    
    # Draw bounding box with a thicker border
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    
    # Display label and confidence score with background
    text = f"{label}: {score:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_offset_x, text_offset_y = int(x1), int(y1) - 10

    # Draw filled rectangle as background for text
    cv2.rectangle(image, (text_offset_x, text_offset_y - text_height - 5),
                  (text_offset_x + text_width + 5, text_offset_y + 5), color, -1)

    # Put text on top of the rectangle
    cv2.putText(image, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 0, 0), 4, cv2.LINE_AA)

# Save and show the output image
output_path = "output3.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
