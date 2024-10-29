import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import base64
import sys
import json

# Load YOLO model
def load_yolo_model(cfg_path, weights_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Use OpenCV backend
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU
    return net

# Detect hands using YOLO
def detect_hands_yolo(frame, net):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    results = [(boxes[i], confidences[i]) for i in indices.flatten()] if len(indices) > 0 else []
    return results

# Load ASL classification model
def load_asl_model(weights_path, num_classes=27):
    model = models.mobilenet_v2(pretrained=True)  # Load pretrained weights
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess image for ASL model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(cropped_frame):
    image = Image.fromarray(cropped_frame)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Paths
cfg_path = "D:/ASL_Alphabet_Dataset/Sign_language/cross-hands.cfg"
weights_path_yolo = "D:/ASL_Alphabet_Dataset/Sign_language/cross-hands.weights"
weights_path_asl = "D:/ASL_Alphabet_Dataset/Sign_language/SignLanguageToText/asl_to_text_model_74.pth"

# Class labels
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space"]

# Load models
yolo_net = load_yolo_model(cfg_path, weights_path_yolo)
asl_model = load_asl_model(weights_path_asl)

# Processing function for incoming frames
def process_frame(frame_data):
    try:
        # Decode base64 frame
        frame_data = base64.b64decode(frame_data)
        np_img = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Resize the frame to reduce memory usage
        frame = cv2.resize(frame, (640, 480))

        # Detect hands with YOLO
        detections = detect_hands_yolo(frame, yolo_net)

        boxes = []  # List to store bounding box coordinates

        # Process each detected hand
        for (box, confidence) in detections:
            x, y, w, h = box
            
            # Ensure the bounding box is within frame dimensions
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue  # Skip this box if it goes out of bounds

            cropped_frame = frame[y:y+h, x:x+w]

            if cropped_frame.size == 0:
                continue  # Skip if the cropped frame is empty

            asl_input = preprocess_image(cropped_frame)

            # Predict ASL sign
            with torch.no_grad():
                output = asl_model(asl_input)
                _, predicted_class = torch.max(output, 1)

            label = class_names[predicted_class.item()]

            # Append the coordinates and label to boxes
            boxes.append({
                'box': [x, y, w, h],
                'label': label,
                'confidence': confidence
            })

        # Send only the box coordinates and labels to the frontend
        return json.dumps(boxes)  # Return as JSON string

    except Exception as e:
        print(f"Error processing frame: {e}", file=sys.stderr)
        return None


if __name__ == '__main__':
    while True:
        # Read frame data from stdin
        frame_data = sys.stdin.readline().strip()
        if frame_data:
            bounding_boxes = process_frame(frame_data)
            if bounding_boxes:
                # Write bounding boxes back to stdout
                print(bounding_boxes)
                sys.stdout.flush()
