import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

# Load YOLO model
def load_yolo_model(cfg_path, weights_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Use CUDA if available
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # Use CUDA if available
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
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

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
cfg_path = "D:/ASL_Alphabet_Dataset/Sign_language/SignLanguageToText/cross-hands.cfg"
weights_path_yolo = "D:/ASL_Alphabet_Dataset/Sign_language/SignLanguageToText/cross-hands.weights"
weights_path_asl = "D:/ASL_Alphabet_Dataset/Sign_language/SignLanguageToText/asl_to_text_model_74.pth"

# Class labels
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space"]

# Load models
yolo_net = load_yolo_model(cfg_path, weights_path_yolo)
asl_model, device = load_asl_model(weights_path_asl)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hands using YOLO
    detections = detect_hands_yolo(frame, yolo_net)

    # Process each detected hand
    for (box, confidence) in detections:
        (x, y, w, h) = box
        cropped_frame = frame[y:y + h, x:x + w]

        # Preprocess for ASL model
        asl_input = preprocess_image(cropped_frame).to(device)

        # Predict ASL sign
        with torch.no_grad():
            output = asl_model(asl_input)
            _, predicted_class = torch.max(output, 1)

        # Draw bounding box and label on the frame
        label = class_names[predicted_class.item()]
        print(label)  # Print detected letter to the terminal
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("ASL Hand Detection and Classification", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
