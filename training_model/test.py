import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2

# Load the trained model with .keras extension
model = load_model(r"D:\ASL_Alphabet_Dataset\Sign_language\SignLanguageToText\training_model\asl_model_epoch.keras")
imageSize = 75

# Character mapping (assuming you have a dictionary mapping characters to indices)
map_characters = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 
    23: 'X', 24: 'Y', 25: 'Z', 26: 'space'
}

# Function to preprocess the input image
def preprocess_image(img_path):
    img = cv2.imread(img_path)  # Load the image
    img = cv2.resize(img, (imageSize, imageSize))  # Resize to match model input
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to translate an image to an ASL character
def translate_image_to_asl(img_path):
    # Preprocess the image
    processed_image = preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability

    # Map the predicted class index to the corresponding character
    translated_character = map_characters[predicted_class]

    return translated_character

# Example usage
image_path = r"D:\ASL_Alphabet_Dataset\p.jpg"  # Specify the path to your input image
translated_character = translate_image_to_asl(image_path)
print(f'Translated ASL Character: {translated_character}')
