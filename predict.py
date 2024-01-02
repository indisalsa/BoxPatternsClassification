import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Load the saved model
model = tf.keras.models.load_model('C:/Users/asus/PycharmProjects/MagangWingsFood/ResNet50V2_Dense512_KFold10_60ml_20.model')
# model.summary()

# Preprocess the input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((160, 320)) # Resize the image to match the input size of the model
    img = np.array(img) / 255.0 # Normalize the pixel values to the range of [0, 1]
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return img

# Define the class labels
class_labels = ['Pola1_Benar','Pola1_Salah','Pola2_Benar','Pola2_Salah']  # Replace with your actual class labels

# Predict the brain tumor for a new image
image_path = 'C:/Users/asus/Downloads/KP PT. Bumi Alam Segar/Dataset/60/Pola1_Benar/IMG-20231024-WA0450.jpg'
# image_path = 'C:/Users/asus/Downloads/KP PT. Bumi Alam Segar/Dataset/60/Pola1_Salah/IMG-20231102-WA0017.jpg'
# image_path = 'C:/Users/asus/Downloads/KP PT. Bumi Alam Segar/Dataset/60/Pola2_Benar/IMG-20231024-WA0496.jpg'
# image_path = 'C:/Users/asus/Downloads/KP PT. Bumi Alam Segar/Dataset/60/Pola2_Salah/IMG-20231102-WA0033.jpg'
preprocessed_image = preprocess_image(image_path)

# Measure the prediction time
start_time = time.time()
predictions = model.predict(preprocessed_image)
end_time = time.time()
prediction_time = end_time - start_time

predicted_class_index = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class_index]
confidence = predictions[0][predicted_class_index]

print("Predicted class:", predicted_class_label)
print("Confidence:", confidence)
print("Prediction time:", prediction_time, "seconds")