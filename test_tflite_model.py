import tensorflow as tf
import numpy as np
from PIL import Image
import json

interpreter = tf.lite.Interpreter(model_path="rice_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('class_indices.json', 'r') as f:
    class_labels = {v: k for k, v in json.load(f).items()}

image_path = "C:/Users/FAHEEM/Documents/tensorflow/dataset/B1_yani_Adhwar/IMG-20241129-WA0004.jpg"  
image = Image.open(image_path).resize((224, 224))
image_array = np.array(image) / 255.0  
image_array = np.expand_dims(image_array, axis=0).astype(np.float32) 

interpreter.set_tensor(input_details[0]['index'], image_array)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class_index = np.argmax(output_data)

predicted_label = class_labels[predicted_class_index]

print(f"Predicted class index: {predicted_class_index}")
print(f"Predicted class label: {predicted_label}")
