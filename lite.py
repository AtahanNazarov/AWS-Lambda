import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from PIL import Image
import requests

# loading model
model_path = '/var/task/dino-vs-dragon-v2.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

target_size = (150, 150)

def load_image(url, target_size):
    # getting image
    response = requests.get(url)
    image_data = response.content
    image = Image.open(BytesIO(image_data))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # preprocessing image
    image = image.resize(target_size, Image.NEAREST)
    x = np.array(image) / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)

    return x

def predict(image_data, interpreter, input_index, output_index):
    interpreter.set_tensor(input_index, image_data)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_index)

    class_labels = {0: "dino", 1: "dragon"}

    predicted_class = 1 if pred[0] >= 0.5 else 0
    predicted_label = class_labels[predicted_class]

    return {predicted_label: pred[0].tolist()}

def lambda_handler(event, context):
    url = event['url']

    image_data = load_image(url, target_size)

    result = predict(image_data, interpreter, input_index, output_index)
    
    return result