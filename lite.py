import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from PIL import Image
import requests

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url, target_size=(150,150)):
    response = requests.get(url)
    image_data = response.content
    image = Image.open(BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size, Image.NEAREST)

    x = np.array(image)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    
    class_labels = {0: 'dino', 1: 'dragon'}
    predicted_class = 1 if pred >= 0.5 else 0 
    predicted_label = class_labels[predicted_class]

    print(predicted_label)


url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'
predict(url, target_size=(150,150))



