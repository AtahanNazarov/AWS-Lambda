import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model
import tensorflow.lite as tflite

model = load_model('dino_dragon_10_0.899.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('C:\AWS-Lambda\models\lite_model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

interpreter = tflite.Interpreter(model_path='C:\AWS-Lambda\models\lite_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

output_index