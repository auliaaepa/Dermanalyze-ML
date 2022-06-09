import os
import cv2
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(allow_abbrev=False, description="Skin Cancer Prediction")
parser.add_argument('image', type=str, action="extend", nargs="+", help="the image path you want to predict, can be more than one")
parser.add_argument('--model', type=str, default="non_quantized_model_1.0.0.tflite", help='the tflite model path')

args = parser.parse_args()
images_path = args.image
model_path = args.model

labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'unk']

for image_path in images_path:
    print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_dtype = img.dtype

    # apply shade of gray color constancy
    img = img.astype('float32')
    img_power = np.power(img, 6)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/6)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1 / (rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    img = np.clip(img, a_min=0, a_max=255)
    img = img.astype(img_dtype)

    # convert to tensor
    img_tensor = tf.convert_to_tensor(img)
    # resize into input size (256, 256)
    img_tensor = tf.image.resize(img_tensor, (256, 256)) 
    # rescale img to [0, 1]
    img_tensor = tf.cast(img_tensor, dtype=tf.float32) / tf.constant(256, dtype=tf.float32)
    # add dimension for batch
    input_data = tf.expand_dims(img_tensor, axis=0) 

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    tflite_results = interpreter.get_tensor(output_details[0]["index"])
    tflite_probs = tflite_results / tf.constant(256, dtype=tf.float32)
    index = np.argmax(tflite_probs)
    pred_label = labels[index]
    confidence = tflite_probs[0][index] * 100
    print(f"{image_path}: {pred_label} with {confidence:.2f}% probability")