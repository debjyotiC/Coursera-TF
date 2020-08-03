import tensorflow as tf
import numpy as np
from keras.preprocessing import image

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="saved_model/tflite_model/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# predicting images
path = 'content/animal.jpg'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
interpreter.set_tensor(input_details[0]['index'], images)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

if output_data[0] > 0.5:
    print("is a human")
else:
    print("is a horse")

print(input_details)
