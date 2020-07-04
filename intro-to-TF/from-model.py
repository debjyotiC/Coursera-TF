import os
import tensorflow as tf
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

load_model = tf.keras.models.load_model('C:\\Users\\Deb\\PycharmProjects\\Coursera-TF\\'
                                       'intro-to-TF\\saved_model\\my_model')

# Check its architecture
load_model.summary()

# predicting images
path = 'C:\\Users\\Deb\\PycharmProjects\\Coursera-TF\\intro-to-TF\\content\\animal.jpg'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = load_model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print("is a human")
else:
    print("is a horse")