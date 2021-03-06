import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

load_model = tf.keras.models.load_model('saved_model/tf_model')

# Check its architecture
load_model.summary()

# predicting images
path = 'content/boy-4.jpg'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = load_model.predict(images, batch_size=10)
print(classes)
if classes[0] > 0.5:
    print("is a human")
else:
    print("is a horse")
