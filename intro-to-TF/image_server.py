import os
import datetime
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

app = Flask(__name__)

current_dt = datetime.datetime.now()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'uploaded_images'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def classify(path):
    load_model = tf.keras.models.load_model('saved_model\\my_model')
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = load_model.predict(images, batch_size=10)
    return classes[0]


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        f = request.files['file']
        filename = 'uploaded_{date_0}_{time_0}.jpg'.format(date_0=current_dt.strftime('%Y-%m-%d'),
                                                           time_0=current_dt.strftime('%H-%M-%S'))
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        got = classify(path)
        if got > 0.5:
            return "is a human"
        else:
            return "is a horse"


if __name__ == '__main__':
    app.run(debug=True)
