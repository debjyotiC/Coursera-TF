import os
import datetime
from flask import Flask, render_template, request, session
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

app = Flask(__name__)
app.secret_key = os.urandom(24)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = 'static\\uploaded_images'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_file():
    return render_template('index.html', uploaded_image='default-image.png', classification="unclassified")


@app.route('/uploader', methods=['GET', 'POST'])
def image_upload():
    current_dt = datetime.datetime.now()
    if request.method == 'POST':
        f = request.files['file']
        session['logged_in'] = True
        filename = 'uploaded_{date_0}_{time_0}.jpg'.format(date_0=current_dt.strftime('%Y-%m-%d'),
                                                           time_0=current_dt.strftime('%H-%M-%S-%f'))
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        session['path'] = filename
        return render_template('index.html', uploaded_image=filename, classification="unclassified")


@app.route('/classify', methods=['GET', 'POST'])
def image_classify():
    from_path = session['path']
    load_model = tf.keras.models.load_model('saved_model\\my_model')
    img = image.load_img('static\\uploaded_images\\'+from_path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = load_model.predict(images, batch_size=10)
    session['path'] = 'None'
    if classes[0] > 0.5:
        return render_template('index.html', uploaded_image='default-image.png', classification="is a human")
    else:
        return render_template('index.html', uploaded_image='default-image.png', classification="is a horse")


if __name__ == '__main__':
    app.run(host='192.168.0.106', debug=True)