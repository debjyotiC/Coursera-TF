import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
tflite_model = converter.convert()
open("saved_model/tflite_model/converted_model.tflite", "wb").write(tflite_model)