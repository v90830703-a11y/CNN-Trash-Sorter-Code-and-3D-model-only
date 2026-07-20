import tensorflow as tf

model = tf.keras.models.load_model("Training_Code_TensorFlow/best_model.keras")

def rep_data():
    for _ in range(100):
        yield [tf.random.uniform([1, 32, 32, 1])]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

def rep_data():
    for _ in range(100):
        yield [tf.random.uniform([1, 32, 32, 1])]

converter.representative_dataset = rep_data

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
open("Training_Code_TensorFlow/model_int8.tflite", "wb").write(tflite_model)