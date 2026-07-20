import tensorflow as tf

# Load Keras model (NOT SavedModel)
model = tf.keras.models.load_model("Training_Code_TensorFlow/best_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("Training_Code_TensorFlow/final_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")